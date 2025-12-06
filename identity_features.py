# identity_features.py
import numpy as np

# We'll use a subset of stable landmark indices: eyes, nose, mouth corners, chin etc.
# (468 total points in FaceMesh; we just pick some key ones)
KEY_IDXS = [
    1,   # nose tip
    33, 133,  # left/right eye outer corners
    263, 362, # right/left eye outer corners
    61, 291,  # mouth corners
    199,      # chin-ish
    4,        # between eyes
    94, 324,  # cheek-ish
]

def face_to_vector(face_landmarks):
    """
    Convert a FaceMesh landmarks object into a normalized feature vector.

    face_landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    Returns: 1D np.array of shape (len(KEY_IDXS)*2,) with normalized (x,y) coords
    """
    # Extract all (x,y) into array
    pts = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark], dtype=np.float32)

    # Use only selected key points
    key_pts = pts[KEY_IDXS]  # shape (K, 2)

    # Normalize: subtract center + divide by face "size" (std dev)
    center = key_pts.mean(axis=0, keepdims=True)
    centered = key_pts - center

    scale = centered.std() + 1e-6
    normalized = centered / scale

    # Flatten -> [x0,y0,x1,y1,...]
    return normalized.flatten()
