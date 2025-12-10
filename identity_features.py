# identity_features.py
# This file defines HOW we convert a detected face (468 MediaPipe FaceMesh landmarks)
# into a compact numeric feature vector that is suitable for machine learning.
#
# Instead of using all 468 points, we extract only a handful of stable reference
# landmarks (eyes, nose, mouth corners, etc.). These are enough to distinguish
# "Navid" from "other" without requiring a full face recognition model.

import numpy as np

# KEY_IDXS lists the indices of FaceMesh landmarks we want to keep.
# MediaPipe FaceMesh provides 468 facial landmarks numbered 0–467.
# We handpicked a small subset that captures the "shape" of the face:
#
# - Nose tip: highly distinctive, central reference point
# - Eye corners: relative spacing/orientation helps identify a person
# - Mouth corners: mouth width + position differs by identity
# - Chin point: overall face length
# - Center between eyes + cheek points: adds geometric structure
#
# These features are geometric, not pixel-based, so they're lighting-robust.

KEY_IDXS = [
    1,          # nose tip (very stable reference on the face)
    33, 133,    # left/right eye outer corners
    263, 362,   # right/left eye outer corners (opposite side)
    61, 291,    # left/right mouth corners
    199,        # chin-ish region (jaw length)
    4,          # between the eyes (central alignment point)
    94, 324,    # cheek-ish landmarks (face width cues)
]


def face_to_vector(face_landmarks):
    """
    Extract a numeric identity vector from MediaPipe's face landmarks.

    Parameters
    ----------
    face_landmarks :
        A NormalizedLandmarkList from MediaPipe FaceMesh.
        Contains ~468 face points as (x, y, z) normalized coordinates.

    Returns
    -------
    1D np.array
        Flattened vector [x0, y0, x1, y1, ..., xK, yK]
        where K = number of selected KEY_IDXS landmarks.

    WHY THIS VECTOR WORKS
    ---------------------
    - We use (x,y) landmark positions normalized relative to key points.
    - This captures the geometric "signature" of a face: ratios, spacing,
      face width, eye-mouth distances, etc.
    - This is identity-stable but lighting- and background-invariant.
    """

    # Convert all landmarks into a NumPy array of shape (468, 2)
    # Each row is [x, y], normalized coordinates in the image (0 → left/top, 1 → right/bottom)
    pts = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark], dtype=np.float32)

    # Select ONLY the important key points
    # Shape becomes (len(KEY_IDXS), 2)
    key_pts = pts[KEY_IDXS]

    # ------------------- NORMALIZATION STEP -------------------
    # The raw coordinates depend on camera distance and head movement.
    # If we compare raw (x,y)s, size differences would confuse the classifier.
    #
    # To remove scale + position differences:
    # 1) Find the centroid (average x,y) of selected key points.
    center = key_pts.mean(axis=0, keepdims=True)

    # 2) Subtract centroid so face is centered at (0,0)
    centered = key_pts - center

    # 3) Compute a "scale" from standard deviation (face spread)
    # Avoid divide-by-zero by adding a tiny epsilon.
    scale = centered.std() + 1e-6

    # 4) Divide coordinates by scale → face normalized to unit size
    normalized = centered / scale

    # ----------------------------------------------------------
    # The normalized coordinates give us a shape-based identity vector.
    # Example:
    # [x0, y0, x1, y1, ..., xN, yN]
    #
    # Flatten from shape (K, 2) into (2K,)
    return normalized.flatten()
