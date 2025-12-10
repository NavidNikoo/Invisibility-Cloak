# collect_identity_data.py
# This script builds the dataset for the identity model (Navid vs other).
# It uses MediaPipe FaceMesh to extract a compact numeric vector for each face,
# and saves those vectors to a CSV with labels "navid" or "other".

import cv2          # OpenCV: webcam capture, drawing, GUI window
import csv          # csv: read/write CSV files
import os           # os: file existence checks, paths
import mediapipe as mp  # MediaPipe: FaceMesh face landmark detector
from identity_features import face_to_vector  # my function: FaceMesh → numeric feature vector

# Where to save the identity dataset
CSV_PATH = "identity_data.csv"


def init_csv(path):
    """
    Prepare the CSV file:
    - If it does NOT exist, create it and write a temporary header ["label"].
    - If it already exists, just print a message that we will append to it.

    We do NOT know the feature dimension yet (length of vec),
    so we fix the full header later when we save the first sample.
    """
    if not os.path.exists(path):
        # First time: create file with a minimal header
        # (later we will overwrite this when we know the feature length).
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # temporary header; will become ["label", "f0", "f1", ..., "fN"]
            writer.writerow(["label"])
        print(f"[INFO] Created new CSV at {path}")
    else:
        # File already exists; we will just append more rows to that file.
        print(f"[INFO] Appending to existing CSV at {path}")


def append_sample(label, vec, path):
    """
    Append ONE labeled face sample to the CSV.

    Parameters
    ----------
    label : str
        "navid" or "other"
    vec : 1D np.array
        Feature vector produced by face_to_vector(...) for one face.
    path : str
        Path to the CSV file (identity_data.csv)
    """

    # 1) Check whether the file still has only the minimal header ["label"].
    #    If so, rewrite the header to include the full feature column names.
    need_header_fix = False
    with open(path, "r") as f:
        first_line = f.readline().strip()
        if first_line == "label":
            # Means we haven't yet written a full header row with f0, f1, ...
            need_header_fix = True

    if need_header_fix:
        # Read all existing rows (probably just the old header + maybe some data)
        rows = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Build a new header: "label" + f0, f1, ..., f(N-1)
        # len(vec) tells us how many features we have.
        header = ["label"] + [f"f{i}" for i in range(len(vec))]
        rows[0] = header  # replace first row with updated header

        # Rewrite the entire file: new header + existing data rows (if any)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    # 2) Append the new sample as one row at the bottom of the CSV.
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        # Convert vec (NumPy array) to a Python list, prepend the label
        row = [label] + vec.tolist()
        writer.writerow(row)


def main():
    # Ensure CSV exists and has at least a temporary header
    init_csv(CSV_PATH)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # Set up MediaPipe FaceMesh for live video
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,     # video mode: uses tracking after first detection
        max_num_faces=2,             # we allow up to 2 faces in the frame
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Counters so we can see how many samples per class we collected
    navid_count = 0
    other_count = 0

    # Console instructions
    print("=== Identity Data Collection ===")
    print("Controls:")
    print("  N - save current face as 'navid'")
    print("  O - save current face as 'other'")
    print("  Q - quit")
    print("Tips:")
    print("  • Try different angles, distances, lighting.")
    print("  • Make sure only ONE main face is clearly visible when saving.")

    # -------------- MAIN LOOP: read frames and collect samples --------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # camera error / stream ended

        # Flip horizontally so it feels like a mirror (easier for user)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR (OpenCV) to RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run FaceMesh on this frame
        results = face_mesh.process(rgb)

        # current_vec will store the numeric feature vector for the *current* face
        current_vec = None

        if results.multi_face_landmarks:
            # If FaceMesh found at least one face, take the first one as "main face"
            face_landmarks = results.multi_face_landmarks[0]

            # Convert the 468 landmarks into a compact face feature vector
            # using our own feature engineering function
            current_vec = face_to_vector(face_landmarks)

            # Draw face landmarks on the frame so the user sees what is being captured
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,  # full face mesh connections
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

        # ---------- HUD text: show sample counts + controls ----------

        # Top-left: how many samples we have for each class
        cv2.putText(
            frame,
            f"samples navid: {navid_count}  other: {other_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Bottom: keys the user can press
        cv2.putText(
            frame,
            "N: save navid   O: save other   Q: quit",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # Show the preview window
        cv2.imshow("Identity Data Collection", frame)

        # Read one key from the keyboard (non-blocking, every frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit condition
        if key == ord('q'):
            break

        # If FaceMesh did NOT find a stable face, we cannot save anything.
        # This avoids writing "empty" or garbage rows.
        if current_vec is None:
            continue

        # If user pressed 'n' and we have a valid face vector, save as "navid"
        if key == ord('n'):
            append_sample("navid", current_vec, CSV_PATH)
            navid_count += 1
            print(f"[INFO] Saved NAVID sample #{navid_count}")

        # If user pressed 'o' and we have a valid face vector, save as "other"
        elif key == ord('o'):
            append_sample("other", current_vec, CSV_PATH)
            other_count += 1
            print(f"[INFO] Saved OTHER sample #{other_count}")

    # ---------- Cleanup ----------
    cap.release()              # release webcam
    cv2.destroyAllWindows()    # close OpenCV windows


# Standard Python entry point:
# only run main() if this file is executed directly (not imported as a module).
if __name__ == "__main__":
    main()
