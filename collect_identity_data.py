# collect_identity_data.py
import cv2
import csv
import os
import mediapipe as mp
from identity_features import face_to_vector

CSV_PATH = "identity_data.csv"

def init_csv(path):
    if not os.path.exists(path):
        # label + feature_0 ... feature_(N-1)
        # we don't know N yet until we compute one vector, so we’ll infer on first write
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # temporary header; we'll overwrite first row when we know dim
            writer.writerow(["label"])
        print(f"[INFO] Created new CSV at {path}")
    else:
        print(f"[INFO] Appending to existing CSV at {path}")

def append_sample(label, vec, path):
    """
    label: 'navid' or 'other'
    vec: 1D np.array of features
    """
    # If file only has ["label"] header, rewrite header with full feature columns
    need_header_fix = False
    with open(path, "r") as f:
        first_line = f.readline().strip()
        if first_line == "label":
            need_header_fix = True

    if need_header_fix:
        # rewrite file with proper header
        rows = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # build new header
        header = ["label"] + [f"f{i}" for i in range(len(vec))]
        rows[0] = header
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    # append row
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [label] + vec.tolist()
        writer.writerow(row)

def main():
    init_csv(CSV_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    navid_count = 0
    other_count = 0

    print("=== Identity Data Collection ===")
    print("Controls:")
    print("  N - save current face as 'navid'")
    print("  O - save current face as 'other'")
    print("  Q - quit")
    print("Tips:")
    print("  • Try different angles, distances, lighting.")
    print("  • Make sure only ONE main face is clearly visible when saving.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        current_vec = None

        if results.multi_face_landmarks:
            # take the most prominent face (first)
            face_landmarks = results.multi_face_landmarks[0]
            current_vec = face_to_vector(face_landmarks)

            # draw a simple overlay box / landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

        cv2.putText(frame, f"samples navid: {navid_count}  other: {other_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "N: save navid   O: save other   Q: quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Identity Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if current_vec is None:
            continue  # no face, can't save anything

        if key == ord('n'):
            append_sample("navid", current_vec, CSV_PATH)
            navid_count += 1
            print(f"[INFO] Saved NAVID sample #{navid_count}")

        elif key == ord('o'):
            append_sample("other", current_vec, CSV_PATH)
            other_count += 1
            print(f"[INFO] Saved OTHER sample #{other_count}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
