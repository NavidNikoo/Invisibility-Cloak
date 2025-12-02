import cv2
import csv
import os
import mediapipe as mp

CSV_PATH = "gesture_data.csv"

# --------- Setup CSV header if file doesn't exist ---------
def init_csv(path):
    if not os.path.exists(path):
        header = ["label"]
        for i in range(21):  # 21 landmarks
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[INFO] Created new CSV with header at {path}")
    else:
        print(f"[INFO] Appending to existing CSV at {path}")


# --------- Save one sample ---------
def save_sample(label, landmarks, path):
    """
    label: string, e.g. "peace" or "other"
    landmarks: list of 21 landmarks from MediaPipe
    """
    row = [label]
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    init_csv(CSV_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,              # we only record one hand per sample
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    peace_count = 0
    other_count = 0

    print("=== Gesture Data Collection ===")
    print("Controls:")
    print("  P - save current hand as 'peace'")
    print("  O - save current hand as 'other'")
    print("  Q - quit")
    print("Tips:")
    print("  • Make sure exactly ONE hand is visible when you save.")
    print("  • Vary distance, rotation, lighting for better generalization.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_landmarks = None

        if results.multi_hand_landmarks:
            # Take the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmarks = hand_landmarks.landmark

            # Draw landmarks for visual feedback
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # HUD text
        cv2.putText(frame, f"Samples  peace: {peace_count}  other: {other_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "P: save peace   O: save other   Q: quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Only allow saving if we actually see a hand
        if current_landmarks is None:
            continue

        if key == ord('p'):
            save_sample("peace", current_landmarks, CSV_PATH)
            peace_count += 1
            print(f"[INFO] Saved PEACE sample #{peace_count}")
        elif key == ord('o'):
            save_sample("other", current_landmarks, CSV_PATH)
            other_count += 1
            print(f"[INFO] Saved OTHER sample #{other_count}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
