import cv2
import csv
import os
import mediapipe as mp

CSV_PATH = "gesture_data.csv"

GESTURE_KEYS = {
    ord('1'): "peace",
    ord('2'): "thumbs_up",
    ord('3'): "open_palm",
    ord('0'): "other",
}

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


def save_sample(label, landmarks, path):
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
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    counts = {label: 0 for label in set(GESTURE_KEYS.values())}

    print("=== Multi-Gesture Data Collection ===")
    print("Controls (press while ONE hand is visible):")
    print("  1 - save 'peace'")
    print("  2 - save 'thumbs_up'")
    print("  3 - save 'open_palm'")
    print("  0 - save 'other'")
    print("  Q - quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmarks = hand_landmarks.landmark
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # HUD
        y0 = 25
        for label, c in counts.items():
            cv2.putText(frame, f"{label}: {c}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y0 += 25

        cv2.putText(frame, "1: peace  2: thumbs_up  3: open_palm  0: other  Q: quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Multi-Gesture Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if current_landmarks is None:
            continue

        if key in GESTURE_KEYS:
            label = GESTURE_KEYS[key]
            save_sample(label, current_landmarks, CSV_PATH)
            counts[label] += 1
            print(f"[INFO] Saved {label.upper()} sample #{counts[label]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
