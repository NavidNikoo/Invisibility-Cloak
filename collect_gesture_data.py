import cv2           # OpenCV: used for webcam capture, drawing text, and showing windows
import csv           # Python CSV library: used to save gesture samples to a .csv file
import os            # Used to check if the CSV file already exists on disk
import mediapipe as mp  # MediaPipe: provides the Hands model for 3D hand landmarks

# Path to the file where all gesture samples will be stored.
CSV_PATH = "gesture_data.csv"

# Mapping from keyboard key codes → human-readable gesture labels.
# ord('1') gives the integer key code for the '1' key, etc.
GESTURE_KEYS = {
    ord('1'): "peace",
    ord('2'): "thumbs_up",
    ord('3'): "open_palm",
    ord('0'): "other",
}


# --------- Setup CSV header if file doesn't exist ---------
def init_csv(path):
    """
    Ensure that the CSV file exists and has a header row.

    If the file does not exist, create it and write a header like:
      label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20

    If it already exists, we just print a message and keep appending rows.
    """
    if not os.path.exists(path):
        # Start header with a 'label' column (gesture name).
        header = ["label"]

        # MediaPipe Hands returns 21 landmarks per hand.
        # For each landmark we will store x, y, z (3 coordinates).
        for i in range(21):  # 21 landmarks
            header += [f"x{i}", f"y{i}", f"z{i}"]

        # Create the CSV file and write the header row.
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        print(f"[INFO] Created new CSV with header at {path}")
    else:
        # File already exists → we will just append new samples to it.
        print(f"[INFO] Appending to existing CSV at {path}")


def save_sample(label, landmarks, path):
    """
    Save ONE gesture example (one frame) to the CSV file.

    - label: string like "peace", "thumbs_up", ...
    - landmarks: list of 21 MediaPipe landmark objects, each with (x, y, z)
    - path: where to append the row (CSV_PATH)
    """
    # Start row with the gesture label.
    row = [label]

    # For each of the 21 landmarks, append x, y, z to the row.
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])

    # Open file in append mode ("a") and write the row as a new line.
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    # Make sure the CSV file exists and has the correct header.
    init_csv(CSV_PATH)

    # Open the default webcam (index 0).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # ----- Set up MediaPipe Hands -----
    mp_hands = mp.solutions.hands

    # Create the Hands object that will detect and track a single hand.
    hands = mp_hands.Hands(
        static_image_mode=False,   # Treat input as a video stream (track after first detect)
        max_num_hands=1,           # Only track one hand; simplifies data
        model_complexity=1,        # Balance between speed and accuracy
        min_detection_confidence=0.5,  # Threshold for first detection
        min_tracking_confidence=0.5,   # Threshold for tracking across frames
    )

    # Helper used to draw the landmarks and connections on the image for visualization.
    mp_draw = mp.solutions.drawing_utils

    # Track how many samples we have collected per label in THIS run.
    # Example: {"peace": 0, "thumbs_up": 0, "open_palm": 0, "other": 0}
    counts = {label: 0 for label in set(GESTURE_KEYS.values())}

    # Print controls to the terminal so the user knows what keys to press.
    print("=== Multi-Gesture Data Collection ===")
    print("Controls (press while ONE hand is visible):")
    print("  1 - save 'peace'")
    print("  2 - save 'thumbs_up'")
    print("  3 - save 'open_palm'")
    print("  0 - save 'other'")
    print("  Q - quit")

    # ----- Main loop: run once per video frame -----
    while True:
        # Read one frame from the webcam.
        ok, frame = cap.read()
        if not ok:
            break  # If reading fails, exit the loop.

        # Flip the frame horizontally so it feels like a mirror.
        frame = cv2.flip(frame, 1)

        # Get height and width for placing text later.
        h, w, _ = frame.shape

        # MediaPipe expects RGB images instead of OpenCV's default BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the MediaPipe Hands model on this frame.
        results = hands.process(rgb)

        # This will hold the landmark list for the current frame (if a hand is detected).
        current_landmarks = None

        if results.multi_hand_landmarks:
            # We asked for at most 1 hand, so take the first one.
            hand_landmarks = results.multi_hand_landmarks[0]

            # This is a list of 21 landmark objects, each with x, y, z.
            current_landmarks = hand_landmarks.landmark

            # Draw the landmarks and bones on the frame so the user can see the detection.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ----- HUD (heads-up display) -----
        # Show how many samples have been collected for each label.
        y0 = 25
        for label, c in counts.items():
            cv2.putText(
                frame,
                f"{label}: {c}",             # e.g., "peace: 120"
                (10, y0),                    # position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,    # font
                0.6,                         # font scale
                (255, 255, 255),             # color (white)
                2                            # thickness
            )
            y0 += 25  # move text down for the next label

        # Show key instructions at the bottom of the window.
        cv2.putText(
            frame,
            "1: peace  2: thumbs_up  3: open_palm  0: other  Q: quit",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # Display the frame with landmarks and text.
        cv2.imshow("Multi-Gesture Data Collection", frame)

        # Wait 1 ms for a key press and get its code.
        key = cv2.waitKey(1) & 0xFF

        # If user presses 'q' or 'Q', exit the loop.
        if key == ord('q'):
            break

        # If no hand is currently detected, do NOT save anything even if a key was pressed.
        if current_landmarks is None:
            continue

        # If the pressed key corresponds to one of our gesture labels...
        if key in GESTURE_KEYS:
            # Look up its string label (e.g., 49 → "peace").
            label = GESTURE_KEYS[key]

            # Save this gesture example (label + 63 numbers) to the CSV file.
            save_sample(label, current_landmarks, CSV_PATH)

            # Update the count for this label and print feedback to the console.
            counts[label] += 1
            print(f"[INFO] Saved {label.upper()} sample #{counts[label]}")

    # ----- Cleanup: release resources -----
    cap.release()             # Free the webcam
    cv2.destroyAllWindows()   # Close any OpenCV windows


# Standard Python pattern: only run main() if this file is executed directly.
if __name__ == "__main__":
    main()
