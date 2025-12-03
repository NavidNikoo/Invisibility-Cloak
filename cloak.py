import cv2
import numpy as np
import time
import mediapipe as mp
import joblib
import os

MODEL_PATH = "gesture_model.pkl"  # trained multi-gesture model

# Directory for gesture-triggered snapshots
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# --------- Background capture utility ---------
def capture_background(cap, seconds=1.0, max_frames=60):
    """
    Capture a short burst of frames and return their median as the background.
    User should move out of the frame while this runs.
    """
    frames = []
    t0 = time.time()
    print("[INFO] Capturing background... please step out of frame.")
    while (time.time() - t0) < seconds and len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        frames.append(f)

    if not frames:
        print("[WARN] No frames captured for background.")
        return None

    bg = np.median(frames, axis=0).astype(np.uint8)
    print("[INFO] Background captured.")
    return bg


# --------- Hand landmark → feature vector (same format as training) ---------
def landmarks_to_vector(hand_landmarks):
    """
    Convert MediaPipe hand landmarks into a flat [x0, y0, z0, ..., x20, y20, z20] list.
    This must match the format used in collect_gesture_data.py.
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def main():
    # --------- Setup camera ---------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --------- Setup MediaPipe: person segmentation + hands ---------
    mp_selfie = mp.solutions.selfie_segmentation
    segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --------- Load trained gesture model ---------
    print(f"[INFO] Loading gesture model from {MODEL_PATH} ...")
    gesture_model = joblib.load(MODEL_PATH)
    print("[INFO] Gesture model loaded.")

    background = None
    cloak_on = True   # whether invisibility is active
    snapshot_idx = 0  # for naming snapshot files

    # For edge-triggered gesture logic
    active_gestures = set()   # labels present in previous frame
    should_quit = False       # set to True when open_palm is detected

    # Gesture → description for HUD / logs
    # Make sure these keys match the label strings in gesture_data.csv
    GESTURE_DESCRIPTIONS = {
        "peace": "Toggle cloak",
        "open_palm": "Quit app",
        "thumbs_up": "Snapshot"
    }

    print("=== Invisibility Cloak – Multi-Gesture Control ===")
    print("Keyboard controls:")
    print("  B - capture background (step out of the frame)")
    print("  C - toggle cloak on/off manually")
    print("  Q - quit")
    print("Gesture controls (from model labels):")
    print("  peace       → toggle cloak on/off")
    print("  open_palm   → quit application")
    print("  thumbs_up   → save snapshot to ./snapshots")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        out = frame.copy()

        # Convert once to RGB for both segmentation and hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------- Person segmentation for invisibility ---------
        if background is not None and cloak_on:
            seg_results = segmentor.process(rgb)
            mask = seg_results.segmentation_mask  # float [0,1]

            # 1) threshold (slightly low to capture more of the body)
            bin_mask = (mask > 0.4).astype(np.uint8)

            # 2) Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 3) inner core + edge band
            dilated = cv2.dilate(closed, kernel, iterations=2)
            eroded = cv2.erode(closed, kernel, iterations=1)

            core = eroded
            edge = cv2.subtract(dilated, eroded)

            frame_f = frame.astype(np.float32)
            bg_f = background.astype(np.float32)
            out_f = frame_f.copy()

            # fully replace core with background
            core_idx = core.astype(bool)
            out_f[core_idx] = bg_f[core_idx]

            # mix edge band
            edge_idx = edge.astype(bool)
            alpha = 0.6
            out_f[edge_idx] = alpha * bg_f[edge_idx] + (1.0 - alpha) * frame_f[edge_idx]

            out = out_f.astype(np.uint8)

        # --------- Gesture detection (MediaPipe Hands + classifier) ---------
        hands_results = hands.process(rgb)
        frame_gestures = set()   # gestures seen in THIS frame

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                features = landmarks_to_vector(hand_landmarks)
                pred_label = gesture_model.predict([features])[0]  # e.g. 'peace', 'open_palm', ...

                frame_gestures.add(pred_label)

                # Optional: only draw for "known" gestures to avoid clutter
                if pred_label in GESTURE_DESCRIPTIONS:
                    mp.solutions.drawing_utils.draw_landmarks(
                        out,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

        # --------- Edge-triggered gesture handling ---------
        # rising_edges = gestures that just appeared in this frame
        rising_edges = frame_gestures - active_gestures

        for label in rising_edges:
            if label == "peace":
                cloak_on = not cloak_on
                state = "ON" if cloak_on else "OFF"
                print(f"[GESTURE] peace → toggle cloak → {state}")

            elif label == "open_palm":
                # use open_palm as "quit" gesture
                print("[GESTURE] open_palm → QUIT")
                should_quit = True

            elif label == "thumbs_up":
                # save snapshot of current output frame
                fname = os.path.join(SNAPSHOT_DIR, f"snapshot_{snapshot_idx:04d}.png")
                cv2.imwrite(fname, out)
                snapshot_idx += 1
                print(f"[GESTURE] thumbs_up → saved snapshot to {fname}")

            else:
                # other labels: no bound action (but still visible in HUD)
                pass

        # update active gestures for next frame
        active_gestures = frame_gestures

        # if a quit gesture was seen, break out of the main loop
        if should_quit:
            break

        # --------- HUD text ---------
        status1 = f"Cloak: {'ON' if cloak_on else 'OFF'} | BG: {'SET' if background is not None else 'NONE'}"

        # Show all gestures currently visible (even if they didn't just trigger)
        if frame_gestures:
            gest_str = ", ".join(sorted(list(frame_gestures)))
        else:
            gest_str = "NONE"

        status2 = f"Gestures: {gest_str}"

        cv2.putText(out, status1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, status2, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(out, "Q: quit  |  B: capture background  |  C: toggle cloak",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Invisibility Cloak – Multi-Gesture Control", out)

        # --------- Keyboard fallbacks ---------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            background = capture_background(cap, seconds=1.0)
        elif key == ord('c'):
            cloak_on = not cloak_on

    # --------- Cleanup ---------
    hands.close()
    segmentor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
