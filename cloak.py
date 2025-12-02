import cv2
import numpy as np
import time
import mediapipe as mp
import joblib


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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
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

    # --------- Load trained peace-sign model ---------
    print("[INFO] Loading gesture model from peace_model.pkl ...")
    gesture_model = joblib.load("peace_model.pkl")
    print("[INFO] Gesture model loaded.")

    background = None
    cloak_on = True   # whether invisibility is active

    # For gesture toggle logic
    gesture_active = False       # True while peace sign is currently visible
    last_gesture_label = "none"  # "peace" or "none"

    print("=== Invisibility Cloak with Peace-Sign Trigger ===")
    print("Controls:")
    print("  B - capture background (step out of the frame)")
    print("  C - toggle cloak on/off manually")
    print("  Q - quit")
    print("Gesture:")
    print("  Show a PEACE SIGN with your hand to toggle cloak on/off ✔")

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
        gesture_label = "none"
        hands_results = hands.process(rgb)

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                features = landmarks_to_vector(hand_landmarks)
                pred = gesture_model.predict([features])[0]  # 'peace' or 'other'

                if pred == "peace":
                    gesture_label = "peace"
                    # Optional: draw landmarks to visualize
                    mp.solutions.drawing_utils.draw_landmarks(
                        out,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                    break  # we only need one peace hand to trigger

        # Edge-triggered toggle: only toggle when we go from "no peace" -> "peace"
        if gesture_label == "peace" and not gesture_active:
            # rising edge: peace sign just appeared
            cloak_on = not cloak_on
            gesture_active = True
            print(f"[INFO] Peace sign detected → cloak_on = {cloak_on}")
        elif gesture_label != "peace":
            # reset when peace sign disappears
            gesture_active = False

        # --------- HUD text ---------
        status1 = f"Cloak: {'ON' if cloak_on else 'OFF'} | BG: {'SET' if background is not None else 'NONE'}"
        status2 = f"Gesture: {gesture_label.upper() if gesture_label != 'none' else 'NONE'} (peace toggles cloak)"

        cv2.putText(out, status1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, status2, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(out, "Q: quit  |  B: capture background  |  C: toggle cloak",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Phase 1+2 – Invisibility Cloak (Peace Trigger)", out)

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
