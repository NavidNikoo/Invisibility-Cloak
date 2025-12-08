import cv2
import numpy as np
import time
import mediapipe as mp
import joblib
import os

from identity_features import face_to_vector   # face -> vector for identity model

GESTURE_MODEL_PATH = "gesture_model.pkl"      # trained multi-gesture model
IDENTITY_MODEL_PATH = "identity_model.pkl"    # trained navid vs other

# Directory for gesture-triggered snapshots
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# --------- Background capture utility ---------
def capture_background(cap, seconds=1.0, max_frames=60):
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


# --------- Hand landmarks → feature vector ---------
def landmarks_to_vector(hand_landmarks):
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --------- Setup MediaPipe ---------
    mp_selfie = mp.solutions.selfie_segmentation
    segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --------- Load gesture model ---------
    print(f"[INFO] Loading gesture model from {GESTURE_MODEL_PATH} ...")
    gesture_model = joblib.load(GESTURE_MODEL_PATH)
    print("[INFO] Gesture model loaded.")

    # --------- Load identity model (navid vs other) ---------
    try:
        print(f"[INFO] Loading identity model from {IDENTITY_MODEL_PATH} ...")
        identity_model = joblib.load(IDENTITY_MODEL_PATH)
        identity_enabled = True
        print("[INFO] Identity model loaded (Navid vs Other).")
    except Exception as e:
        print(f"[WARN] Could not load identity model: {e}")
        print("[WARN] Running WITHOUT identity gating (everyone cloaked, everyone can control).")
        identity_model = None
        identity_enabled = False

    background = None
    cloak_on = True
    snapshot_idx = 0

    # for edge-triggered actions, we track only the *control* gesture(s)
    active_control_gestures = set()

    # Countdown state
    countdown_active = False
    countdown_start = 0.0
    countdown_duration = 5.0  # seconds

    # Gesture labels (for drawing only)
    GESTURE_DESCRIPTIONS = {
        "peace": "Toggle cloak",
        "open_palm": "Recapture background",
        "thumbs_up": "Snapshot"
    }

    print("=== Invisibility Cloak – Multi-Gesture Control + Identity ===")
    print("Gesture controls:")
    print("  peace       → toggle cloak")
    print("  open_palm   → recapture background (5s timer, open again to cancel)")
    print("  thumbs_up   → snapshot")
    print("Identity:")
    print("  Only NAVID is cloaked / allowed to control (when identity_model.pkl is present).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        out = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------------------------
        # 1) PERSON SEGMENTATION (base mask + people count)
        # -------------------------------------------------
        seg_results = segmentor.process(rgb)
        mask = seg_results.segmentation_mask
        bin_mask = (mask > 0.4).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            clean_mask, connectivity=4
        )

        # Count people (large connected components)
        people_count = sum(
            1 for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > 0.015 * h * w
        )

        # -------------------------------------------------
        # 2) IDENTITY DETECTION (who is Navid?)
        # -------------------------------------------------
        navid_centers = []
        has_navid = False
        has_other = False
        identity_status = "OFF"

        if identity_enabled:
            face_results = face_mesh.process(rgb)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    vec = face_to_vector(face_landmarks)
                    pred_id = identity_model.predict([vec])[0]   # 'navid' or 'other'

                    xs = [lm.x for lm in face_landmarks.landmark]
                    ys = [lm.y for lm in face_landmarks.landmark]
                    cx = int(np.mean(xs) * w)
                    cy = int(np.mean(ys) * h)

                    if pred_id == "navid":
                        has_navid = True
                        navid_centers.append((cx, cy))
                        cv2.circle(out, (cx, cy), 10, (0, 255, 0), 2)  # green for Navid
                    else:
                        has_other = True
                        cv2.circle(out, (cx, cy), 10, (0, 0, 255), 2)  # red for others

            # Build status string for HUD
            if has_navid and has_other:
                identity_status = "Navid and other"
            elif has_navid:
                identity_status = "Navid"
            elif has_other:
                identity_status = "other"
            else:
                identity_status = "no face"
        else:
            identity_status = "disabled"

        # After identity detection, estimate Navid's horizontal position (0–1)
        navid_x_norm = None
        if navid_centers:
            navid_x_norm = np.mean([cx for (cx, cy) in navid_centers]) / float(w)

        # -------------------------------------------------
        # 3) BUILD NAVID-ONLY PERSON MASK
        # -------------------------------------------------
        # Default: cloak everyone in the person mask
        target_mask = bin_mask.copy()

        if identity_enabled and navid_centers:
            navid_person_mask = np.zeros_like(bin_mask)

            for (fx, fy) in navid_centers:
                if 0 <= fx < w and 0 <= fy < h:
                    label_id = labels_cc[fy, fx]  # which connected component?
                    if label_id != 0:
                        # optional: ignore tiny blobs
                        if stats[label_id, cv2.CC_STAT_AREA] > 0.015 * h * w:
                            navid_person_mask[labels_cc == label_id] = 1

            # If we found any Navid blobs, use them; otherwise cloak no one
            if navid_person_mask.sum() > 0:
                target_mask = navid_person_mask
            else:
                target_mask = np.zeros_like(bin_mask)

        # -------------------------------------------------
        # 4) APPLY INVISIBILITY USING target_mask
        # -------------------------------------------------
        if background is not None and cloak_on:
            closed = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            dilated = cv2.dilate(closed, kernel, iterations=2)
            eroded = cv2.erode(closed, kernel, iterations=1)

            core = eroded
            edge = cv2.subtract(dilated, eroded)

            frame_f = frame.astype(np.float32)
            bg_f = background.astype(np.float32)
            out_f = frame_f.copy()

            core_idx = core.astype(bool)
            out_f[core_idx] = bg_f[core_idx]

            edge_idx = edge.astype(bool)
            alpha = 0.6
            out_f[edge_idx] = alpha * bg_f[edge_idx] + (1.0 - alpha) * frame_f[edge_idx]

            out = out_f.astype(np.uint8)

        # -------------------------------------------------
        # 5) GESTURE DETECTION (hands)
        # -------------------------------------------------
        hands_results = hands.process(rgb)

        # For HUD: all gestures we see in the frame
        frame_raw_gestures = set()
        # For control: only gestures from hands that belong to Navid
        navid_control_gestures = set()

        if hands_results.multi_hand_landmarks:
            for hand in hands_results.multi_hand_landmarks:
                # mean hand x-position in normalized coords (0–1)
                hand_x_mean = np.mean([lm.x for lm in hand.landmark])

                # classify gesture
                features = landmarks_to_vector(hand)
                pred_label = gesture_model.predict([features])[0]
                frame_raw_gestures.add(pred_label)

                # decide if this hand belongs to Navid
                is_navid_hand = False
                if identity_enabled and navid_x_norm is not None:
                    # only accept hands whose center is close to Navid's face in x
                    if abs(hand_x_mean - navid_x_norm) < 0.18:
                        is_navid_hand = True
                else:
                    # if identity is disabled, allow all hands
                    is_navid_hand = True

                if is_navid_hand:
                    navid_control_gestures.add(pred_label)
                    if pred_label in GESTURE_DESCRIPTIONS:
                        mp.solutions.drawing_utils.draw_landmarks(
                            out, hand, mp_hands.HAND_CONNECTIONS
                        )
                else:
                    # optionally: draw other people's hands differently or skip
                    pass

        # ---------- Single-gesture control logic (Navid's hands only) ----------
        if len(navid_control_gestures) == 1:
            control_gestures = set(navid_control_gestures)   # allowed to trigger
            gesture_mode = "single"
        elif len(navid_control_gestures) > 1:
            control_gestures = set()                         # ambiguous, disable
            gesture_mode = "multi"
        else:
            control_gestures = set()
            gesture_mode = "none"

        # ---------- Identity gate for gesture control ----------
        # If identity is enabled, only let gestures trigger when Navid is present.
        gestures_allowed = True
        if identity_enabled and not navid_centers:
            gestures_allowed = False

        # Edge-triggered gesture actions (only from control_gestures)
        rising_edges = control_gestures - active_control_gestures
        if gestures_allowed:
            for label in rising_edges:
                if label == "peace":
                    cloak_on = not cloak_on
                    print("[GESTURE] peace → toggle cloak")

                elif label == "open_palm":
                    # If countdown is already active, cancel it instead of starting a new one
                    if countdown_active:
                        countdown_active = False
                        print("[GESTURE] open_palm → CANCEL countdown")
                    else:
                        countdown_active = True
                        countdown_start = time.time()
                        print("[GESTURE] open_palm → starting countdown")

                elif label == "thumbs_up":
                    fname = os.path.join(SNAPSHOT_DIR, f"snapshot_{snapshot_idx:04d}.png")
                    cv2.imwrite(fname, out)
                    snapshot_idx += 1
                    print(f"[GESTURE] Saved snapshot → {fname}")
        else:
            if rising_edges:
                print("[INFO] Gestures detected but ignored (Navid not present).")

        active_control_gestures = control_gestures

        # -------------------------------------------------
        # 6) Countdown UI for background recapture
        # -------------------------------------------------
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed

            if remaining > 0:
                text = f"Recapturing background in {remaining:.1f}s"
                cv2.putText(out, text, (int(w * 0.22), int(h * 0.45)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4)
                cv2.putText(out, "Please step out of frame",
                            (int(w * 0.28), int(h * 0.55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                bg = capture_background(cap, seconds=1.0)
                if bg is not None:
                    background = bg
                countdown_active = False

        # -------------------------------------------------
        # 7) HUD
        # -------------------------------------------------
        status = f"Cloak: {'ON' if cloak_on else 'OFF'} | BG: {'SET' if background is not None else 'NONE'}"

        if gesture_mode == "single":
            gest_str = ", ".join(sorted(frame_raw_gestures))
        elif gesture_mode == "multi":
            gest_str = "AMBIGUOUS (multiple gestures)"
        else:
            gest_str = "NONE"

        cv2.putText(out, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(out, f"Gestures: {gest_str}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(out, f"People: {people_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, f"Identity: {identity_status}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if "Navid" in identity_status else (255, 255, 255), 2)

        cv2.putText(out, "Q: quit  |  B: capture background  |  C: toggle cloak",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Invisibility Cloak – Multi-Gesture + Identity", out)

        # Keyboard fallbacks
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            bg = capture_background(cap, seconds=1.0)
            if bg is not None:
                background = bg
        elif key == ord('c'):
            cloak_on = not cloak_on

    hands.close()
    face_mesh.close()
    segmentor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
