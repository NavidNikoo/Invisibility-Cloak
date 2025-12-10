import cv2                    # OpenCV: webcam capture, drawing, windows
import numpy as np            # NumPy: array operations, blending, masks
import time                   # time: for countdowns, timestamps
import mediapipe as mp        # MediaPipe: segmentation, hands, face mesh
import joblib                 # joblib: load saved sklearn models (.pkl)
import os                     # os: file system operations (directories, paths)

from identity_features import face_to_vector   # Converts FaceMesh landmarks -> numeric feature vector

# Paths to trained models on disk
GESTURE_MODEL_PATH = "gesture_model.pkl"      # Multi-class hand gesture model
IDENTITY_MODEL_PATH = "identity_model.pkl"    # Binary face model: "navid" vs "other"

# Directory to store screenshots taken by "thumbs_up" gesture
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)      # Create folder if it does not exist


# --------- Background capture utility ---------
def capture_background(cap, seconds=1.0, max_frames=60):
    """
    Capture a short burst of frames and compute their median to estimate
    the "clean background" with no person.

    - cap: OpenCV VideoCapture object
    - seconds: how long to capture
    - max_frames: safety limit on number of frames
    """
    frames = []
    t0 = time.time()
    print("[INFO] Capturing background... please step out of frame.")

    # Keep reading frames until time or frame limit is reached
    while (time.time() - t0) < seconds and len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)     # Flip like main loop (mirror view)
        frames.append(f)

    if not frames:
        print("[WARN] No frames captured for background.")
        return None

    # Take the pixel-wise median over all captured frames to remove moving noise
    bg = np.median(frames, axis=0).astype(np.uint8)
    print("[INFO] Background captured.")
    return bg


# --------- Hand landmarks → feature vector ---------
def landmarks_to_vector(hand_landmarks):
    """
    Convert MediaPipe's 21 hand landmarks into a flat feature vector:
    [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

    - hand_landmarks: MediaPipe hand landmarks object
    """
    coords = []
    for lm in hand_landmarks.landmark:  # 21 landmarks
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def main():
    # --------- Setup camera ---------
    cap = cv2.VideoCapture(0)  # Open default webcam
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # Optionally set resolution (helps make output more consistent)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --------- Setup MediaPipe components ---------

    # 1) Person segmentation for cloak effect (SelfieSegmentation)
    mp_selfie = mp.solutions.selfie_segmentation
    segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

    # 2) Hands for gesture recognition
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,               # track up to 2 hands
        min_detection_confidence=0.5,  # threshold for initial hand detection
        min_tracking_confidence=0.5    # threshold to keep tracking
    )

    # 3) FaceMesh for identity recognition (Navid vs other)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,        # video mode (tracking, not just detect)
        max_num_faces=4,               # allow up to 4 faces in frame
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --------- Load gesture model ---------
    print(f"[INFO] Loading gesture model from {GESTURE_MODEL_PATH} ...")
    gesture_model = joblib.load(GESTURE_MODEL_PATH)  # sklearn pipeline
    print("[INFO] Gesture model loaded.")

    # --------- Load identity model (navid vs other) ---------
    try:
        print(f"[INFO] Loading identity model from {IDENTITY_MODEL_PATH} ...")
        identity_model = joblib.load(IDENTITY_MODEL_PATH)  # sklearn pipeline
        identity_enabled = True
        print("[INFO] Identity model loaded (Navid vs Other).")
    except Exception as e:
        # If identity model not found or fails to load, fall back to "everyone"
        print(f"[WARN] Could not load identity model: {e}")
        print("[WARN] Running WITHOUT identity gating (everyone cloaked, everyone can control).")
        identity_model = None
        identity_enabled = False

    # State variables for the cloak and background
    background = None          # Will hold the clean background image
    cloak_on = True            # Whether invisibility effect is active
    snapshot_idx = 0           # Indexing snapshots: snapshot_0000.png, etc.

    # Edge-triggered gesture logic: remember the previous frame's "active" gestures
    active_control_gestures = set()

    # Countdown state for background recapture (triggered by open_palm)
    countdown_active = False
    countdown_start = 0.0
    countdown_duration = 5.0  # seconds until recapture

    # Human-readable descriptions for gesture labels on HUD
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

    # ---------------------- MAIN LOOP (per frame) ----------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]         # height, width for text placement and masks
        out = frame.copy()             # 'out' is the frame we draw the final effect onto
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB

        # -------------------------------------------------
        # 1) PERSON SEGMENTATION (base mask + people count)
        # -------------------------------------------------
        seg_results = segmentor.process(rgb)
        mask = seg_results.segmentation_mask          # float mask: each pixel ∈ [0, 1]

        # Convert segmentation mask to binary: 1 = person, 0 = background
        bin_mask = (mask > 0.4).astype(np.uint8)

        # Clean noise in the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Connected components on the clean mask (to find separate "blobs" = people)
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            clean_mask, connectivity=4
        )

        # Count how many "people" are in the frame (large enough connected components)
        people_count = sum(
            1 for i in range(1, num_labels)  # skip label 0 (background)
            if stats[i, cv2.CC_STAT_AREA] > 0.015 * h * w
        )

        # -------------------------------------------------
        # 2) IDENTITY DETECTION (who is Navid?)
        # -------------------------------------------------
        navid_centers = []         # list of (x, y) centers for Navid's faces
        has_navid = False          # at least one Navid detected?
        has_other = False          # at least one non-Navid face detected?
        identity_status = "OFF"    # string displayed on HUD

        if identity_enabled:
            # Run FaceMesh on the same RGB frame
            face_results = face_mesh.process(rgb)

            if face_results.multi_face_landmarks:
                # Iterate over all detected faces
                for face_landmarks in face_results.multi_face_landmarks:
                    # Convert all face landmarks into a numeric feature vector
                    vec = face_to_vector(face_landmarks)
                    # Use identity model to predict "navid" or "other"
                    pred_id = identity_model.predict([vec])[0]

                    # Compute approximate 2D face center (average of all landmark x and y)
                    xs = [lm.x for lm in face_landmarks.landmark]
                    ys = [lm.y for lm in face_landmarks.landmark]
                    cx = int(np.mean(xs) * w)
                    cy = int(np.mean(ys) * h)

                    if pred_id == "navid":
                        has_navid = True
                        navid_centers.append((cx, cy))
                        # Green circle for Navid's detected face
                        cv2.circle(out, (cx, cy), 10, (0, 255, 0), 2)
                    else:
                        has_other = True
                        # Red circle for other people's faces
                        cv2.circle(out, (cx, cy), 10, (0, 0, 255), 2)

            # Build a friendly identity label for the HUD
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

        # After identity detection, estimate Navid's horizontal position (x in [0,1]).
        # We will use this to decide which hands belong to Navid.
        navid_x_norm = None
        if navid_centers:
            # Take mean x of all Navid faces, normalize by image width
            navid_x_norm = np.mean([cx for (cx, cy) in navid_centers]) / float(w)

        # -------------------------------------------------
        # 3) BUILD NAVID-ONLY PERSON MASK
        # -------------------------------------------------
        # Start with target_mask = all people (bin_mask)
        target_mask = bin_mask.copy()

        if identity_enabled and navid_centers:
            # We want to restrict the mask so it includes only the connected components
            # (person blobs) that Navid's face lies inside of.
            navid_person_mask = np.zeros_like(bin_mask)

            for (fx, fy) in navid_centers:
                # Make sure center is inside image bounds
                if 0 <= fx < w and 0 <= fy < h:
                    # Get which connected component label the point (fx, fy) falls into
                    label_id = labels_cc[fy, fx]
                    if label_id != 0:
                        # Optionally ignore very small blobs (noise)
                        if stats[label_id, cv2.CC_STAT_AREA] > 0.015 * h * w:
                            # Set mask=1 for all pixels in this connected component
                            navid_person_mask[labels_cc == label_id] = 1

            # If we found any area for Navid, use that as the target_mask.
            # Otherwise, if Navid isn't properly matched to a blob, cloak no one.
            if navid_person_mask.sum() > 0:
                target_mask = navid_person_mask
            else:
                target_mask = np.zeros_like(bin_mask)

        # -------------------------------------------------
        # 4) APPLY INVISIBILITY USING target_mask
        # -------------------------------------------------
        if background is not None and cloak_on:
            # Smooth the person mask edges to avoid harsh cutouts
            closed = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            dilated = cv2.dilate(closed, kernel, iterations=2)
            eroded = cv2.erode(closed, kernel, iterations=1)

            core = eroded                 # solid region where we fully replace with background
            edge = cv2.subtract(dilated, eroded)  # thin edge region for blending

            # Convert to float for blending operations
            frame_f = frame.astype(np.float32)
            bg_f = background.astype(np.float32)
            out_f = frame_f.copy()

            # Replace core region completely with background
            core_idx = core.astype(bool)
            out_f[core_idx] = bg_f[core_idx]

            # Blend at the edge region between frame and background
            edge_idx = edge.astype(bool)
            alpha = 0.6   # weight for background at edges
            out_f[edge_idx] = alpha * bg_f[edge_idx] + (1.0 - alpha) * frame_f[edge_idx]

            # Convert back to uint8 image
            out = out_f.astype(np.uint8)

        # -------------------------------------------------
        # 5) GESTURE DETECTION (hands)
        # -------------------------------------------------
        hands_results = hands.process(rgb)

        # For HUD: all gesture labels detected in the frame (from any hand)
        frame_raw_gestures = set()
        # For control: only gestures from hands that belong to Navid (gated by identity)
        navid_control_gestures = set()

        if hands_results.multi_hand_landmarks:
            for hand in hands_results.multi_hand_landmarks:
                # Estimate horizontal position of this hand (average x of landmarks, 0–1)
                hand_x_mean = np.mean([lm.x for lm in hand.landmark])

                # Convert landmarks to 63D feature vector and use gesture model to predict
                features = landmarks_to_vector(hand)
                pred_label = gesture_model.predict([features])[0]
                frame_raw_gestures.add(pred_label)

                # Decide whether this hand is "Navid's hand" based on x position
                is_navid_hand = False
                if identity_enabled and navid_x_norm is not None:
                    # If hand's x is near Navid's face center x, we assume it's Navid
                    if abs(hand_x_mean - navid_x_norm) < 0.18:
                        is_navid_hand = True
                else:
                    # If identity model is not active, accept all hands as controllers
                    is_navid_hand = True

                if is_navid_hand:
                    # Add to set of gestures that can control the system
                    navid_control_gestures.add(pred_label)

                    # Draw Navid's hand landmarks on the output frame (for clarity)
                    if pred_label in GESTURE_DESCRIPTIONS:
                        mp.solutions.drawing_utils.draw_landmarks(
                            out, hand, mp_hands.HAND_CONNECTIONS
                        )
                else:
                    # For other people's hands, we skip drawing / controlling (or could draw differently)
                    pass

        # ---------- Single-gesture control logic (Navid's hands only) ----------
        # We only allow control when exactly one gesture type is detected from Navid.
        if len(navid_control_gestures) == 1:
            control_gestures = set(navid_control_gestures)   # this set drives actions
            gesture_mode = "single"
        elif len(navid_control_gestures) > 1:
            # If we see multiple gesture types at once from Navid, treat as ambiguous
            control_gestures = set()
            gesture_mode = "multi"
        else:
            # No valid Navid-controlled gestures
            control_gestures = set()
            gesture_mode = "none"

        # ---------- Identity gate for gesture control ----------
        # If identity is enabled and no Navid face is present, ignore gesture controls.
        gestures_allowed = True
        if identity_enabled and not navid_centers:
            gestures_allowed = False

        # Edge-triggered logic: we only react when a gesture appears (not every frame).
        # rising_edges = gestures in this frame that were NOT there in previous frame.
        rising_edges = control_gestures - active_control_gestures

        if gestures_allowed:
            for label in rising_edges:
                if label == "peace":
                    # Toggle cloak on/off
                    cloak_on = not cloak_on
                    print("[GESTURE] peace → toggle cloak")

                elif label == "open_palm":
                    # If countdown already running, cancel it; otherwise start it.
                    if countdown_active:
                        countdown_active = False
                        print("[GESTURE] open_palm → CANCEL countdown")
                    else:
                        countdown_active = True
                        countdown_start = time.time()
                        print("[GESTURE] open_palm → starting countdown")

                elif label == "thumbs_up":
                    # Save snapshot of current 'out' frame
                    fname = os.path.join(SNAPSHOT_DIR, f"snapshot_{snapshot_idx:04d}.png")
                    cv2.imwrite(fname, out)
                    snapshot_idx += 1
                    print(f"[GESTURE] Saved snapshot → {fname}")
        else:
            # If gestures are disallowed (no Navid present), print info for debugging
            if rising_edges:
                print("[INFO] Gestures detected but ignored (Navid not present).")

        # Update active_control_gestures for next frame's edge detection
        active_control_gestures = control_gestures

        # -------------------------------------------------
        # 6) Countdown UI for background recapture
        # -------------------------------------------------
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed

            if remaining > 0:
                # Show countdown text on the screen
                text = f"Recapturing background in {remaining:.1f}s"
                cv2.putText(
                    out, text,
                    (int(w * 0.22), int(h * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 255, 255), 4
                )
                cv2.putText(
                    out, "Please step out of frame",
                    (int(w * 0.28), int(h * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2
                )
            else:
                # Once countdown is done, actually recapture background
                bg = capture_background(cap, seconds=1.0)
                if bg is not None:
                    background = bg
                countdown_active = False

        # -------------------------------------------------
        # 7) HUD (status text)
        # -------------------------------------------------
        # Show cloak status and whether background is set
        status = f"Cloak: {'ON' if cloak_on else 'OFF'} | BG: {'SET' if background is not None else 'NONE'}"

        # Summarize gestures for HUD
        if gesture_mode == "single":
            gest_str = ", ".join(sorted(frame_raw_gestures))   # show detected gestures
        elif gesture_mode == "multi":
            gest_str = "AMBIGUOUS (multiple gestures)"
        else:
            gest_str = "NONE"

        # Draw status line
        cv2.putText(
            out, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
        # Draw gestures line
        cv2.putText(
            out, f"Gestures: {gest_str}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )
        # Draw people count
        cv2.putText(
            out, f"People: {people_count}", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        # Draw identity status (green if Navid present)
        cv2.putText(
            out, f"Identity: {identity_status}", (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if "Navid" in identity_status else (255, 255, 255),
            2
        )

        # Draw keyboard fallback controls
        cv2.putText(
            out, "Q: quit  |  B: capture background  |  C: toggle cloak",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )

        # Show the final processed frame
        cv2.imshow("Invisibility Cloak – Multi-Gesture + Identity", out)

        # Keyboard fallbacks for debugging or if gestures fail
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # Manually capture background immediately
            bg = capture_background(cap, seconds=1.0)
            if bg is not None:
                background = bg
        elif key == ord('c'):
            # Manually toggle cloak state
            cloak_on = not cloak_on

    # --------- Cleanup: close all resources ---------
    hands.close()
    face_mesh.close()
    segmentor.close()
    cap.release()
    cv2.destroyAllWindows()


# Standard Python entry point: run main() only if this file is executed directly
if __name__ == "__main__":
    main()
