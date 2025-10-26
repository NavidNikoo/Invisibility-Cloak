import cv2, numpy as np, time, json, os, datetime

# ---------- Config ----------
CFG_PATH = "config.json"
DEFAULT_CFG = {
    "resolution": [1280, 720],
    "mode": "chroma",                 # "chroma" | "ml"
    "chroma_color": "red",            # "red" | "green"
    "hsv_red1": [0, 120, 70], "hsv_red2": [10, 255, 255],
    "hsv_red3": [170, 120, 70], "hsv_red4": [180, 255, 255],
    "hsv_green1": [35, 40, 40], "hsv_green2": [90, 255, 255],
    "morph_kernel": 5,
    "smooth_alpha": 0.35,             # EMA for mask smoothing (0..1)
}
def load_cfg():
    if os.path.exists(CFG_PATH):
        try: return json.load(open(CFG_PATH))
        except: pass
    json.dump(DEFAULT_CFG, open(CFG_PATH, "w"), indent=2)
    return DEFAULT_CFG.copy()
def save_cfg(cfg): json.dump(cfg, open(CFG_PATH, "w"), indent=2)

# ---------- Optional ML segmentation ----------
USE_ML = False
try:
    import mediapipe as mp
    mp_selfie = mp.solutions.selfie_segmentation
    USE_ML = True
except Exception:
    USE_ML = False

# ---------- Utils ----------
def median_background(cap, seconds=1.5, max_frames=60):
    frames = []
    t0 = time.time()
    while time.time()-t0 < seconds and len(frames) < max_frames:
        ok, f = cap.read()
        if ok: frames.append(f)
    if not frames: return None
    return np.median(frames, axis=0).astype(np.uint8)

def put_text(img, txt, y, x=10):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# ---------- Main ----------
def main():
    cfg = load_cfg()
    W, H = cfg["resolution"]
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    time.sleep(0.5)

    # Background
    print("Capturing background: step out of frame...")
    background = median_background(cap, seconds=1.5)
    if background is None:
        print("Camera error."); return

    # ML session
    ml = None
    if USE_ML:
        ml = mp_selfie.SelfieSegmentation(model_selection=1)

    # Recording
    os.makedirs("recordings", exist_ok=True)
    writer = None
    recording = False

    ema_mask = None
    mode = cfg["mode"] if (cfg["mode"] in ["chroma","ml"]) else "chroma"
    color = cfg["chroma_color"]
    kernel = np.ones((cfg["morph_kernel"], cfg["morph_kernel"]), np.uint8)

    fps_t0, fps_cnt, fps_val = time.time(), 0, 0.0

    print("Hotkeys: Q quit | B recapture bg | C chroma | M ML | 1 red | 2 green | [/] smooth | R record | S snapshot")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)  # mirror view

        # --- mask compute ---
        if mode == "chroma":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if color == "red":
                l1,u1 = np.array(cfg["hsv_red1"]), np.array(cfg["hsv_red2"])
                l2,u2 = np.array(cfg["hsv_red3"]), np.array(cfg["hsv_red4"])
                m = cv2.inRange(hsv, l1,u1) | cv2.inRange(hsv, l2,u2)
            else: # green
                l,u = np.array(cfg["hsv_green1"]), np.array(cfg["hsv_green2"])
                m = cv2.inRange(hsv, l,u)
        else:  # ML
            if USE_ML and ml is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = ml.process(rgb)
                seg = (res.segmentation_mask * 255).astype(np.uint8)
                _, m = cv2.threshold(seg, 128, 255, cv2.THRESH_BINARY)  # person=white
            else:
                m = np.zeros(frame.shape[:2], dtype=np.uint8)

        # For ML: we want to hide person; for chroma: hide cloak color.
        # Either way, "m" denotes region to replace with background.
        # Refine mask
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel, iterations=1)

        # EMA smoothing
        alpha = float(cfg["smooth_alpha"])
        if ema_mask is None: ema_mask = m.astype(np.float32)
        else: ema_mask = alpha*m + (1-alpha)*ema_mask
        m_smooth = ema_mask.astype(np.uint8)

        m_inv = cv2.bitwise_not(m_smooth)
        part_bg = cv2.bitwise_and(background, background, mask=m_smooth)
        part_fg = cv2.bitwise_and(frame, frame, mask=m_inv)
        out = cv2.add(part_bg, part_fg)

        # FPS
        fps_cnt += 1
        if time.time() - fps_t0 >= 0.5:
            fps_val = fps_cnt / (time.time() - fps_t0)
            fps_cnt, fps_t0 = 0, time.time()

        # HUD
        put_text(out, f"Mode: {mode.upper()} ({'red' if color=='red' else 'green'} cloak)" if mode=='chroma' else f"Mode: ML Segmentation ({'enabled' if USE_ML else 'unavailable'})", 25)
        put_text(out, f"FPS: {fps_val:.1f} | Smooth alpha: {cfg['smooth_alpha']:.2f}", 50)
        put_text(out, "Keys: Q quit  B bg  C chroma  M ml  1 red  2 green  [/] smooth  R rec  S snap", 75)

        cv2.imshow("Invisibility Cloak", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('b'):
            print("Re-capturing background..."); background = median_background(cap, seconds=1.2); ema_mask=None
        elif k == ord('c'):
            mode = "chroma"; cfg["mode"]=mode
        elif k == ord('m'):
            mode = "ml"; cfg["mode"]=mode
        elif k == ord('1'):
            color = "red"; cfg["chroma_color"]=color
        elif k == ord('2'):
            color = "green"; cfg["chroma_color"]=color
        elif k == ord('['):
            cfg["smooth_alpha"] = max(0.05, cfg["smooth_alpha"] - 0.05)
        elif k == ord(']'):
            cfg["smooth_alpha"] = min(0.95, cfg["smooth_alpha"] + 0.05)
        elif k == ord('r'):
            recording = not recording
            if recording:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(os.path.join("recordings", f"cloak_{ts}.mp4"), fourcc, 20.0, (out.shape[1], out.shape[0]))
                print("Recording ON")
            else:
                if writer: writer.release(); writer=None
                print("Recording OFF")
        elif k == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("recordings", f"snapshot_{ts}.png")
            cv2.imwrite(path, out); print(f"Saved {path}")

        # write frame if recording
        if recording and writer: writer.write(out)

    save_cfg(cfg)
    if writer: writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
