import cv2, time, numpy as np

def capture_median(cap, seconds=1.5, max_frames=60):
    frames=[]; t0=time.time()
    while time.time()-t0 < seconds and len(frames)<max_frames:
        ok,f=cap.read()
        if ok: frames.append(f)
    return np.median(frames,axis=0).astype(np.uint8) if frames else None

def update_dynamic(background, frame, mask_inv, alpha=0.02):
    # mask_inv: 255 where background is visible (don’t overwrite with person)
    m3 = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR) // 255
    bg = background.astype(np.float32)
    bg[m3>0] = (1-alpha)*bg[m3>0] + alpha*frame.astype(np.float32)[m3>0]
    return bg.astype(np.uint8)
