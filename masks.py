import cv2, numpy as np
try:
    import mediapipe as mp
    MP = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
except Exception:
    MP = None

def chroma_mask(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if cfg["chroma_color"]=="red":
        m = cv2.inRange(hsv, np.array(cfg["hsv_red1"]), np.array(cfg["hsv_red2"])) \
          | cv2.inRange(hsv, np.array(cfg["hsv_red3"]), np.array(cfg["hsv_red4"]))
    else:
        m = cv2.inRange(hsv, np.array(cfg["hsv_green1"]), np.array(cfg["hsv_green2"]))
    return m

def ml_mask(frame):
    if MP is None: return np.zeros(frame.shape[:2], np.uint8)
    res = MP.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask = (res.segmentation_mask*255).astype(np.uint8)
    _,mask = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)
    return mask

def hybrid_mask(frame, cfg):
    person = ml_mask(frame)
    cloak  = chroma_mask(frame, cfg)
    return cv2.bitwise_and(person, cloak)

def refine(mask, k=5, alpha=0.35, ema_state=None):
    kernel = np.ones((k,k), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel, iterations=1)
    if ema_state is None:
        ema_state = m.astype(np.float32)
    else:
        ema_state = alpha*m + (1-alpha)*ema_state
    return ema_state.astype(np.uint8), ema_state
