import cv2, numpy as np

def compose_background(frame, background, mask):
    inv = cv2.bitwise_not(mask)
    return cv2.add(cv2.bitwise_and(background, background, mask=mask),
                   cv2.bitwise_and(frame, frame, mask=inv))

def inpaint(frame, mask, radius=3):
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)

def effect(frame, mask, kind="background", background=None):
    if kind=="background" and background is not None:
        return compose_background(frame, background, mask)
    if kind=="blur":
        blur = cv2.GaussianBlur(frame,(21,21),0)
        return np.where(mask[...,None]>0, blur, frame)
    if kind=="pixelate":
        h,w=frame.shape[:2]; small=cv2.resize(frame,(w//20,h//20))
        pix=cv2.resize(small,(w,h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask[...,None]>0, pix, frame)
    if kind=="heatmap":
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heat=cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return np.where(mask[...,None]>0, heat, frame)
    if kind=="glitch":
        shift=8; gl=frame.copy(); gl[:,shift:,2]=frame[:,:-shift,2]
        return np.where(mask[...,None]>0, gl, frame)
    return frame
