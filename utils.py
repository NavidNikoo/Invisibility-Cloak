import cv2, time

class FPSTimer:
    def __init__(self, interval=0.5): self.t0=time.time(); self.c=0; self.val=0; self.int=interval
    def tick(self):
        self.c+=1
        if time.time()-self.t0>=self.int:
            self.val=self.c/(time.time()-self.t0); self.c=0; self.t0=time.time()
        return self.val

def hud(img, lines, start_y=24):
    for i,txt in enumerate(lines):
        cv2.putText(img, txt, (10, start_y+25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
