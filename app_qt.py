# app_qt.py — Invisibility Cloak (Qt GUI)
import sys, os, datetime
import cv2, numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QRadioButton, QButtonGroup, QSlider, QCheckBox, QGroupBox,
    QFileDialog, QStatusBar
)

from config import load, save
from background import capture_median, update_dynamic
from masks import chroma_mask, ml_mask, hybrid_mask, refine
from effects import effect, inpaint
from utils import FPSTimer

EFFECTS = ["background", "inpaint", "blur", "pixelate", "heatmap", "glitch"]

def bgr_to_qpixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

class CloakApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invisibility Cloak – Qt")
        self.cfg = load()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg["resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["resolution"][1])

        self.background = capture_median(self.cap, seconds=1.5)
        if self.background is None:
            raise RuntimeError("Camera error: could not capture background.")

        self.ema = None
        self.mode = self.cfg.get("mode", "chroma")
        self.effect_mode = self.cfg.get("effect_mode", "background")
        self.dyn_bg = False
        self.recording = False
        self.writer = None
        os.makedirs("recordings", exist_ok=True)
        self.fps = FPSTimer()

        self._build_ui()
        self._wire_events()

        # timer for video loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)  # ~33 FPS target

    # -------- UI ----------
    def _build_ui(self):
        # Preview
        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111; color:#bbb;")

        # Modes
        mode_box = QGroupBox("Mode")
        self.btn_chroma = QRadioButton("Chroma")
        self.btn_ml     = QRadioButton("ML")
        self.btn_hybrid = QRadioButton("Hybrid")
        for w in (self.btn_chroma, self.btn_ml, self.btn_hybrid): w.setChecked(False)
        {"chroma":self.btn_chroma, "ml":self.btn_ml, "hybrid":self.btn_hybrid}[self.mode].setChecked(True)
        g_mode = QButtonGroup(self)
        g_mode.addButton(self.btn_chroma); g_mode.addButton(self.btn_ml); g_mode.addButton(self.btn_hybrid)
        hb_mode = QHBoxLayout(); hb_mode.addWidget(self.btn_chroma); hb_mode.addWidget(self.btn_ml); hb_mode.addWidget(self.btn_hybrid)
        mode_box.setLayout(hb_mode)

        # Color
        color_box = QGroupBox("Cloak Color")
        self.btn_red   = QRadioButton("Red")
        self.btn_green = QRadioButton("Green")
        (self.btn_red if self.cfg.get("chroma_color","green")=="red" else self.btn_green).setChecked(True)
        g_color = QButtonGroup(self); g_color.addButton(self.btn_red); g_color.addButton(self.btn_green)
        hb_col = QHBoxLayout(); hb_col.addWidget(self.btn_red); hb_col.addWidget(self.btn_green)
        color_box.setLayout(hb_col)

        # Effect
        eff_box = QGroupBox("Effect")
        self.combo_effect = QComboBox()
        self.combo_effect.addItems(EFFECTS)
        self.combo_effect.setCurrentIndex(EFFECTS.index(self.effect_mode))
        self.chk_dynbg = QCheckBox("Dynamic Background")
        hb_eff = QHBoxLayout(); hb_eff.addWidget(self.combo_effect); hb_eff.addWidget(self.chk_dynbg)
        eff_box.setLayout(hb_eff)

        # Sliders
        slid_box = QGroupBox("Tuning")
        self.sld_smooth = QSlider(Qt.Horizontal)
        self.sld_smooth.setRange(5, 95)
        self.sld_smooth.setValue(int(float(self.cfg["smooth_alpha"])*100))
        self.sld_kernel = QSlider(Qt.Horizontal)
        self.sld_kernel.setRange(3, 11)
        self.sld_kernel.setValue(int(self.cfg["morph_kernel"]))
        v_s = QVBoxLayout()
        v_s.addWidget(QLabel("Smoothing"))
        v_s.addWidget(self.sld_smooth)
        v_s.addWidget(QLabel("Kernel (odd)"))
        v_s.addWidget(self.sld_kernel)
        slid_box.setLayout(v_s)

        # Buttons
        hb_buttons = QHBoxLayout()
        self.btn_bg   = QPushButton("Recapture BG")
        self.btn_snap = QPushButton("Snapshot")
        self.btn_rec  = QPushButton("Record")
        self.btn_quit = QPushButton("Quit")
        for b in (self.btn_bg, self.btn_snap, self.btn_rec, self.btn_quit):
            b.setMinimumWidth(120)
        hb_buttons.addWidget(self.btn_bg); hb_buttons.addWidget(self.btn_snap)
        hb_buttons.addWidget(self.btn_rec); hb_buttons.addWidget(self.btn_quit)

        # Status bar
        self.status = QStatusBar()
        self.status.showMessage("Ready")

        # Layout
        root = QVBoxLayout()
        root.addWidget(self.preview)
        root.addWidget(mode_box)
        root.addWidget(color_box)
        root.addWidget(eff_box)
        root.addWidget(slid_box)
        root.addLayout(hb_buttons)
        root.addWidget(self.status)
        self.setLayout(root)

    def _wire_events(self):
        self.btn_chroma.toggled.connect(lambda v: v and self._set_mode("chroma"))
        self.btn_ml.toggled.connect(lambda v: v and self._set_mode("ml"))
        self.btn_hybrid.toggled.connect(lambda v: v and self._set_mode("hybrid"))

        self.btn_red.toggled.connect(lambda v: v and self._set_color("red"))
        self.btn_green.toggled.connect(lambda v: v and self._set_color("green"))

        self.combo_effect.currentIndexChanged.connect(self._set_effect)
        self.chk_dynbg.stateChanged.connect(lambda s: self._set_dynbg(s == Qt.Checked))

        self.sld_smooth.valueChanged.connect(self._set_smooth)
        self.sld_kernel.valueChanged.connect(self._set_kernel)

        self.btn_bg.clicked.connect(self._recapture_bg)
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_rec.clicked.connect(self._toggle_record)
        self.btn_quit.clicked.connect(self._quit)

    # -------- Actions ----------
    def _set_mode(self, m):
        self.mode = m; self.cfg["mode"] = m

    def _set_color(self, c):
        self.cfg["chroma_color"] = c

    def _set_effect(self, idx):
        self.effect_mode = EFFECTS[idx]; self.cfg["effect_mode"] = self.effect_mode

    def _set_dynbg(self, on):
        self.dyn_bg = bool(on)

    def _set_smooth(self, val):
        self.cfg["smooth_alpha"] = max(0.05, min(0.95, val/100.0))

    def _set_kernel(self, k):
        k = max(3, min(11, k))
        self.cfg["morph_kernel"] = k if k % 2 == 1 else k - 1

    def _recapture_bg(self):
        self.ema = None
        self.background = capture_median(self.cap, seconds=1.2)
        self.status.showMessage("Background recaptured", 1500)

    def _snapshot(self):
        if not hasattr(self, "last_out"): return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("recordings", f"snapshot_{ts}.png")
        cv2.imwrite(path, self.last_out)
        self.status.showMessage(f"Saved {path}", 2000)

    def _toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = self.last_out.shape[:2] if hasattr(self, "last_out") else (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.writer = cv2.VideoWriter(os.path.join("recordings", f"cloak_{ts}.mp4"), fourcc, 20.0, (w, h))
            self.btn_rec.setText("Stop")
            self.status.showMessage("Recording ON", 1500)
        else:
            if self.writer: self.writer.release(); self.writer = None
            self.btn_rec.setText("Record")
            self.status.showMessage("Recording OFF", 1500)

    def _quit(self):
        self.close()

    # -------- Frame loop ----------
    def _tick(self):
        ok, frame = self.cap.read()
        if not ok: return
        frame = cv2.flip(frame, 1)

        # build mask by mode
        if self.mode == "ml":
            m = ml_mask(frame)
        elif self.mode == "hybrid":
            m = hybrid_mask(frame, self.cfg)
        else:
            m = chroma_mask(frame, self.cfg)

        # refine mask (morph + EMA smoothing)
        m, self.ema = refine(m, k=self.cfg["morph_kernel"], alpha=self.cfg["smooth_alpha"], ema_state=self.ema)

        # dynamic background update
        if self.dyn_bg:
            m_inv = cv2.bitwise_not(m)
            self.background = update_dynamic(self.background, frame, m_inv, self.cfg["bg_alpha"])

        # compose
        if self.effect_mode == "inpaint":
            out = inpaint(frame, m, radius=3)
        else:
            out = effect(frame, m, kind=self.effect_mode, background=self.background)

        # draw to preview
        self.last_out = out
        self.preview.setPixmap(bgr_to_qpixmap(out))

        # status (FPS + params)
        fps_val = self.fps.tick()
        self.status.showMessage(
            f"Mode: {self.mode.upper()}  |  Effect: {self.effect_mode}  |  FPS: {fps_val:.1f}  |  "
            f"Smoothing: {self.cfg['smooth_alpha']:.2f}  |  Kernel: {self.cfg['morph_kernel']}  |  DynBG: {self.dyn_bg}"
        )

        # record if enabled
        if self.recording and self.writer:
            self.writer.write(out)

    # -------- Cleanup ----------
    def closeEvent(self, event):
        save(self.cfg)
        if self.writer: self.writer.release()
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ui = CloakApp()
    ui.resize(900, 900)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
