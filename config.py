import json, os
DEFAULT = { "resolution":[1280,720], "mode":"chroma", "chroma_color":"green",
            "hsv_green1":[40,120,60], "hsv_green2":[85,255,255],
            "hsv_red1":[0,120,70], "hsv_red2":[10,255,255],
            "hsv_red3":[170,120,70], "hsv_red4":[180,255,255],
            "morph_kernel":5, "smooth_alpha":0.35, "bg_alpha":0.02,
            "effect_mode":"background" }

def load(path="config.json"):
    if os.path.exists(path):
        try: return json.load(open(path))
        except: pass
    save(DEFAULT, path); return DEFAULT.copy()

def save(cfg, path="config.json"):
    json.dump(cfg, open(path,"w"), indent=2)
