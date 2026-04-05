"""
Generate two demo GIFs for the RSSI Indoor Localization project.
GIF 1: Training pipeline terminal animation
GIF 2: Flask API request/response demo
"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Colours ───────────────────────────────────────────────────────────────
BG       = (13,  17,  23)    # GitHub dark bg
FG       = (201, 209, 217)   # default text
GREEN    = (63,  185, 80)    # saved / success
YELLOW   = (241, 196, 15)    # metric values
CYAN     = (88,  217, 240)   # model names
ORANGE   = (240, 136, 62)    # best model
BLUE     = (88,  166, 255)   # prompt
DIMGRAY  = (110, 118, 125)   # separator
TITLEBAR = (33,  38,  45)
RED_DOT  = (255, 95,  87)
YEL_DOT  = (255, 189, 46)
GRN_DOT  = (40,  202, 65)

W, H = 800, 460
FONT_SIZE = 14
PAD_X, PAD_Y = 24, 20
LINE_H = 22

def load_font(size=FONT_SIZE):
    """Try to load a monospace font, fall back to default."""
    candidates = [
        "C:/Windows/Fonts/consola.ttf",   # Consolas
        "C:/Windows/Fonts/cour.ttf",      # Courier New
        "C:/Windows/Fonts/lucon.ttf",     # Lucida Console
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT      = load_font(FONT_SIZE)
FONT_BOLD = load_font(FONT_SIZE)   # PIL TTF doesn't do bold; same font is fine

def draw_window(title):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # Title bar
    d.rectangle([0, 0, W, 38], fill=TITLEBAR)
    for i, col in enumerate([RED_DOT, YEL_DOT, GRN_DOT]):
        cx = 18 + i * 22
        d.ellipse([cx-6, 13, cx+6, 25], fill=col)
    d.text((80, 11), title, font=FONT, fill=(150, 150, 150))
    return img, d

def text_lines(d, lines, start_y):
    """Draw list of (text, colour) tuples starting at start_y."""
    y = start_y
    for text, col in lines:
        if text is not None:
            d.text((PAD_X, y), text, font=FONT, fill=col)
        y += LINE_H
    return y

# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 — Training pipeline
# ═══════════════════════════════════════════════════════════════════════════
TRAIN_SCRIPT = [
    ("~/rssi-indoor-localization $ python main.py", BLUE),
]

TRAIN_OUTPUT = [
    ("", FG),
    ("=== RSSI Indoor Localization ===", FG),
    ("", FG),
    ("[SAVED] models/knn.pkl",    GREEN),
    ("[SAVED] models/mlp.pkl",    GREEN),
    ("[SAVED] models/svm.pkl",    GREEN),
    ("[SAVED] models/scaler.pkl", GREEN),
    ("", FG),
    ("=" * 42,                    DIMGRAY),
    ("KNN",                       CYAN),
    ("  Accuracy : 90.0%",        YELLOW),
    ("  F1 Score : 0.8980",       YELLOW),
    ("  Mean Distance Error: 0.21 m", YELLOW),
    ("=" * 42,                    DIMGRAY),
    ("", FG),
    ("=" * 42,                    DIMGRAY),
    ("MLP",                       CYAN),
    ("  Accuracy : 80.0%",        YELLOW),
    ("  F1 Score : 0.7910",       YELLOW),
    ("  Mean Distance Error: 0.41 m", YELLOW),
    ("=" * 42,                    DIMGRAY),
    ("", FG),
    ("=" * 42,                    DIMGRAY),
    ("SVM",                       CYAN),
    ("  Accuracy : 91.1%",        YELLOW),
    ("  F1 Score : 0.9083",       YELLOW),
    ("  Mean Distance Error: 0.19 m", YELLOW),
    ("=" * 42,                    DIMGRAY),
    ("", FG),
    ("Best model: SVM (91.1%)",   ORANGE),
    ("", FG),
    ("[SAVED] outputs/confusion_matrices.png", GREEN),
    ("[SAVED] outputs/floor_map.png",          GREEN),
    ("[SAVED] outputs/model_comparison.png",   GREEN),
    ("", FG),
    ("All outputs saved to outputs/",          FG),
    ("Run 'python app.py' to start Flask API", FG),
]

def make_training_gif():
    frames, durations = [], []

    all_lines = TRAIN_SCRIPT + TRAIN_OUTPUT
    # Build frames: reveal one line at a time
    for reveal in range(1, len(all_lines) + 1):
        img, d = draw_window("rssi-indoor-localization — python main.py")
        visible = all_lines[:reveal]
        text_lines(d, visible, PAD_Y + 38)
        frames.append(img)
        # Slower for key lines, fast for separators
        line_text = all_lines[reveal-1][0]
        if "===" in line_text or line_text == "":
            durations.append(80)
        elif any(k in line_text for k in ["Best model", "RSSI Indoor", "All outputs", "Run '"]):
            durations.append(400)
        elif "SAVED" in line_text:
            durations.append(220)
        elif any(m in line_text for m in ["KNN", "MLP", "SVM"]) and len(line_text) <= 4:
            durations.append(300)
        elif "Accuracy" in line_text or "F1" in line_text or "Mean" in line_text:
            durations.append(180)
        else:
            durations.append(120)

    # Hold last frame
    for _ in range(8):
        frames.append(frames[-1].copy())
        durations.append(500)

    path = os.path.join(OUT, "demo_training.gif")
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=False
    )
    print(f"[SAVED] {path}  ({len(frames)} frames)")

# ═══════════════════════════════════════════════════════════════════════════
# GIF 2 — Flask API demo
# ═══════════════════════════════════════════════════════════════════════════
API_STEPS = [
    # (hold_ms, lines_so_far)
    # Step 0: start server
    (600, [
        ("~/rssi-indoor-localization $ python app.py", BLUE),
        ("", FG),
        (" * Running on http://127.0.0.1:5000", GREEN),
        (" * Debug mode: on", FG),
    ]),
    # Step 1: GET /
    (500, [
        ("", FG),
        ("$ curl http://localhost:5000/", BLUE),
    ]),
    (600, [
        ("", FG),
        ("$ curl http://localhost:5000/", BLUE),
        ("{", FG),
        ('  "project": "RSSI Indoor Localization",', FG),
        ('  "endpoints": {', FG),
        ('    "GET  /":          "This help message",', CYAN),
        ('    "POST /predict":   "Predict location from RSSI values",', CYAN),
        ('    "GET  /locations": "List all reference locations"', CYAN),
        ("  }", FG),
        ("}", FG),
    ]),
    # Step 2: POST /predict
    (500, [
        ("", FG),
        ('$ curl -X POST http://localhost:5000/predict \\', BLUE),
        ('    -d \'{"rssi": [-60,-55,-70,-65,-50,-75]}\'', BLUE),
    ]),
    (800, [
        ("", FG),
        ('$ curl -X POST http://localhost:5000/predict \\', BLUE),
        ('    -d \'{"rssi": [-60,-55,-70,-65,-50,-75]}\'', BLUE),
        ("", FG),
        ("{", FG),
        ('  "predicted_location_id":   4,', YELLOW),
        ('  "predicted_location_name": "Center Hub",', YELLOW),
        ('  "confidence":              80.0,', YELLOW),
        ('  "all_probabilities": {', FG),
        ('    "Center Hub":    80.0,', GREEN),
        ('    "Meeting Room":  10.0,', FG),
        ('    "Lab East":       6.0,', FG),
        ('    "...":           "..."', DIMGRAY),
        ("  }", FG),
        ("}", FG),
    ]),
    # Step 3: GET /locations
    (500, [
        ("", FG),
        ("$ curl http://localhost:5000/locations", BLUE),
    ]),
    (700, [
        ("", FG),
        ("$ curl http://localhost:5000/locations", BLUE),
        ("", FG),
        ('{ "0":"Entrance", "1":"Corridor A", "2":"Corridor B",', FG),
        ('  "3":"Lab West", "4":"Center Hub", "5":"Lab East",',  FG),
        ('  "6":"Office A", "7":"Meeting Room", "8":"Office B" }', FG),
    ]),
]

def make_api_gif():
    frames, durations = [], []

    header = [
        ("~/rssi-indoor-localization $ python app.py", BLUE),
        ("", FG),
        (" * Running on http://127.0.0.1:5000", GREEN),
        (" * Debug mode: on", FG),
        ("", FG),
    ]

    for hold_ms, step_lines in API_STEPS:
        img, d = draw_window("rssi-indoor-localization — Flask API  •  localhost:5000")
        text_lines(d, header + step_lines, PAD_Y + 38)
        frames.append(img)
        durations.append(hold_ms)

    # Hold last frame
    for _ in range(6):
        frames.append(frames[-1].copy())
        durations.append(600)

    path = os.path.join(OUT, "demo_api.gif")
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=False
    )
    print(f"[SAVED] {path}  ({len(frames)} frames)")

if __name__ == "__main__":
    make_training_gif()
    make_api_gif()
    print("Done!")
