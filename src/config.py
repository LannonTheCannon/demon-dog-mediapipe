"""Central place for every tunable knob.

Keeping these out of main.py means you can tweak behavior without touching
the camera loop — and later stages (gesture detection, compositing) read from
here too, so there's one source of truth.
"""

from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
DEMON_PNG = ASSETS_DIR / "demon.png"

# --- models (MediaPipe Tasks) ---
MODELS_DIR = ROOT / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# --- camera ---
CAM_INDEX = 1            # built-in FaceTime camera. (Index 0 is the iPhone Continuity Camera here.)
                         # Run scripts/probe_cameras.py to re-check if this ever changes.
REQ_WIDTH = 1280         # requested capture width (the driver may pick the nearest supported)
REQ_HEIGHT = 720         # requested capture height
MIRROR = True            # flip horizontally so the feed reads like a mirror (selfie view)

# --- display ---
WINDOW_NAME = "demon-dog"
SHOW_FPS = True
QUIT_KEYS = (ord("q"), 27)   # 'q' or ESC

# --- hand tracking (used from v0 onward; harmless to define now) ---
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
