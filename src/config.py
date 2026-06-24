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
CAPTURES_DIR = ROOT / "captures"   # screenshots saved with the 's' key
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
DEBUG_OVERLAY = True         # draw hand skeleton + portal box (toggle live with 'd')

# --- summon (v2) ---
DEMON_SCALE = 1.0           # demon size as a fraction of the fitted portal box (tune to taste)

# --- oriented placement (Tier 1) ---
ORIENT_ENABLED = True       # rotate the demon to match the finger-frame's tilt
ORIENT_SIGN = -1.0          # makes the demon turn the SAME way as your frame; flip to +1.0 if reversed

# --- anchor rig (Tier 3) ---
ANCHOR_ENABLED = True       # pin the demon's ears to your fingertips (supersedes plain oriented placement)
# Ear-tip + snout positions inside the demon image, as fractions of (width, height).
# These match the generated placeholder; update if you swap the art.
DEMON_EAR_L_FRAC = (0.266, 0.117)
DEMON_EAR_R_FRAC = (0.734, 0.117)
DEMON_SNOUT_FRAC = (0.500, 0.887)   # the nose/snout — the third anchor that pins the face direction

ANCHOR_USE_SNOUT = True             # 3-point pin (ears + snout). False = 2-point (ears only).
SNOUT_WEIGHT = 0.35                 # how much the snout pulls vs the ears (lower = ears pin tighter)

# --- facing cone (RPG-style, points where the snout points) ---
AIM_FLAT_FACING = 0.6      # |normal.z| above this = palm flat to camera; below = looking through the hole
CONE_LENGTH = 150          # how far the facing wedge extends past the snout (px)
CONE_HALF_ANGLE = 26       # half-width of the wedge (degrees)
CONE_ALPHA = 0.35          # translucency of the wedge fill
CONE_COLOR = (255, 210, 130)   # light blue (BGR)

# --- hand tracking (used from v0 onward; harmless to define now) ---
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

# --- finger-frame gesture (v1) ---
THUMB_SPLAY_RATIO = 0.5   # thumb tip must sit this far from the index knuckle (× hand size) to count as "out"
PORTAL_PADDING = 0.15     # grow the portal box by this fraction beyond the fingertips
PORTAL_MIN_FRAC = 0.12    # reject frames smaller than this fraction of the image (hands too close together)
