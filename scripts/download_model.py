"""Download the MediaPipe hand-landmarker model into models/.

Optional — HandTracker downloads it automatically on first run — but handy for
pre-fetching (e.g. before going offline).

Run:
    python scripts/download_model.py
"""

from src.models import ensure_hand_model

if __name__ == "__main__":
    print("model ready at:", ensure_hand_model())
