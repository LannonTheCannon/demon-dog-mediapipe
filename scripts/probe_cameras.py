"""Probe camera indices 0-3 and report which one delivers real (non-black) pixels.

Handy on macOS, where the iPhone Continuity Camera often grabs index 0 and the
built-in FaceTime camera ends up on index 1. Whichever index prints
'LIVE IMAGE' is the one to put in config.CAM_INDEX.

Run:
    python scripts/probe_cameras.py
"""

import cv2
import numpy as np

MAX_INDEX = 4
FRAMES = 10
BLACK_THRESHOLD = 1.0  # mean pixel brightness below this == effectively black


def probe(index: int) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"index {index}: not opened")
        cap.release()
        return
    brightness = []
    for _ in range(FRAMES):
        ok, frame = cap.read()
        if ok and frame is not None:
            brightness.append(float(np.mean(frame)))
    cap.release()
    if not brightness:
        print(f"index {index}: opened but no frames")
        return
    mean = sum(brightness) / len(brightness)
    verdict = "BLACK (idle/no-permission)" if mean < BLACK_THRESHOLD else "LIVE IMAGE ✅"
    print(f"index {index}: {len(brightness)} frames, brightness {mean:5.1f}  ->  {verdict}")


def main() -> None:
    print("probing cameras (the live one's green light will turn on)...")
    for i in range(MAX_INDEX):
        probe(i)


if __name__ == "__main__":
    main()
