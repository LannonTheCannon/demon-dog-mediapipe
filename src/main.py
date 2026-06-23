"""Entry point — opens the webcam and shows the live feed.

This is the v0 foundation: just prove the capture → display loop works on your
machine. Hand tracking, gesture detection, and the summon get layered on top in
later commits, but the loop you see here stays the spine of the whole thing.

Run:
    python -m src.main              # open the webcam window
    python -m src.main --selftest   # grab a few frames headless (no window) and report

Press 'q' or ESC to quit the window.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from . import config


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a capture device and nudge it toward our requested resolution."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.REQ_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.REQ_HEIGHT)
    return cap


def draw_hud(frame, fps: float) -> None:
    """Overlay a little heads-up text so the window is informative, not just raw video."""
    h = frame.shape[0]
    cv2.putText(frame, "demon-dog  |  v0: camera", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    if config.SHOW_FPS:
        cv2.putText(frame, f"{fps:4.1f} fps", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "press q / esc to quit", (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def selftest(index: int, n_frames: int = 5) -> int:
    """Headless sanity check: can we actually pull frames? No window, no GUI needed.

    Useful for CI or for verifying the camera works over SSH / without a display.
    Returns a process exit code.
    """
    cap = open_camera(index)
    if not cap.isOpened():
        print(f"[selftest] FAILED: could not open camera index {index}", file=sys.stderr)
        return 1
    ok_count = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_count += 1
            if i == 0:
                print(f"[selftest] frame shape: {frame.shape} (h, w, channels)")
    cap.release()
    print(f"[selftest] captured {ok_count}/{n_frames} frames")
    return 0 if ok_count == n_frames else 2


def run(index: int) -> int:
    """Main interactive loop: capture → (mirror) → HUD → display."""
    cap = open_camera(index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {index}. "
              f"Is it in use, or does the terminal lack camera permission?",
              file=sys.stderr)
        return 1

    prev = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: dropped a frame", file=sys.stderr)
                continue

            if config.MIRROR:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # smoothed

            draw_hud(frame, fps)
            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            # window closed via the title-bar X
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="demon-dog webcam loop")
    parser.add_argument("--camera", type=int, default=config.CAM_INDEX,
                        help="camera index (default from config)")
    parser.add_argument("--selftest", action="store_true",
                        help="grab a few frames headless and exit (no window)")
    args = parser.parse_args()

    if args.selftest:
        return selftest(args.camera)
    return run(args.camera)


if __name__ == "__main__":
    raise SystemExit(main())
