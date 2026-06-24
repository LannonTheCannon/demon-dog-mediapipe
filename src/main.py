"""Entry point — webcam loop: hand tracking → fox sign → summon the demon.

capture → mirror → hand tracking → detect the one-handed fox sign ("Kon") →
pin the demon fox's ears to your index + pinky tips. Press 'd' to toggle the
debug overlay (skeleton + box). This loop stays the spine of the project.

Run:
    python -m src.main              # open the webcam window
    python -m src.main --selftest   # grab a few frames headless (no window) and report

Press 'q' or ESC to quit the window.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from . import config
from .compositor import Compositor
from .gesture import FrameResult, PortalBox, detect_fox
from .hand_tracker import HAND_CONNECTIONS, Hand, HandTracker

# Landmark indices we'll lean on later (thumb + finger tips). Highlighted now so
# you can see they're tracked cleanly before any gesture logic depends on them.
FINGERTIPS = (4, 8, 12, 16, 20)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a capture device and nudge it toward our requested resolution."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.REQ_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.REQ_HEIGHT)
    return cap


def _delivers_live_frames(cap: cv2.VideoCapture, n: int = 5) -> bool:
    """True if the camera returns real (non-black) frames — guards against the idle
    iPhone Continuity Camera, which opens fine but hands back black frames."""
    vals = []
    for _ in range(n):
        ok, frame = cap.read()
        if ok and frame is not None:
            vals.append(float(np.mean(frame)))
    return bool(vals) and (sum(vals) / len(vals) > 1.0)


def find_camera(preferred: int):
    """Return (capture, index) for the first camera that delivers live frames,
    trying `preferred` first then probing 0..CAMERA_PROBE_MAX. (None, -1) if none."""
    order = [preferred] + [i for i in range(config.CAMERA_PROBE_MAX) if i != preferred]
    for i in order:
        cap = open_camera(i)
        if cap.isOpened() and _delivers_live_frames(cap):
            print(f"[camera] using index {i}")
            return cap, i
        cap.release()
    return None, -1


def draw_hud(frame, fps: float, n_hands: int, gesture: FrameResult, debug: bool) -> None:
    """Overlay a little heads-up text so the window is informative, not just raw video."""
    h = frame.shape[0]
    cv2.putText(frame, "demon-dog  |  Kon (fox sign)", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    if config.SHOW_FPS:
        cv2.putText(frame, f"{fps:4.1f} fps   hands: {n_hands}", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)

    # gesture status: green when the frame is locked, amber as a hint otherwise
    color = (60, 255, 120) if gesture.detected else (60, 200, 255)
    cv2.putText(frame, gesture.reason, (16, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    hint = f"q/esc quit   |   k: KON   |   d: debug {'ON' if debug else 'OFF'}   |   s: screenshot"
    cv2.putText(frame, hint, (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_portal(frame, box: PortalBox) -> None:
    """Draw the portal box where the demon will be summoned (v2)."""
    cv2.rectangle(frame, box.p1, box.p2, (40, 60, 255), 3, cv2.LINE_AA)   # bold red-orange
    cv2.rectangle(frame, box.p1, box.p2, (180, 220, 255), 1, cv2.LINE_AA)  # inner highlight
    cx, cy = box.center
    cv2.drawMarker(frame, (cx, cy), (200, 230, 255), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)


def draw_cone(frame, box: PortalBox) -> None:
    """Translucent light-blue 3D cone from the hole center along the palm normal.

    Because it's a projected 3D cone: aim at the camera → it reads as a ring opening
    toward you; aim sideways/through the hole → it becomes a directional cone. The
    normal is de-mirrored by handedness upstream, so left and right hands match.
    """
    apex = np.array([box.hole_center[0], box.hole_center[1], 0.0])
    axis = np.array(box.normal, dtype=float)
    axis /= (np.linalg.norm(axis) or 1e-6)
    base_c = apex + axis * config.CONE_LENGTH

    # orthonormal basis perpendicular to the axis -> the base circle
    ref = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref); u /= (np.linalg.norm(u) or 1e-6)
    v = np.cross(axis, u)
    rim = []
    for k in range(32):
        th = 2 * math.pi * k / 32
        p = base_c + config.CONE_RADIUS * (math.cos(th) * u + math.sin(th) * v)
        rim.append((int(p[0]), int(p[1])))            # orthographic projection
    rim_np = np.array(rim, np.int32)
    apex2 = (int(apex[0]), int(apex[1]))

    # translucent fill: the cone body (hull of apex + base) plus the base disk
    overlay = frame.copy()
    hull = cv2.convexHull(np.vstack([rim_np, np.array([apex2], np.int32)]))
    cv2.fillPoly(overlay, [hull], config.CONE_COLOR)
    cv2.fillPoly(overlay, [rim_np], config.CONE_COLOR)
    cv2.addWeighted(overlay, config.CONE_ALPHA, frame, 1.0 - config.CONE_ALPHA, 0, frame)

    # outlines so it reads as a cone
    cv2.polylines(frame, [rim_np], True, config.CONE_COLOR, 2, cv2.LINE_AA)
    for idx in range(0, 32, 8):
        cv2.line(frame, apex2, rim[idx], config.CONE_COLOR, 1, cv2.LINE_AA)


def draw_hand(frame, hand: Hand) -> None:
    """Draw one hand's skeleton: connections, joints, and highlighted fingertips."""
    px = hand.landmarks_px

    # bones
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 200, 120), 2, cv2.LINE_AA)

    # joints
    for i, p in enumerate(px):
        if i in FINGERTIPS:
            cv2.circle(frame, p, 7, (40, 120, 255), -1, cv2.LINE_AA)   # fingertips pop
        else:
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)

    # label near the wrist
    cv2.putText(frame, f"{hand.label} {hand.score:.0%}", (px[0][0] - 10, px[0][1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)


def save_screenshot(frame, count: int):
    """Save the rendered frame to captures/ so the result can be reviewed offline."""
    config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CAPTURES_DIR / f"capture_{count:03d}.png"
    cv2.imwrite(str(path), frame)
    print(f"[screenshot] saved {path}")
    return path


def selftest(index: int, n_frames: int = 5) -> int:
    """Headless sanity check: can we actually pull frames? No window, no GUI needed.

    Useful for CI or for verifying the camera works over SSH / without a display.
    Returns a process exit code.
    """
    cap, used = find_camera(index)
    if cap is None:
        print(f"[selftest] FAILED: no working camera (tried {index}, then 0..{config.CAMERA_PROBE_MAX - 1})",
              file=sys.stderr)
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
    """Main interactive loop: capture → mirror → track → detect → summon → display."""
    cap, used = find_camera(index)
    if cap is None:
        print(f"ERROR: no working camera found (tried {index}, then 0..{config.CAMERA_PROBE_MAX - 1}). "
              f"Is it in use, or does the terminal lack camera permission?",
              file=sys.stderr)
        return 1

    prev = time.perf_counter()
    fps = 0.0
    debug = config.DEBUG_OVERLAY
    shots = 0
    tracker = HandTracker()
    compositor = Compositor()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: dropped a frame", file=sys.stderr)
                continue

            if config.MIRROR:
                frame = cv2.flip(frame, 1)

            # Track hands on the same (mirrored) frame we display, so landmarks
            # and the portal line up with what you see.
            hands = tracker.process(frame)
            gesture = detect_fox(hands, frame.shape)

            # the summon: pin the fox to the hand (paused while an eruption plays)
            if gesture.detected and gesture.box is not None and not compositor.is_erupting():
                compositor.summon(frame, gesture.box)

            # debug overlays (toggle live with 'd')
            if debug and not compositor.is_erupting():
                for hand in hands:
                    draw_hand(frame, hand)
                if gesture.detected and gesture.box is not None:
                    draw_portal(frame, gesture.box)
                    draw_cone(frame, gesture.box)
                    cv2.putText(frame, f"tilt: {compositor.last_tilt_deg:+.0f} deg", (16, 118),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 1, cv2.LINE_AA)
                    # diagnostic readout: handedness + de-mirrored normal
                    if hands:
                        n = gesture.box.normal
                        cv2.putText(frame,
                                    f"{hands[0].label}  n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})",
                                    (16, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            # the eruption renders on top of everything
            if compositor.draw_eruption(frame):
                cv2.putText(frame, "KON!", (frame.shape[1] // 2 - 70, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 2.2, (60, 60, 255), 4, cv2.LINE_AA)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # smoothed

            draw_hud(frame, fps, len(hands), gesture, debug)
            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("s"):
                shots += 1
                save_screenshot(frame, shots)
            if key == config.ERUPT_KEY and gesture.detected and gesture.box is not None:
                compositor.trigger_eruption(gesture.box)
            # window closed via the title-bar X
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
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
