"""One-handed fox-sign detection — Aki's "Kon" from Chainsaw Man.

The summon gesture is the kitsune (fox) hand sign on a single hand: index and
pinky extended as the ears, middle and ring folded down toward the thumb as the
snout. When we see it we expose the two ear anchors (index tip + pinky tip) so
the compositor can pin the demon fox's ears to your fingertips.

Pure geometry over a `Hand` object — no MediaPipe, no OpenCV.

Landmark indices: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .hand_tracker import Hand

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


@dataclass
class PortalBox:
    """Where/how the demon is summoned for one hand.

    `ear_left`/`ear_right` are the fingertip anchors (index & pinky tips, ordered
    left→right) that the demon's ears get pinned to. x/y/w/h bound the hand for
    the debug box and the oriented fallback.
    """
    x: int
    y: int
    w: int
    h: int
    angle_deg: float = 0.0
    ear_left: tuple[int, int] = (0, 0)
    ear_right: tuple[int, int] = (0, 0)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def p1(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def p2(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h)


@dataclass
class FrameResult:
    """Outcome of a detection pass. `reason` drives the on-screen HUD."""
    detected: bool
    box: PortalBox | None = None
    reason: str = ""


# --- geometry helpers (normalized 0..1 coords) ---

def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_extended(h: Hand, tip: int, pip: int) -> bool:
    """A finger points away from the palm when its tip is farther from the wrist
    than its middle joint."""
    n = h.landmarks_norm
    return _d(n[tip], n[WRIST]) > _d(n[pip], n[WRIST])


def is_fox_sign(h: Hand) -> bool:
    """Kitsune sign: index + pinky extended (ears), middle + ring folded (snout)."""
    return (
        _finger_extended(h, INDEX_TIP, INDEX_PIP)
        and _finger_extended(h, PINKY_TIP, PINKY_PIP)
        and not _finger_extended(h, MIDDLE_TIP, MIDDLE_PIP)
        and not _finger_extended(h, RING_TIP, RING_PIP)
    )


def _hand_box(h: Hand, frame_shape):
    """Padded bounding box (pixels) around the whole hand."""
    xs = [p[0] for p in h.landmarks_px]
    ys = [p[1] for p in h.landmarks_px]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    H, W = frame_shape[:2]
    pad_x = int((x1 - x0) * config.PORTAL_PADDING)
    pad_y = int((y1 - y0) * config.PORTAL_PADDING)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x)
    y1 = min(H, y1 + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def _fox_angle_deg(h: Hand) -> float:
    """Tilt of the fox from upright: the wrist→ears axis measured from vertical."""
    wx, wy = h.landmarks_px[WRIST]
    ix, iy = h.landmarks_px[INDEX_TIP]
    pkx, pky = h.landmarks_px[PINKY_TIP]
    ear_mid = ((ix + pkx) / 2.0, (iy + pky) / 2.0)
    dx, dy = ear_mid[0] - wx, ear_mid[1] - wy
    return math.degrees(math.atan2(dx, -dy))   # 0 when the ears are straight above the wrist


def detect_fox(hands: list[Hand], frame_shape) -> FrameResult:
    """Detect the one-handed fox sign and return the ear anchors if found."""
    if not hands:
        return FrameResult(False, reason="show your hand")

    h = hands[0]
    if not is_fox_sign(h):
        return FrameResult(False, reason="fox sign: index + pinky up, fold middle + ring")

    x, y, w, hh = _hand_box(h, frame_shape)
    ears = sorted([h.landmarks_px[INDEX_TIP], h.landmarks_px[PINKY_TIP]], key=lambda p: p[0])
    box = PortalBox(x, y, w, hh, angle_deg=_fox_angle_deg(h),
                    ear_left=tuple(ears[0]), ear_right=tuple(ears[1]))
    return FrameResult(True, box=box, reason="KON")
