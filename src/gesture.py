"""Finger-frame gesture detection — the heart of the Chainsaw Man effect.

A "finger frame" is both hands making an L (thumb + index extended, the other
three fingers curled), positioned so the two L's outline a rectangle — the way
you'd frame a shot with your hands. When we see it, we compute the *portal box*:
the opening between your hands where the demon will be summoned.

This module is pure geometry over `Hand` objects — no MediaPipe, no OpenCV — so
it's easy to reason about and tune. Thresholds live in config.

Landmark indices (MediaPipe hand model):
    0 wrist
    thumb:  1 cmc, 2 mcp, 3 ip, 4 tip
    index:  5 mcp, 6 pip, 7 dip, 8 tip
    middle: 9 mcp, 10 pip, 11 dip, 12 tip
    ring:   13 mcp, 14 pip, 15 dip, 16 tip
    pinky:  17 mcp, 18 pip, 19 dip, 20 tip
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .hand_tracker import Hand

WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


@dataclass
class PortalBox:
    """Axis-aligned box (in pixels) where the demon gets summoned.

    `points` holds the four framing fingertips (both thumbs + both index tips),
    so downstream placement can recover the frame's *orientation*, not just its
    bounding box.
    """
    x: int
    y: int
    w: int
    h: int
    points: tuple[tuple[int, int], ...] = ()
    angle_deg: float = 0.0   # tilt of the frame's edges (how rotated the "paper" is)

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
    """Outcome of a detection pass. `reason` is for the on-screen debug HUD."""
    detected: bool
    box: PortalBox | None = None
    reason: str = ""


# --- low-level geometry helpers (operate on normalized 0..1 coords) ---

def _d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hand_scale(h: Hand) -> float:
    """A resolution-independent size for the hand: wrist -> middle-finger knuckle."""
    return max(_d(h.landmarks_norm[WRIST], h.landmarks_norm[MIDDLE_MCP]), 1e-6)


def _finger_extended(h: Hand, tip: int, pip: int) -> bool:
    """A finger points away from the palm when its tip is farther from the wrist
    than its middle joint."""
    n = h.landmarks_norm
    return _d(n[tip], n[WRIST]) > _d(n[pip], n[WRIST])


def _thumb_extended(h: Hand) -> bool:
    """Thumb is out when its tip reaches past the IP joint AND splays away from
    the index knuckle."""
    n = h.landmarks_norm
    reaches = _d(n[THUMB_TIP], n[WRIST]) > _d(n[THUMB_IP], n[WRIST])
    splayed = _d(n[THUMB_TIP], n[INDEX_MCP]) > config.THUMB_SPLAY_RATIO * _hand_scale(h)
    return reaches and splayed


def is_L_shape(h: Hand) -> bool:
    """Index + thumb extended, with at least two of middle/ring/pinky curled."""
    if not _finger_extended(h, INDEX_TIP, INDEX_PIP):
        return False
    if not _thumb_extended(h):
        return False
    curled = sum(
        not _finger_extended(h, tip, pip)
        for tip, pip in ((MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
    )
    return curled >= 2


def _seg_angle(h: Hand, i: int, j: int) -> float:
    """Angle (radians) of the landmark i -> j segment, in image coords."""
    p = h.landmarks_px
    return math.atan2(p[j][1] - p[i][1], p[j][0] - p[i][0])


def _frame_angle_deg(a: Hand, b: Hand) -> float:
    """Orientation of the frame's edges = how rotated the 'paper rectangle' is.

    The index fingers form the frame's vertical edges and the thumbs the
    horizontal ones, so each hand gives us two edge directions. We pool all four
    (rotating thumb directions 90° to line them up with the indexes) and take a
    circular mean modulo 180° — orientation, not direction, so opposite-pointing
    fingers on the two hands reinforce instead of cancel. This rotates with your
    wrists even when your hands stay in place.
    """
    angles = []
    for h in (a, b):
        angles.append(_seg_angle(h, INDEX_MCP, INDEX_TIP))
        angles.append(_seg_angle(h, THUMB_IP, THUMB_TIP) + math.pi / 2)
    sx = sum(math.cos(2 * t) for t in angles)
    sy = sum(math.sin(2 * t) for t in angles)
    raw = math.degrees(0.5 * math.atan2(sy, sx))
    # A level frame has vertical index fingers (raw ~= 90deg); subtract that
    # reference and wrap into (-90, 90] so a level frame reads 0 and the value is
    # the frame's absolute tilt from upright.
    return (raw % 180.0) - 90.0


def _portal_from_hands(a: Hand, b: Hand, frame_shape) -> PortalBox:
    """Box spanning the framing fingertips of both hands, with a little padding."""
    keys = [a.landmarks_px[THUMB_TIP], a.landmarks_px[INDEX_TIP],
            b.landmarks_px[THUMB_TIP], b.landmarks_px[INDEX_TIP]]
    xs = [p[0] for p in keys]
    ys = [p[1] for p in keys]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

    h, w = frame_shape[:2]
    pad_x = int((x1 - x0) * config.PORTAL_PADDING)
    pad_y = int((y1 - y0) * config.PORTAL_PADDING)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return PortalBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                     points=tuple(keys), angle_deg=_frame_angle_deg(a, b))


def detect_frame(hands: list[Hand], frame_shape) -> FrameResult:
    """Detect the two-handed finger-frame and return the portal box if found."""
    if len(hands) < 2:
        return FrameResult(False, reason="show both hands")

    a, b = hands[0], hands[1]
    if not (is_L_shape(a) and is_L_shape(b)):
        return FrameResult(False, reason="make an L with each hand (thumb + index)")

    box = _portal_from_hands(a, b, frame_shape)

    # Reject when the hands are basically on top of each other — there's no opening.
    h, w = frame_shape[:2]
    min_w = config.PORTAL_MIN_FRAC * w
    min_h = config.PORTAL_MIN_FRAC * h
    if box.w < min_w or box.h < min_h:
        return FrameResult(False, reason="spread your hands apart")

    return FrameResult(True, box=box, reason="FRAME DETECTED")
