"""Compositing — drop the demon into the portal box.

This is the swappable seam of the realism ladder. Today it does the simplest
thing: scale the transparent demon PNG to fit the box and alpha-blend it over the
video. Tier 1+ (oriented placement, anchored rig, mesh warp) replaces the body of
`summon()` while everything else in the app stays the same.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import config
from .gesture import PortalBox


def _load_rgba(path) -> np.ndarray:
    """Load an image as BGRA, synthesizing an opaque alpha channel if it lacks one."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f"demon asset not found at {path} — run `python scripts/make_placeholder.py`"
        )
    if img.ndim == 3 and img.shape[2] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    return img


def _similarity_matrix(s1, s2, d1, d2) -> np.ndarray:
    """2x3 affine that maps the segment s1->s2 onto d1->d2 (rotation + uniform
    scale + translation; no shear, so the demon keeps its proportions)."""
    sx, sy = s2[0] - s1[0], s2[1] - s1[1]
    dx, dy = d2[0] - d1[0], d2[1] - d1[1]
    ns = (sx * sx + sy * sy) or 1e-6
    a = (sx * dx + sy * dy) / ns          # = scale * cos(theta)
    b = (sx * dy - sy * dx) / ns          # = scale * sin(theta)
    tx = d1[0] - (a * s1[0] - b * s1[1])
    ty = d1[1] - (b * s1[0] + a * s1[1])
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def _rotate_rgba(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a BGRA image about its center, expanding the canvas so nothing clips.
    Newly exposed pixels are fully transparent."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def overlay_rgba(frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA `overlay` onto a BGR `frame` with its top-left at (x, y).

    Clips cleanly when the overlay would fall partly outside the frame.
    """
    H, W = frame.shape[:2]
    h, w = overlay.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return  # fully off-screen

    ov = overlay[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]

    alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (alpha * ov[:, :, :3] + (1.0 - alpha) * roi).astype(roi.dtype)


class Compositor:
    """Loads the demon asset once, then summons it into a PortalBox on demand."""

    def __init__(self, asset_path=config.DEMON_PNG) -> None:
        self.demon = _load_rgba(asset_path)
        self.last_tilt_deg: float = 0.0           # exposed for the debug HUD

    def summon(self, frame: np.ndarray, box: PortalBox) -> None:
        """Place the demon onto the fox-sign hand and blend it in."""
        if config.ANCHOR_ENABLED:
            self._summon_anchored(frame, box)
            return
        self._summon_oriented(frame, box)

    def _summon_anchored(self, frame: np.ndarray, box: PortalBox) -> None:
        """Tier 3 (anchor rig): pin the demon onto your hand.

        With the snout anchor on, a 3-point affine maps the demon's ears + snout
        onto your index tip, pinky tip, and mouth — so the *face direction* is
        locked to your hand, not just floating below the ears. Without it, a
        2-point similarity pins the ears only.
        """
        h, w = frame.shape[:2]
        dh, dw = self.demon.shape[:2]
        ear_l = (config.DEMON_EAR_L_FRAC[0] * dw, config.DEMON_EAR_L_FRAC[1] * dh)
        ear_r = (config.DEMON_EAR_R_FRAC[0] * dw, config.DEMON_EAR_R_FRAC[1] * dh)

        if config.ANCHOR_USE_SNOUT:
            snout = (config.DEMON_SNOUT_FRAC[0] * dw, config.DEMON_SNOUT_FRAC[1] * dh)
            src = np.float32([ear_l, ear_r, snout])
            dst = np.float32([box.ear_left, box.ear_right, box.mouth])
            m = cv2.getAffineTransform(src, dst)
        else:
            m = _similarity_matrix(ear_l, ear_r, box.ear_left, box.ear_right)

        warped = cv2.warpAffine(self.demon, m, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        overlay_rgba(frame, warped, 0, 0)
        self.last_tilt_deg = box.angle_deg

    def _summon_oriented(self, frame: np.ndarray, box: PortalBox) -> None:
        """Tier 1 (oriented placement): the demon is a sticker on the paper your
        hands frame — centered, scaled to fit, tilted to the frame's absolute angle.
        """
        dh, dw = self.demon.shape[:2]
        if box.w <= 0 or box.h <= 0:
            return

        # contain: largest size that fits inside the box, times a tuning factor
        scale = min(box.w / dw, box.h / dh) * config.DEMON_SCALE
        nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))
        demon = cv2.resize(self.demon, (nw, nh), interpolation=cv2.INTER_AREA)

        if config.ORIENT_ENABLED:
            self.last_tilt_deg = box.angle_deg * config.ORIENT_SIGN
            demon = _rotate_rgba(demon, self.last_tilt_deg)

        # center the (possibly rotated, larger) demon on the portal center
        cx, cy = box.center
        ox = cx - demon.shape[1] // 2
        oy = cy - demon.shape[0] // 2
        overlay_rgba(frame, demon, ox, oy)
