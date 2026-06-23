"""Compositing — drop the demon into the portal box.

This is the swappable seam of the realism ladder. Today it does the simplest
thing: scale the transparent demon PNG to fit the box and alpha-blend it over the
video. Tier 1+ (oriented placement, anchored rig, mesh warp) replaces the body of
`summon()` while everything else in the app stays the same.
"""

from __future__ import annotations

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

    def summon(self, frame: np.ndarray, box: PortalBox) -> None:
        """Scale the demon to fit `box` (preserving aspect) and blend it in, centered."""
        dh, dw = self.demon.shape[:2]
        if box.w <= 0 or box.h <= 0:
            return

        # contain: largest size that fits inside the box, times a tuning factor
        scale = min(box.w / dw, box.h / dh) * config.DEMON_SCALE
        nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))
        resized = cv2.resize(self.demon, (nw, nh), interpolation=cv2.INTER_AREA)

        ox = box.x + (box.w - nw) // 2
        oy = box.y + (box.h - nh) // 2
        overlay_rgba(frame, resized, ox, oy)
