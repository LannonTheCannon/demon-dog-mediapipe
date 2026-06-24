"""MediaPipe hand tracking, wrapped so the rest of the app never touches MediaPipe.

Why the wrapper: downstream code (gesture detection, compositing) shouldn't care
that landmarks come from MediaPipe. This module takes a BGR frame and returns
plain `Hand` objects with pixel + normalized coordinates. Swap the backend and
nothing else has to change.

Uses the MediaPipe **Tasks** API (HandLandmarker) — the current, supported path;
the older `mp.solutions.hands` API isn't shipped in this build.

Landmark indices follow MediaPipe's hand model (0 = wrist, 4 = thumb tip,
8 = index tip, 12 = middle tip, 16 = ring tip, 20 = pinky tip).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from . import config
from .models import ensure_hand_model
from .smoothing import LandmarkSmoother

# Standard 21-point hand topology (pairs of landmark indices), for drawing the
# skeleton. Hardcoded because the Tasks API doesn't expose the old
# solutions.HAND_CONNECTIONS constant.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


@dataclass
class Hand:
    """One detected hand, decoupled from MediaPipe's types."""
    label: str                       # 'Left' / 'Right' (see mirroring note below)
    score: float                     # handedness confidence 0..1
    landmarks_px: list[tuple[int, int]]               # 21 (x, y) pixel coords — for drawing/placement
    landmarks_norm: list[tuple[float, float, float]]  # 21 (x, y, z) image-normalized 0..1
    landmarks_world: list[tuple[float, float, float]] # 21 (x, y, z) TRUE 3D (meters) — for recognition

    def point(self, i: int) -> tuple[int, int]:
        """Pixel coordinate of landmark `i` (convenience for gesture code)."""
        return self.landmarks_px[i]


class HandTracker:
    """Stateful tracker — create once, call `process()` per frame, `close()` at the end.

    NOTE on mirroring: we feed it the same (mirrored) frame we display, so the
    drawn landmarks line up with what you see. MediaPipe then reports Left/Right
    backwards (it's looking at a mirror image), so when `mirrored=True` we swap the
    labels back to the hand you're actually holding up.
    """

    def __init__(
        self,
        max_hands: int = config.MAX_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        mirrored: bool = config.MIRROR,
    ) -> None:
        self._mirrored = mirrored
        model_path = ensure_hand_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,     # track across frames (smoother, faster than per-image)
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1   # VIDEO mode requires strictly increasing timestamps
        self._smoother = (
            LandmarkSmoother(config.SMOOTH_MIN_CUTOFF, config.SMOOTH_BETA)
            if config.SMOOTHING_ENABLED else None
        )

    def process(self, frame_bgr) -> list[Hand]:
        """Detect hands in a BGR frame. Returns [] when none are found."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.perf_counter() * 1000)
        if ts_ms <= self._last_ts_ms:           # guarantee monotonic increase
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        hands: list[Hand] = []
        if not result.hand_landmarks:
            return hands

        world = result.hand_world_landmarks or []
        for i, lms in enumerate(result.hand_landmarks):
            label, score = "Unknown", 0.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
                if self._mirrored:   # undo the mirror so the label matches the real hand
                    label = {"Left": "Right", "Right": "Left"}.get(label, label)

            norm = [(p.x, p.y, p.z) for p in lms]
            wl = world[i] if i < len(world) else lms   # true 3D (meters) for recognition
            world_pts = [(p.x, p.y, p.z) for p in wl]

            if self._smoother is not None:
                t = ts_ms / 1000.0
                norm = self._smoother.smooth(("n", i), norm, t)
                world_pts = self._smoother.smooth(("w", i), world_pts, t)

            # derive pixels from the (smoothed) normalized landmarks
            px = [(int(x * w), int(y * h)) for (x, y, _z) in norm]
            hands.append(Hand(label=label, score=score, landmarks_px=px,
                              landmarks_norm=norm, landmarks_world=world_pts))

        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
