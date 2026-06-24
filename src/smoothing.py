"""Temporal smoothing for noisy landmarks — the One Euro filter.

When the hand turns edge-on and fingers occlude each other, MediaPipe's landmark
estimates get jittery (the model is genuinely uncertain). A One Euro filter damps
that jitter when things are slow/uncertain but adapts during fast motion so it
doesn't lag. Reference: Casiez, Roussel & Vogel (2012).

Usage: keep one LandmarkSmoother per tracker; call `smooth(key, points, t)` each
frame, where `key` distinguishes streams (e.g. ('norm', hand_index)).
"""

from __future__ import annotations

import math


class OneEuroFilter:
    """One Euro filter for a single scalar signal."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None:
            self._x_prev, self._t_prev = x, t
            return x
        dt = max(1e-6, t - self._t_prev)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)   # adapt: faster motion -> higher cutoff
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


class LandmarkSmoother:
    """Smooths streams of (x, y, z) landmark lists, one One Euro filter per coord."""

    def __init__(self, min_cutoff: float, beta: float) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._filters: dict = {}

    def _filter(self, key) -> OneEuroFilter:
        f = self._filters.get(key)
        if f is None:
            f = OneEuroFilter(self._min_cutoff, self._beta)
            self._filters[key] = f
        return f

    def smooth(self, key, points, t: float):
        out = []
        for i, (x, y, z) in enumerate(points):
            fx = self._filter((key, i, 0))
            fy = self._filter((key, i, 1))
            fz = self._filter((key, i, 2))
            out.append((fx(x, t), fy(y, t), fz(z, t)))
        return out
