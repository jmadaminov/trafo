"""One Euro filter (Casiez et al. 2012) for the predicted gaze point.

Adapts smoothing strength to speed: heavy smoothing when the gaze is steady
(kills jitter for dwell detection), light smoothing during saccades (low lag).
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.015, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None or t <= self._t:
            self._x = x
            self._dx = np.zeros_like(x)
            self._t = t
            return x

        dt = t - self._t
        self._t = t

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x
