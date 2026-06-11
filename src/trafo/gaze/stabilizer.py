"""Output stabilization: spike rejection, adaptive smoothing, fixation freeze.

Three stages, addressing distinct artifacts:
1. Component-wise median over a short window kills single-frame spikes
   (residual blink corruption, landmark glitches) outright.
2. A One Euro filter smooths what remains with low lag.
3. Fixation detection: when the recent dispersion is small the user is
   fixating — the output snaps to the running median and stays put, instead
   of wandering with the noise. A real saccade exceeds the dispersion radius
   immediately and passes through unhindered.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .smoothing import OneEuroFilter


class GazeStabilizer:
    def __init__(
        self,
        spike_window: int = 5,
        fixation_window: int = 10,
        fixation_radius_px: float = 60.0,
        min_cutoff: float = 0.3,
        beta: float = 0.008,
    ):
        self.fixation_radius_px = fixation_radius_px
        self._spike_buf: deque[np.ndarray] = deque(maxlen=spike_window)
        self._fix_buf: deque[np.ndarray] = deque(maxlen=fixation_window)
        self._euro = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._frozen: np.ndarray | None = None

    @property
    def is_fixating(self) -> bool:
        return self._frozen is not None

    def update(self, point: np.ndarray, t: float) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        self._spike_buf.append(point)
        despiked = np.median(np.stack(self._spike_buf), axis=0)
        smoothed = self._euro.filter(despiked, t)
        self._fix_buf.append(smoothed)

        if len(self._fix_buf) == self._fix_buf.maxlen:
            pts = np.stack(self._fix_buf)
            center = np.median(pts, axis=0)
            if np.max(np.linalg.norm(pts - center, axis=1)) <= self.fixation_radius_px:
                # Fixating: freeze. Keep the original anchor while the eye
                # stays inside the radius so micro-drift doesn't creep.
                if (
                    self._frozen is None
                    or np.linalg.norm(center - self._frozen) > self.fixation_radius_px
                ):
                    self._frozen = center
                return self._frozen.copy()

        self._frozen = None
        return smoothed

    def reset(self) -> None:
        self._spike_buf.clear()
        self._fix_buf.clear()
        self._euro.reset()
        self._frozen = None
