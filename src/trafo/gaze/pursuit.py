"""Smooth-pursuit calibration: moving-dot path + pursuit-lag compensation.

Instead of staring at a handful of static dots, the user follows one dot
sweeping the whole screen, yielding ~1000 densely-distributed training
samples per display. Two physiological corrections make that data usable:

- Pursuit lag: the eye trails a moving target by ~100-150 ms, so pairing a
  frame's features with where the dot is *now* mislabels every sample by
  lag x dot speed. The lag is estimated per screen from the data itself and
  targets are time-shifted to where the dot *was*.
- Catch-up saccades (the eye jumping ahead after falling behind) are removed
  downstream by the residual trimming in ScreenLockedMapper.fit.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .calibration import GazeMapper

PURSUIT_DURATION_S = 35.0  # per screen; ~1000 samples at 30 fps

_F1, _F2 = 0.10, 0.143  # Hz; ~7:10 ratio sweeps the rect without repeating
_AMPLITUDE = 0.42  # fraction of width/height — same 8%-92% margins as the old grid


def pursuit_path(
    rect: tuple[float, float, float, float],
    duration_s: float = PURSUIT_DURATION_S,
) -> Callable[[float], np.ndarray]:
    """target_fn(t): elapsed seconds -> global (x, y) along a Lissajous sweep.

    Sinusoidal velocity slows toward the edges and corners — exactly where
    smooth pursuit is hardest. t is clamped to [0, duration_s], so querying
    slightly out of range (lag shifting) pins to the path's endpoints.
    """
    x0, y0, w, h = rect
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    ax, ay = _AMPLITUDE * w, _AMPLITUDE * h

    def target_fn(t: float) -> np.ndarray:
        t = min(max(t, 0.0), duration_s)
        x = cx + ax * np.sin(2 * np.pi * _F1 * t + np.pi / 2)
        y = cy + ay * np.sin(2 * np.pi * _F2 * t)
        return np.array([x, y])

    return target_fn


def pair_with_lag(
    times: np.ndarray,
    features: list[np.ndarray],
    target_fn: Callable[[float], np.ndarray],
    lags: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Pair pursuit samples with lag-shifted targets.

    Grid-searches the pursuit lag (0-250 ms): for each candidate, features are
    paired with target_fn(t - lag) and scored by a throwaway GazeMapper fit
    (milliseconds each). Returns (lag_s, targets) for the best candidate —
    the labels the regression explains best are the ones the eye actually
    looked at.
    """
    if lags is None:
        lags = np.arange(0.0, 0.251, 0.025)
    times = np.asarray(times, dtype=float)
    x = np.stack(features)
    best: tuple[float, float, np.ndarray] | None = None
    for lag in lags:
        targets = np.stack([target_fn(t - lag) for t in times])
        rms = GazeMapper().fit(x, targets)
        if best is None or rms < best[0]:
            best = (rms, float(lag), targets)
    return best[1], best[2]
