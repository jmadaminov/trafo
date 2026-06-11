"""Continuous recalibration from mouse clicks.

People look at what they click. Each accepted click therefore pairs the
current gaze features with a known screen position — free ground truth.
The gate filters out the cases where that assumption is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GazeHistoryEntry:
    t: float
    features: np.ndarray
    raw_pred: np.ndarray  # un-biased model prediction (global px)


@dataclass
class ClickSampleGate:
    """Decides whether a click may be used as a training sample.

    Accept only when:
    - enough recent gaze history exists (face visible, not blinking),
    - the gaze was stable just before the click (fixating, not scanning),
    - the prediction already lands near the click (clicking from muscle
      memory while looking elsewhere must NOT poison the model),
    - a cooldown has passed since the last accepted click.
    """

    window_s: float = 0.4
    min_samples: int = 5
    stability_px: float = 150.0
    max_distance_px: float = 400.0
    cooldown_s: float = 1.0
    _last_accept_t: float = field(default=-1e9, init=False)

    def evaluate(
        self,
        click_xy: tuple[float, float],
        t: float,
        history: list[GazeHistoryEntry],
    ) -> np.ndarray | None:
        """Returns the feature vector to train on, or None to reject."""
        if t - self._last_accept_t < self.cooldown_s:
            return None
        recent = [h for h in history if t - h.t <= self.window_s]
        if len(recent) < self.min_samples:
            return None

        preds = np.stack([h.raw_pred for h in recent])
        center = np.median(preds, axis=0)
        if np.max(np.linalg.norm(preds - center, axis=1)) > self.stability_px:
            return None  # eyes were moving; the user wasn't fixating the target
        if np.linalg.norm(center - np.asarray(click_xy, dtype=float)) > self.max_distance_px:
            return None  # looking somewhere else while clicking

        self._last_accept_t = t
        return np.median(np.stack([h.features for h in recent]), axis=0)


class ClickListener:
    """Global mouse-press listener (pynput); fires callback(x, y) from its own thread."""

    def __init__(self, callback):
        self._callback = callback
        self._listener = None

    def start(self) -> bool:
        try:
            from pynput import mouse

            def on_click(x, y, button, pressed):
                if pressed:
                    self._callback(float(x), float(y))

            self._listener = mouse.Listener(on_click=on_click)
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
