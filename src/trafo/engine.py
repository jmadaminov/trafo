"""Focus-follows-gaze engine: dwell timer + hysteresis over window hit-tests.

Pure logic with injected timestamps — no Qt, no clocks — so the focus rules
are unit-testable with synthetic gaze sequences.

Rules:
- A window must be gazed at for `dwell_s` before it is focused.
- Brief flickers away (gaze noise, saccades across a boundary) shorter than
  `flicker_tolerance_s` do not reset the dwell.
- The window with OS input focus is never re-focused. Focus is asked from the
  OS — the topmost window in z-order is not necessarily the focused one.
- Spatial hysteresis for small targets (`edge_margin_px`):
  - a near-miss just outside a window that sits *above* the plain hit in
    z-order resolves to that higher window (small windows on top of large
    ones win against calibration error);
  - a hit landing just outside the focused window onto something *behind* it
    is ignored (gaze spill while reading must not raise the window behind).
- Own windows (the Trafo UI itself) block hits but are never focus targets.
- After a focus switch, `cooldown_s` must pass before the next one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .winmgr.base import WindowInfo, WindowManager


@dataclass
class EngineConfig:
    dwell_s: float = 0.5
    flicker_tolerance_s: float = 0.15
    cooldown_s: float = 1.0
    list_refresh_s: float = 0.25  # how often the window list is re-fetched
    edge_margin_px: float = 75.0  # spatial hysteresis scale (~typical gaze error)


class FocusEngine:
    def __init__(self, wm: WindowManager, config: EngineConfig | None = None):
        self.wm = wm
        self.cfg = config or EngineConfig()
        self._windows: list[WindowInfo] = []
        self._focused: WindowInfo | None = None
        self._last_refresh = -math.inf
        self._pending: WindowInfo | None = None
        self._pending_since = 0.0
        self._mismatch_since: float | None = None
        self._cooldown_until = -math.inf

    @property
    def pending(self) -> WindowInfo | None:
        """The window currently accumulating dwell, if any."""
        return self._pending

    def update(self, x: float, y: float, t: float) -> WindowInfo | None:
        """Feed one gaze point; returns the window if this tick switched focus."""
        if t < self._cooldown_until:
            return None
        if t - self._last_refresh >= self.cfg.list_refresh_s:
            self._windows = self.wm.list_windows()
            self._focused = self.wm.focused_window()
            self._last_refresh = t

        hit = self._resolve_hit(x, y)

        if hit is None or (self._pending is not None and hit.id != self._pending.id):
            # Gaze is off the pending window: tolerate brief flickers, then
            # either drop the dwell or restart it on the new window.
            if self._pending is None:
                if hit is not None:
                    self._start_dwell(hit, t)
                return None
            if self._mismatch_since is None:
                self._mismatch_since = t
            elif t - self._mismatch_since > self.cfg.flicker_tolerance_s:
                if hit is not None:
                    self._start_dwell(hit, t)
                else:
                    self._pending = None
            return None

        if self._pending is None:
            self._start_dwell(hit, t)
            return None

        # Gaze is (still) on the pending window.
        self._mismatch_since = None
        if t - self._pending_since < self.cfg.dwell_s:
            return None

        target, self._pending = self._pending, None
        self._cooldown_until = t + self.cfg.cooldown_s
        self._last_refresh = -math.inf  # frontmost changed; refresh next tick
        return target if self.wm.focus(target) else None

    def _start_dwell(self, window: WindowInfo, t: float) -> None:
        self._pending = window
        self._pending_since = t
        self._mismatch_since = None

    def _resolve_hit(self, x: float, y: float) -> WindowInfo | None:
        """Window the gaze should count toward, applying z-preference and
        focused-window hysteresis. None = nothing to dwell on."""
        margin = self.cfg.edge_margin_px
        plain_i = next((i for i, w in enumerate(self._windows) if w.contains(x, y)), None)
        if plain_i is None:
            return None

        # Near-miss preference: a window above the plain hit whose expanded
        # rect contains the point is what the user is actually looking at.
        hit_i, hit = plain_i, self._windows[plain_i]
        for i in range(plain_i):
            if self._windows[i].contains(x, y, margin):
                hit_i, hit = i, self._windows[i]
                break

        if hit.own:
            return None  # our own UI: blocks the gaze, never a target
        if self._focused is not None:
            if hit.id == self._focused.id:
                return None  # already has focus
            focused_i = next(
                (i for i, w in enumerate(self._windows) if w.id == self._focused.id), None
            )
            if (
                focused_i is not None
                and hit_i > focused_i
                and self._focused.contains(x, y, margin)
            ):
                # Gaze spilled just outside the focused window onto a window
                # behind it — reading near an edge must not raise the back one.
                return None
        return hit
