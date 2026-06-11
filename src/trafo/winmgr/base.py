"""WindowManager interface; platform backends implement it.

All rectangles are in global virtual-desktop coordinates with a top-left
origin and y growing downward — the same convention Qt uses, so gaze points
and window rects can be compared directly.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowInfo:
    id: int  # platform-specific window id
    title: str
    app: str
    pid: int
    rect: tuple[float, float, float, float]  # x, y, width, height (global coords)
    own: bool = False  # belongs to this process (never a focus target, but blocks hits)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        rx, ry, rw, rh = self.rect
        return rx - margin <= x < rx + rw + margin and ry - margin <= y < ry + rh + margin


class WindowManager(ABC):
    @abstractmethod
    def list_windows(self) -> list[WindowInfo]:
        """Visible, normal application windows, front-to-back (z-order).

        Includes this process's own normal windows flagged `own=True` so
        hit-testing can treat them as blockers rather than see through them.
        """

    @abstractmethod
    def focus(self, window: WindowInfo) -> bool:
        """Raise the window and give it focus. Returns True on success."""

    def focused_window(self) -> WindowInfo | None:
        """The window that currently has input focus.

        Note: NOT necessarily the topmost window — floating/utility windows of
        other apps can sit above it in z-order. Platform backends should ask
        the OS; this fallback approximates it with the topmost window.
        """
        for w in self.list_windows():
            if not w.own:
                return w
        return None

    def window_at(self, x: float, y: float) -> WindowInfo | None:
        """Topmost window containing the point (z-order = list order)."""
        for w in self.list_windows():
            if w.contains(x, y):
                return w
        return None

    def frontmost(self) -> WindowInfo | None:
        windows = self.list_windows()
        return windows[0] if windows else None

    def permissions_missing(self) -> list[str]:
        """Names of OS permissions still needed for full functionality."""
        return []


def get_window_manager() -> WindowManager:
    if sys.platform == "darwin":
        from .macos import MacWindowManager

        return MacWindowManager()
    if sys.platform == "win32":
        raise NotImplementedError("Windows backend arrives in milestone 7")
    raise NotImplementedError("Linux/X11 backend arrives in milestone 7")
