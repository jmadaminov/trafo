"""macOS window manager: Quartz for enumeration, the AX API for focusing.

Permissions:
- Screen Recording — required for other apps' window *titles* (rects and
  owners come through without it).
- Accessibility — required to raise/focus other apps' windows.

Quartz global coordinates are top-left-origin points, identical to Qt's
logical coordinates, so no conversion is needed.
"""

from __future__ import annotations

import os

import ApplicationServices as AS
import Quartz
from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication, NSWorkspace

from .base import WindowInfo, WindowManager

_MIN_SIZE = 40  # skip tooltip/utility scraps


def _ax_attr(element, name):
    err, value = AS.AXUIElementCopyAttributeValue(element, name, None)
    return value if err == 0 else None


def _ax_rect(axwin) -> tuple[float, float, float, float] | None:
    pos_v = _ax_attr(axwin, AS.kAXPositionAttribute)
    size_v = _ax_attr(axwin, AS.kAXSizeAttribute)
    if pos_v is None or size_v is None:
        return None
    ok_p, pos = AS.AXValueGetValue(pos_v, AS.kAXValueCGPointType, None)
    ok_s, size = AS.AXValueGetValue(size_v, AS.kAXValueCGSizeType, None)
    if not (ok_p and ok_s):
        return None
    return (pos.x, pos.y, size.width, size.height)


class MacWindowManager(WindowManager):
    def list_windows(self) -> list[WindowInfo]:
        infos = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        own_pid = os.getpid()
        out: list[WindowInfo] = []
        for w in infos or []:
            if w.get(Quartz.kCGWindowLayer, 1) != 0:  # 0 = normal app windows
                continue
            pid = w.get(Quartz.kCGWindowOwnerPID)
            if pid is None:
                continue
            if w.get(Quartz.kCGWindowAlpha, 1) == 0:
                continue
            b = w.get(Quartz.kCGWindowBounds) or {}
            rect = (b.get("X", 0), b.get("Y", 0), b.get("Width", 0), b.get("Height", 0))
            if rect[2] < _MIN_SIZE or rect[3] < _MIN_SIZE:
                continue
            out.append(
                WindowInfo(
                    id=int(w.get(Quartz.kCGWindowNumber, 0)),
                    title=str(w.get(Quartz.kCGWindowName) or ""),
                    app=str(w.get(Quartz.kCGWindowOwnerName) or ""),
                    pid=int(pid),
                    rect=rect,
                    own=int(pid) == own_pid,
                )
            )
        return out  # CGWindowList is already front-to-back

    def focused_window(self) -> WindowInfo | None:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return super().focused_window()
        pid = app.processIdentifier()
        for w in self.list_windows():
            if w.pid == pid:
                return w
        return None

    def focus(self, window: WindowInfo) -> bool:
        # The AX API has no direct CGWindowID lookup (that mapping is private
        # API), so find the app's AX window that best matches title + rect.
        app_ref = AS.AXUIElementCreateApplication(window.pid)
        ax_windows = _ax_attr(app_ref, AS.kAXWindowsAttribute)
        if not ax_windows:
            return False

        best, best_score = None, float("inf")
        for axwin in ax_windows:
            score = 0.0
            rect = _ax_rect(axwin)
            if rect is not None:
                score += sum(abs(a - b) for a, b in zip(rect, window.rect))
            else:
                score += 10_000
            title = _ax_attr(axwin, AS.kAXTitleAttribute)
            if window.title and title != window.title:
                score += 500
            if score < best_score:
                best, best_score = axwin, score

        if best is None:
            return False
        if AS.AXUIElementPerformAction(best, AS.kAXRaiseAction) != 0:
            return False
        AS.AXUIElementSetAttributeValue(best, AS.kAXMainAttribute, True)

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(window.pid)
        if app is not None:
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        return True

    def permissions_missing(self) -> list[str]:
        missing = []
        if not AS.AXIsProcessTrusted():
            missing.append("Accessibility (System Settings > Privacy & Security > Accessibility)")
        if not Quartz.CGPreflightScreenCaptureAccess():
            missing.append(
                "Screen Recording (System Settings > Privacy & Security > Screen Recording)"
                " — window titles will be empty without it"
            )
        return missing

    def request_permissions(self) -> None:
        """Trigger the OS permission prompts (once each)."""
        AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
        Quartz.CGRequestScreenCaptureAccess()
