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
import time

import ApplicationServices as AS
import Quartz
from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication, NSWorkspace

from .base import WindowInfo, WindowManager

_MIN_SIZE = 40  # skip tooltip/utility scraps

# System chrome that reports full-display windows (the Dock's window spans the
# whole screen) — never gaze blockers, never targets.
_SYSTEM_OWNERS = {
    "Dock", "Window Server", "Notification Center", "Control Center",
    "Spotlight", "Screenshot",
}
_FULLSCREEN_COVERAGE = 0.9  # fraction of a display an overlay must cover


def _display_rects() -> list[tuple[float, float, float, float]]:
    err, ids, _ = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != 0:
        return []
    out = []
    for d in ids or []:
        r = Quartz.CGDisplayBounds(d)
        out.append((r.origin.x, r.origin.y, r.size.width, r.size.height))
    return out


def _covers_a_display(rect, displays) -> bool:
    """True when rect covers ≥ _FULLSCREEN_COVERAGE of any display."""
    x, y, w, h = rect
    for dx, dy, dw, dh in displays:
        ix = max(0.0, min(x + w, dx + dw) - max(x, dx))
        iy = max(0.0, min(y + h, dy + dh) - max(y, dy))
        if dw * dh > 0 and ix * iy >= _FULLSCREEN_COVERAGE * dw * dh:
            return True
    return False


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
        displays = _display_rects()
        out: list[WindowInfo] = []
        for w in infos or []:
            pid = w.get(Quartz.kCGWindowOwnerPID)
            if pid is None:
                continue
            if w.get(Quartz.kCGWindowAlpha, 1) == 0:
                continue
            b = w.get(Quartz.kCGWindowBounds) or {}
            rect = (b.get("X", 0), b.get("Y", 0), b.get("Width", 0), b.get("Height", 0))
            if rect[2] < _MIN_SIZE or rect[3] < _MIN_SIZE:
                continue
            app = str(w.get(Quartz.kCGWindowOwnerName) or "")
            layer = w.get(Quartz.kCGWindowLayer, 1)
            blocker = False
            if layer != 0:  # 0 = normal app windows
                # Fullscreen overlays (video players etc.) sit above the
                # normal layer; without them in the list, gaze falls through
                # to the windows underneath and steals focus mid-video. Keep
                # them as untargetable blockers; skip everything else.
                if app in _SYSTEM_OWNERS:
                    continue
                if not (layer > 0 and _covers_a_display(rect, displays)):
                    continue
                blocker = True
            out.append(
                WindowInfo(
                    id=int(w.get(Quartz.kCGWindowNumber, 0)),
                    title=str(w.get(Quartz.kCGWindowName) or ""),
                    app=app,
                    pid=int(pid),
                    rect=rect,
                    # `own` means "blocks gaze, never a focus target" — true
                    # for Trafo's windows and for fullscreen overlays.
                    own=int(pid) == own_pid or blocker,
                )
            )
        return out  # CGWindowList is already front-to-back

    def focused_window(self) -> WindowInfo | None:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return super().focused_window()
        pid = app.processIdentifier()
        ours = [w for w in self.list_windows() if w.pid == pid]
        if not ours:
            return None
        if len(ours) == 1:
            return ours[0]
        # Multi-window app: the first CG window of the pid is not necessarily
        # the focused one — ask AX which window actually has focus and match
        # it back to the CG list by rect.
        ax_focused = _ax_attr(
            AS.AXUIElementCreateApplication(pid), AS.kAXFocusedWindowAttribute
        )
        rect = _ax_rect(ax_focused) if ax_focused is not None else None
        if rect is None:
            return ours[0]
        return min(
            ours, key=lambda w: sum(abs(a - b) for a, b in zip(rect, w.rect))
        )

    def focus(self, window: WindowInfo) -> bool:
        # The AX API has no direct CGWindowID lookup (that mapping is private
        # API), so find the app's AX window that best matches title + rect.
        app_ref = AS.AXUIElementCreateApplication(window.pid)
        ax_windows = _ax_attr(app_ref, AS.kAXWindowsAttribute)
        if not ax_windows:
            return False
        best = self._match_ax_window(ax_windows, window)
        if best is None:
            return False

        # Order matters for multi-window apps: activating an app brings its
        # *main* window forward, so the target must already be main before
        # activation — otherwise the app's previous window (possibly on
        # another display) is the one that gets raised.
        AS.AXUIElementSetAttributeValue(best, AS.kAXMainAttribute, True)
        AS.AXUIElementSetAttributeValue(app_ref, AS.kAXFocusedWindowAttribute, best)
        if AS.AXUIElementPerformAction(best, AS.kAXRaiseAction) != 0:
            return False

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(window.pid)
        if app is not None and not app.isActive():
            # Real activation is asynchronous and ends by bringing the app's
            # previous key window forward — which overrides the raise above
            # for multi-window apps. Wait for the activation to finish, then
            # re-assert the target so it has the last word.
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            deadline = time.time() + 0.3
            while not app.isActive() and time.time() < deadline:
                time.sleep(0.01)
            time.sleep(0.03)  # let the window-server reorder settle
            AS.AXUIElementSetAttributeValue(best, AS.kAXMainAttribute, True)
            AS.AXUIElementSetAttributeValue(app_ref, AS.kAXFocusedWindowAttribute, best)
            AS.AXUIElementPerformAction(best, AS.kAXRaiseAction)
        return True

    @staticmethod
    def _match_ax_window(ax_windows, window: WindowInfo):
        """The app's AX window best matching the CG window's rect (and title)."""
        best, best_score = None, float("inf")
        for axwin in ax_windows:
            if _ax_attr(axwin, AS.kAXMinimizedAttribute):
                continue  # CG hit is on-screen; a minimized window can't be it
            rect = _ax_rect(axwin)
            score = (
                sum(abs(a - b) for a, b in zip(rect, window.rect))
                if rect is not None else 10_000.0
            )
            title = _ax_attr(axwin, AS.kAXTitleAttribute)
            # Tie-break by title only when both sides actually have one
            # (CG names are empty without Screen Recording permission).
            if window.title and title and title != window.title:
                score += 500
            if score < best_score:
                best, best_score = axwin, score
        return best

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
