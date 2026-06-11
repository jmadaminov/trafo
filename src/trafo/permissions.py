"""Cross-platform permission probes used by the onboarding flow.

Each probe returns one of: "ok", "missing", "unknown" (can't tell without
triggering work), or "n/a" (not applicable on this OS). Probes never raise.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class Permission:
    key: str
    name: str
    why: str
    settings_hint: str


MAC_PERMISSIONS = [
    Permission("camera", "Camera", "See your eyes to track gaze.",
               "Privacy & Security ▸ Camera"),
    Permission("screen", "Screen Recording", "Read other apps' window titles and positions.",
               "Privacy & Security ▸ Screen Recording"),
    Permission("accessibility", "Accessibility", "Bring the window you look at to the front.",
               "Privacy & Security ▸ Accessibility"),
    Permission("input", "Input Monitoring", "Learn from your clicks to stay accurate.",
               "Privacy & Security ▸ Input Monitoring"),
]

# System Settings ▸ Privacy & Security anchors, keyed by permission.
_SETTINGS_ANCHOR = {
    "camera": "Privacy_Camera",
    "screen": "Privacy_ScreenCapture",
    "accessibility": "Privacy_Accessibility",
    "input": "Privacy_ListenEvent",
}


def permissions_for_platform() -> list[Permission]:
    if sys.platform == "darwin":
        return MAC_PERMISSIONS
    return []


def check(key: str) -> str:
    if sys.platform != "darwin":
        return "n/a"
    try:
        if key == "camera":
            from AVFoundation import (
                AVAuthorizationStatusAuthorized,
                AVCaptureDevice,
                AVMediaTypeVideo,
            )

            status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
            return "ok" if status == AVAuthorizationStatusAuthorized else "missing"
        if key == "screen":
            import Quartz

            return "ok" if Quartz.CGPreflightScreenCaptureAccess() else "missing"
        if key == "accessibility":
            import ApplicationServices as AS

            return "ok" if AS.AXIsProcessTrusted() else "missing"
        if key == "input":
            # No silent probe exists for Input Monitoring; reported as unknown
            # so the UI shows a neutral state with a "grant" affordance.
            return "unknown"
    except Exception:
        return "unknown"
    return "unknown"


def request(key: str) -> None:
    """Trigger the OS prompt / open the relevant settings pane.

    Must never block the caller's thread: this runs from a GUI button handler,
    so any synchronous OS call that can stall (notably the screen-capture TCC
    request on ad-hoc-signed builds) is dispatched off-thread or replaced by
    opening System Settings, which is always fast.
    """
    if sys.platform != "darwin":
        return
    try:
        if key == "camera":
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

            # Async by design (completion handler) — never blocks the GUI.
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeVideo, lambda granted: None
            )
        elif key == "screen":
            import threading

            import Quartz

            # CGRequestScreenCaptureAccess() is synchronous and can hang the
            # calling thread for re-signed/unsigned bundles, freezing the
            # window. Fire it off-thread (registers the app + shows the system
            # prompt) and open the pane so the user can toggle it directly.
            threading.Thread(
                target=Quartz.CGRequestScreenCaptureAccess, daemon=True
            ).start()
            open_settings_pane(_SETTINGS_ANCHOR["screen"])
        elif key == "accessibility":
            import ApplicationServices as AS

            AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
            open_settings_pane(_SETTINGS_ANCHOR["accessibility"])
        elif key == "input":
            open_settings_pane(_SETTINGS_ANCHOR["input"])
    except Exception:
        pass


def bundle_path() -> str | None:
    """Path to the .app bundle when running frozen on macOS, else None."""
    exe = sys.executable
    marker = ".app/Contents/MacOS"
    if sys.platform == "darwin" and marker in exe:
        return exe.split(marker)[0] + ".app"
    return None


def relaunch_app() -> bool:
    """Spawn a fresh instance of the .app (caller should then quit).

    macOS applies new TCC grants only to freshly launched processes, so
    "grant, then restart" is the only way for a permission to take effect.
    Returns False when not running from a bundle (dev mode).
    """
    path = bundle_path()
    if path is None:
        return False
    import subprocess

    subprocess.Popen(["open", "-n", path])
    return True


def open_settings_pane(anchor: str = "") -> None:
    """Open System Settings (optionally at a Privacy anchor) on macOS."""
    if sys.platform != "darwin":
        return
    import subprocess

    url = "x-apple.systempreferences:com.apple.preference.security"
    if anchor:
        url += f"?{anchor}"
    subprocess.Popen(["open", url])
