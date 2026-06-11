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
    """Trigger the OS prompt / open the relevant settings pane."""
    if sys.platform != "darwin":
        return
    try:
        if key == "camera":
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeVideo, lambda granted: None
            )
        elif key == "screen":
            import Quartz

            Quartz.CGRequestScreenCaptureAccess()
        elif key == "accessibility":
            import ApplicationServices as AS

            AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
        elif key == "input":
            open_settings_pane("Privacy_ListenEvent")
    except Exception:
        pass


def open_settings_pane(anchor: str = "") -> None:
    """Open System Settings (optionally at a Privacy anchor) on macOS."""
    if sys.platform != "darwin":
        return
    import subprocess

    url = "x-apple.systempreferences:com.apple.preference.security"
    if anchor:
        url += f"?{anchor}"
    subprocess.Popen(["open", url])
