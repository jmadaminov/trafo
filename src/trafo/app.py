"""Application entry: wires the controller, tray, main window and onboarding."""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from .config import Settings
from .permissions import bundle_path, ensure_camera_access, open_settings_pane
from .ui import theme
from .ui.controller import TrafoController
from .ui.main_window import MainWindow
from .ui.tray import Tray


def _maybe_move_to_applications() -> bool:
    """Offer to install the bundle into /Applications on first launch.

    Returns True when the app relocated itself (a fresh copy was launched and
    this process should exit). Standard macOS app etiquette — and permission
    grants are easier to reason about when the app has a stable home.
    """
    src = bundle_path()
    if src is None or src.startswith("/Applications/"):
        return False

    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox()
    box.setWindowTitle("Move to Applications?")
    box.setText("Move Trafo to your Applications folder?")
    box.setInformativeText(
        "Trafo works best from /Applications: macOS remembers permission "
        "grants more reliably for installed apps."
    )
    move = box.addButton("Move to Applications", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not move:
        return False

    import shutil
    import subprocess

    dest = "/Applications/Trafo.app"
    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        subprocess.run(["ditto", src, dest], check=True)
        subprocess.Popen(["open", "-n", dest])
        return True
    except Exception:
        return False  # no permission to write /Applications etc. — keep running


def _ensure_camera_or_warn() -> None:
    """Resolve the camera permission before the capture thread opens the device.

    Onboarding handles the first-launch grant, but already-onboarded installs
    (and dev runs from a terminal that was never prompted) reach the camera
    with the permission still NotDetermined — and OpenCV is configured to not
    ask. Prompt here; if denied, point at the settings pane.
    """
    if ensure_camera_access():
        return
    from PySide6.QtWidgets import QMessageBox

    QMessageBox.warning(
        None,
        "Camera access needed",
        "Trafo needs camera access to track your gaze.\n\n"
        "Enable it under System Settings ▸ Privacy & Security ▸ Camera "
        "for this app (or your terminal in dev mode), then restart Trafo.",
    )
    open_settings_pane("Privacy_Camera")


def run_app(camera_index: int = 0, auto_calibrate: bool = False, overlay_on: bool = False) -> int:
    # The camera opens on a worker thread; macOS can only show the camera
    # permission prompt from the main loop. Skip the in-thread auth request so
    # a missing grant surfaces as a clear message instead of a hang.
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Trafo")
    app.setWindowIcon(theme.make_icon())
    theme.apply_theme(app)
    # Background app: closing the window hides to tray, doesn't quit.
    app.setQuitOnLastWindowClosed(False)

    if _maybe_move_to_applications():
        return 0  # relocated; the /Applications copy is taking over

    controller = TrafoController(camera_index)
    window = MainWindow(controller)

    have_tray = QSystemTrayIcon.isSystemTrayAvailable()
    tray = None
    if have_tray:
        tray = Tray(controller, show_window=lambda: _show(window))
        tray.show()
        controller.notice.connect(
            lambda m: tray.showMessage("Trafo", m, theme.make_icon(), 4000)
        )
    else:
        # No tray (e.g. some Linux sessions): keep the app alive via the window.
        app.setQuitOnLastWindowClosed(True)
        window.hide_on_close = False

    app.aboutToQuit.connect(controller.shutdown)

    settings = Settings.load()
    if auto_calibrate:
        _ensure_camera_or_warn()
        controller.start()
        _show(window)
        win = controller.begin_calibration()
        win.start()
    elif not settings.onboarded:  # first launch
        from .ui.onboarding import OnboardingWindow

        onboarding = OnboardingWindow()

        def _after_onboarding():
            # Start the camera only now: onboarding has granted the camera
            # permission, so opening the device succeeds on the first try.
            settings.onboarded = True
            settings.save()
            controller.start()
            _show(window)
            win = controller.begin_calibration()
            win.start()

        onboarding.done.connect(_after_onboarding)
        onboarding.show()
        app._onboarding = onboarding  # keep a reference alive
    else:
        _ensure_camera_or_warn()
        controller.start()
        _show(window)
        if overlay_on and controller.is_calibrated:
            controller.set_overlay(True)

    return app.exec()


def _show(window) -> None:
    window.show()
    window.raise_()
    window.activateWindow()
