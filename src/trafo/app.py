"""Application entry: wires the controller, tray, main window and onboarding."""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from .config import Settings
from .ui import theme
from .ui.controller import TrafoController
from .ui.main_window import MainWindow
from .ui.tray import Tray


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
    controller.start()

    settings = Settings.load()
    if auto_calibrate:
        _show(window)
        win = controller.begin_calibration()
        win.start()
    elif not settings.onboarded:  # first launch
        from .ui.onboarding import OnboardingWindow

        onboarding = OnboardingWindow()

        def _after_onboarding():
            settings.onboarded = True
            settings.save()
            _show(window)
            win = controller.begin_calibration()
            win.start()

        onboarding.done.connect(_after_onboarding)
        onboarding.show()
        app._onboarding = onboarding  # keep a reference alive
    else:
        _show(window)
        if overlay_on and controller.is_calibrated:
            controller.set_overlay(True)

    return app.exec()


def _show(window) -> None:
    window.show()
    window.raise_()
    window.activateWindow()
