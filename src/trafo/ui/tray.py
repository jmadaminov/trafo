"""System-tray / menu-bar presence so Trafo can run in the background."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import theme
from .controller import TrafoController


class Tray(QSystemTrayIcon):
    def __init__(self, controller: TrafoController, show_window):
        super().__init__(theme.make_icon())
        self.c = controller
        self._show_window = show_window
        self.setToolTip("Trafo")

        menu = QMenu()
        self._open_action = menu.addAction("Open Trafo")
        self._open_action.triggered.connect(self._open)
        menu.addSeparator()

        self._overlay_action = QAction("Show gaze dot", menu, checkable=True)
        self._overlay_action.triggered.connect(self.c.set_overlay)
        menu.addAction(self._overlay_action)

        self._engine_action = QAction("Focus follows gaze", menu, checkable=True)
        self._engine_action.triggered.connect(self.c.set_engine)
        menu.addAction(self._engine_action)

        menu.addSeparator()
        self._calibrate_action = menu.addAction("Recalibrate…")
        self._calibrate_action.triggered.connect(self._calibrate)

        menu.addSeparator()
        quit_action = menu.addAction("Quit Trafo")
        quit_action.triggered.connect(QApplication.instance().quit)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

        # Keep menu checkmarks in sync with controller state.
        self.c.overlay_changed.connect(self._overlay_action.setChecked)
        self.c.engine_changed.connect(self._engine_action.setChecked)
        self.c.calibration_changed.connect(self._sync_enabled)
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        calibrated = self.c.is_calibrated
        for a in (self._overlay_action, self._engine_action):
            a.setEnabled(calibrated)

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._open()

    def _open(self) -> None:
        self._show_window()

    def _calibrate(self) -> None:
        self._show_window()
        window = self.c.begin_calibration()
        window.start()
