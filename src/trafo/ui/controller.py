"""TrafoController: the headless core the UI observes.

Owns the gaze worker, mapper, focus engine, overlay, stabilizer and click
learning, and exposes everything as Qt signals/slots. The main window, system
tray and debug view are all thin views onto this one object, so they stay in
sync and none of them owns the pipeline.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QGuiApplication

from ..config import Settings
from ..engine import EngineConfig, FocusEngine
from ..gaze.calibration import ScreenLockedMapper
from ..gaze.clicks import ClickListener, ClickSampleGate, GazeHistoryEntry
from ..gaze.estimator import GazeSample
from ..gaze.stabilizer import GazeStabilizer
from ..winmgr import get_window_manager
from .overlay import GazeOverlay
from .worker import GazeWorker


class TrafoController(QObject):
    # Tracking state for the UI.
    tracking_changed = Signal(str)  # "starting" | "tracking" | "blinking" | "no_face" | "error"
    error = Signal(str)
    # Live data.
    gaze_point = Signal(object)  # stabilized np.ndarray (global px), or None while blinking
    sample_ready = Signal(object)  # raw GazeSample (for the debug view)
    focus_switched = Signal(object)  # WindowInfo
    # State the UI reflects.
    calibration_changed = Signal()
    clicks_learned = Signal(int)
    overlay_changed = Signal(bool)
    engine_changed = Signal(bool)
    click_learning_changed = Signal(bool)
    notice = Signal(str)  # transient human-readable message
    _click = Signal(float, float)  # marshals pynput-thread clicks onto this object's thread

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.settings = Settings.load()
        self.mapper = ScreenLockedMapper.load()
        self.overlay = GazeOverlay()

        self._calibrating = False
        self._engine: FocusEngine | None = None
        self._overlay_on = False
        self._last_face_t = 0.0

        self._history: deque[GazeHistoryEntry] = deque(maxlen=20)
        self._bias = np.zeros(2)
        self._stabilizer = GazeStabilizer()
        self._click_gate = ClickSampleGate()
        self._click_listener: ClickListener | None = None

        self.worker = GazeWorker(camera_index, parent=self)
        self.worker.sample.connect(self._on_sample, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._click.connect(self._handle_click, Qt.ConnectionType.QueuedConnection)

    def start(self) -> None:
        self.tracking_changed.emit("starting")
        self.worker.start()
        if self.settings.learn_from_clicks and self.mapper is not None:
            self.set_click_learning(True)

    # -- queries -------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self.mapper is not None

    @property
    def calibrating(self) -> bool:
        return self._calibrating

    def calibration_summary(self) -> str:
        if self.mapper is None:
            return "Not calibrated"
        if self.display_layout_changed():
            return "Display layout changed — recalibrate"
        clicks = self.mapper.click_count
        base = f"Calibrated · {len(self.mapper.rects)} display(s)"
        return f"{base} · {clicks} clicks learned" if clicks else base

    def display_layout_changed(self) -> bool:
        if self.mapper is None:
            return False
        current = sorted(tuple(s.geometry().getRect()) for s in QGuiApplication.screens())
        return sorted(self.mapper.rects) != current

    # -- calibration ---------------------------------------------------------

    def begin_calibration(self) -> "CalibrationWindow":
        from .calibration import CalibrationWindow

        self.set_overlay(False)
        self.set_engine(False)  # no focus stealing mid-calibration
        self._calibrating = True
        window = CalibrationWindow()
        self.worker.sample.connect(window.on_sample, Qt.ConnectionType.QueuedConnection)
        window.finished.connect(lambda m: self._calibration_finished(window, m))
        return window

    def _calibration_finished(self, window, mapper) -> None:
        try:
            self.worker.sample.disconnect(window.on_sample)
        except (RuntimeError, TypeError):
            pass
        self._calibrating = False
        if mapper is not None:
            self.mapper = mapper
            self._bias = np.zeros(2)
            self._history.clear()
            self._stabilizer.reset()
            if self.settings.learn_from_clicks:
                self.set_click_learning(True)
        self.calibration_changed.emit()

    # -- toggles -------------------------------------------------------------

    def set_overlay(self, on: bool) -> None:
        on = on and self.is_calibrated
        self._overlay_on = on
        self.overlay.show() if on else self.overlay.hide()
        self.overlay_changed.emit(on)

    def set_engine(self, on: bool) -> None:
        if not on:
            self._engine = None
            self.engine_changed.emit(False)
            return
        if not self.is_calibrated:
            self.engine_changed.emit(False)
            return
        try:
            wm = get_window_manager()
        except NotImplementedError as exc:
            self.notice.emit(str(exc))
            self.engine_changed.emit(False)
            return
        missing = wm.permissions_missing()
        if missing:
            self.notice.emit("Missing permission: " + "; ".join(missing))
            if hasattr(wm, "request_permissions"):
                wm.request_permissions()
        self._engine = FocusEngine(wm, EngineConfig(dwell_s=self.settings.dwell_ms / 1000))
        self.engine_changed.emit(True)

    def set_click_learning(self, on: bool) -> None:
        self.settings.learn_from_clicks = on
        self.settings.save()
        if on and self._click_listener is None:
            listener = ClickListener(lambda x, y: self._click.emit(x, y))
            if listener.start():
                self._click_listener = listener
            else:
                self.notice.emit("Click learning unavailable (mouse listener failed).")
                self.click_learning_changed.emit(False)
                return
        elif not on and self._click_listener is not None:
            self._click_listener.stop()
            self._click_listener = None
        self.click_learning_changed.emit(on)

    def set_debug(self, on: bool) -> None:
        self.worker.set_debug(on)

    def set_dwell_ms(self, value: int) -> None:
        self.settings.dwell_ms = value
        self.settings.save()
        if self._engine is not None:
            self._engine.cfg.dwell_s = value / 1000

    def recenter(self, target_xy: tuple[float, float]) -> bool:
        """Cancel residual bias using a known gaze target (the user is looking there)."""
        if self.mapper is None or len(self._history) < 5:
            return False
        raw = np.median(np.stack([h.raw_pred for h in self._history]), axis=0)
        self._bias = raw - np.array(target_xy, dtype=float)
        self._stabilizer.reset()
        self.notice.emit(f"Re-centered (offset {self._bias[0]:+.0f}, {self._bias[1]:+.0f} px).")
        return True

    # -- live data -----------------------------------------------------------

    def _on_sample(self, s: GazeSample) -> None:
        self.sample_ready.emit(s)
        now = time.perf_counter()
        if s.features is None:
            if now - self._last_face_t > 1.0:
                self.tracking_changed.emit("no_face")
            return
        self._last_face_t = now
        self.tracking_changed.emit("blinking" if s.blinking else "tracking")

        if self.mapper is None or s.blinking:
            return
        raw = self.mapper.predict(s.features)
        self._history.append(GazeHistoryEntry(s.timestamp, s.features, raw))
        point = self._stabilizer.update(raw - self._bias, s.timestamp)
        if self._overlay_on:
            self.overlay.set_gaze(point)
        self.gaze_point.emit(point)
        if self._engine is not None:
            focused = self._engine.update(point[0], point[1], s.timestamp)
            if focused is not None:
                self.focus_switched.emit(focused)

    def _handle_click(self, x: float, y: float) -> None:
        if self.mapper is None or self._calibrating:
            return
        features = self._click_gate.evaluate((x, y), time.perf_counter(), list(self._history))
        if features is None:
            return
        if self.mapper.add_click_sample(features, (x, y)):
            self.mapper.save()
        self.clicks_learned.emit(self.mapper.click_count)

    def _on_error(self, message: str) -> None:
        self.tracking_changed.emit("error")
        self.error.emit(message)

    # -- lifecycle -----------------------------------------------------------

    def shutdown(self) -> None:
        if self._click_listener is not None:
            self._click_listener.stop()
        self.worker.stop()
        self.overlay.close()
