"""Main control window — a styled view onto TrafoController."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .controller import TrafoController
from .widgets import Card, StatusDot

_TRACKING = {
    "starting": (theme.WARN, "Starting camera…"),
    "tracking": (theme.GOOD, "Tracking"),
    "blinking": (theme.GOOD, "Tracking (blink)"),
    "no_face": (theme.WARN, "No face detected"),
    "error": (theme.BAD, "Camera error"),
}


class MainWindow(QWidget):
    def __init__(self, controller: TrafoController):
        super().__init__()
        self.c = controller
        self.setWindowTitle("Trafo")
        self.setMinimumWidth(380)
        self._debug_window = None
        # When True the window hides on close (tray keeps the app alive);
        # when False, closing it quits (no tray available).
        self.hide_on_close = True

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Trafo")
        title.setObjectName("Title")
        subtitle = QLabel("Look at a window to bring it to the front")
        subtitle.setObjectName("Subtle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # -- Status card --
        status_card = Card("Status")
        status_row = QHBoxLayout()
        self.dot = StatusDot()
        self.status_label = QLabel("Starting camera…")
        status_row.addWidget(self.dot)
        status_row.addWidget(self.status_label, 1)
        status_card.add_layout(status_row)
        self.cal_label = QLabel()
        self.cal_label.setObjectName("Subtle")
        self.cal_label.setWordWrap(True)
        status_card.add(self.cal_label)
        root.addWidget(status_card)

        # -- Calibration card --
        cal_card = Card("Calibration")
        self.calibrate_btn = QPushButton("Calibrate…")
        self.calibrate_btn.setObjectName("Primary")
        self.calibrate_btn.clicked.connect(self._calibrate)
        cal_card.add(self.calibrate_btn)
        self.recenter_btn = QPushButton("Re-center (look here, then click)")
        self.recenter_btn.setToolTip(
            "Clicking requires looking here — that glance cancels any drift."
        )
        self.recenter_btn.clicked.connect(self._recenter)
        cal_card.add(self.recenter_btn)
        root.addWidget(cal_card)

        # -- Behavior card --
        beh_card = Card("Behavior")
        self.overlay_check = QCheckBox("Show gaze dot")
        self.overlay_check.toggled.connect(self.c.set_overlay)
        beh_card.add(self.overlay_check)

        self.engine_check = QCheckBox("Focus follows gaze")
        self.engine_check.toggled.connect(self.c.set_engine)
        beh_card.add(self.engine_check)

        dwell_row = QHBoxLayout()
        dwell_row.addWidget(QLabel("Dwell time"))
        dwell_row.addStretch()
        self.dwell_spin = QSpinBox()
        self.dwell_spin.setRange(200, 3000)
        self.dwell_spin.setSingleStep(50)
        self.dwell_spin.setSuffix(" ms")
        self.dwell_spin.setValue(self.c.settings.dwell_ms)
        self.dwell_spin.valueChanged.connect(self.c.set_dwell_ms)
        dwell_row.addWidget(self.dwell_spin)
        beh_card.add_layout(dwell_row)

        self.clicks_check = QCheckBox("Learn from clicks (improves accuracy)")
        self.clicks_check.setToolTip(
            "You look at what you click. Stable, on-target clicks become fresh "
            "training data; suspicious ones are ignored."
        )
        self.clicks_check.toggled.connect(self.c.set_click_learning)
        beh_card.add(self.clicks_check)
        root.addWidget(beh_card)

        # -- Footer --
        footer = QHBoxLayout()
        self.notice_label = QLabel()
        self.notice_label.setObjectName("Subtle")
        self.notice_label.setWordWrap(True)
        footer.addWidget(self.notice_label, 1)
        self.debug_btn = QPushButton("Debug view")
        self.debug_btn.clicked.connect(self._open_debug)
        footer.addWidget(self.debug_btn)
        root.addLayout(footer)

        self._wire()
        self._sync_calibration()

    # -- controller wiring ---------------------------------------------------

    def _wire(self) -> None:
        c = self.c
        c.tracking_changed.connect(self._on_tracking)
        c.error.connect(lambda m: self.notice_label.setText(m))
        c.calibration_changed.connect(self._sync_calibration)
        c.clicks_learned.connect(self._on_clicks)
        c.notice.connect(lambda m: self.notice_label.setText(m))
        c.focus_switched.connect(
            lambda w: self.notice_label.setText(f"Focused: {w.app} — {w.title or w.app}")
        )
        # Reflect programmatic toggles (e.g. controller disabling a feature).
        c.overlay_changed.connect(lambda on: self._set_checked(self.overlay_check, on))
        c.engine_changed.connect(lambda on: self._set_checked(self.engine_check, on))
        c.click_learning_changed.connect(lambda on: self._set_checked(self.clicks_check, on))

    @staticmethod
    def _set_checked(box: QCheckBox, on: bool) -> None:
        box.blockSignals(True)
        box.setChecked(on)
        box.blockSignals(False)

    def _on_tracking(self, state: str) -> None:
        color, text = _TRACKING.get(state, (theme.TEXT_DIM, state))
        self.dot.set_color(color)
        self.status_label.setText(text)

    def _on_clicks(self, count: int) -> None:
        self.cal_label.setText(self.c.calibration_summary())

    def _sync_calibration(self) -> None:
        calibrated = self.c.is_calibrated
        self.cal_label.setText(self.c.calibration_summary())
        self.calibrate_btn.setText("Recalibrate…" if calibrated else "Calibrate…")
        for box in (self.overlay_check, self.engine_check, self.clicks_check):
            box.setEnabled(calibrated)
        self.recenter_btn.setEnabled(calibrated)
        if calibrated and self.c.settings.learn_from_clicks:
            self._set_checked(self.clicks_check, True)

    # -- actions -------------------------------------------------------------

    def _calibrate(self) -> None:
        self._set_checked(self.overlay_check, False)
        self._set_checked(self.engine_check, False)
        window = self.c.begin_calibration()
        window.finished.connect(lambda _m: (self.raise_(), self.activateWindow()))
        window.start()

    def _recenter(self) -> None:
        center = self.recenter_btn.mapToGlobal(self.recenter_btn.rect().center())
        if not self.c.recenter((center.x(), center.y())):
            self.notice_label.setText("Look at the button while clicking it.")

    def _open_debug(self) -> None:
        from .debug_view import DebugView

        if self._debug_window is None:
            self._debug_window = DebugView(self.c)
            self._debug_window.destroyed.connect(self._debug_closed)
        self._debug_window.show()
        self._debug_window.raise_()

    def _debug_closed(self) -> None:
        self._debug_window = None

    def closeEvent(self, event) -> None:
        if self.hide_on_close:
            event.ignore()
            self.hide()
        else:
            event.accept()
