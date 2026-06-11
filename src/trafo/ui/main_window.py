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
from .widgets import Card, Pill, StatusDot, captioned

# state -> (dot color, status line, pill kind, pill text)
_TRACKING = {
    "starting": (theme.WARN, "Starting camera…", "warn", "Starting"),
    "tracking": (theme.GOOD, "Tracking your gaze", "good", "Live"),
    "blinking": (theme.GOOD, "Tracking your gaze", "good", "Live"),
    "no_face": (theme.WARN, "No face in view", "warn", "Paused"),
    "error": (theme.BAD, "Camera unavailable", "bad", "Error"),
}


class MainWindow(QWidget):
    def __init__(self, controller: TrafoController):
        super().__init__()
        self.c = controller
        self.setWindowTitle("Trafo")
        self.setFixedWidth(420)
        self._debug_window = None
        # When True the window hides on close (tray keeps the app alive);
        # when False, closing it quits (no tray available).
        self.hide_on_close = True

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(12)

        # -- Header: icon · name+tagline · live pill --
        header = QHBoxLayout()
        header.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(theme.make_icon(64).pixmap(32, 32))
        header.addWidget(logo)
        name_box = QVBoxLayout()
        name_box.setSpacing(0)
        title = QLabel("Trafo")
        title.setObjectName("Title")
        tagline = QLabel("Look at a window. It comes forward.")
        tagline.setObjectName("Subtle")
        name_box.addWidget(title)
        name_box.addWidget(tagline)
        header.addLayout(name_box, 1)
        self.live_pill = Pill("warn", "Starting")
        header.addWidget(self.live_pill, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        # -- Status card --
        status_card = Card("Status")
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self.dot = StatusDot(theme.WARN)
        self.status_label = QLabel("Starting camera…")
        self.status_label.setObjectName("Heading")
        status_row.addWidget(self.dot)
        status_row.addWidget(self.status_label, 1)
        self.retry_btn = QPushButton("Retry camera")
        self.retry_btn.setToolTip(
            "Reopen the camera — use this after granting Camera permission "
            "or freeing the camera from another app."
        )
        self.retry_btn.clicked.connect(self._retry_camera)
        self.retry_btn.hide()
        status_row.addWidget(self.retry_btn)
        status_card.add_layout(status_row)
        self.cal_label = QLabel()
        self.cal_label.setObjectName("Subtle")
        self.cal_label.setWordWrap(True)
        status_card.add(self.cal_label)
        root.addWidget(status_card)

        # -- Calibration card --
        cal_card = Card("Calibration")
        cal_caption = QLabel(
            "Trafo learns how your eyes map to your displays. Recalibrate "
            "after moving the webcam or changing monitors."
        )
        cal_caption.setObjectName("Caption")
        cal_caption.setWordWrap(True)
        cal_card.add(cal_caption)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setObjectName("Primary")
        self.calibrate_btn.clicked.connect(self._calibrate)
        btn_row.addWidget(self.calibrate_btn, 1)
        self.recenter_btn = QPushButton("Re-center")
        self.recenter_btn.setToolTip(
            "Drifted a little? Look at this button and click it — that "
            "glance cancels the drift. No full recalibration needed."
        )
        self.recenter_btn.clicked.connect(self._recenter)
        btn_row.addWidget(self.recenter_btn, 1)
        cal_card.add_layout(btn_row)
        root.addWidget(cal_card)

        # -- Behavior card --
        beh_card = Card("Behavior")

        self.engine_check = QCheckBox("Focus follows gaze")
        self.engine_check.toggled.connect(self.c.set_engine)
        beh_card.add(captioned(
            self.engine_check,
            "Rest your gaze on a window to bring it forward.",
        ))

        self.overlay_check = QCheckBox("Show gaze dot")
        self.overlay_check.toggled.connect(self.c.set_overlay)
        beh_card.add(captioned(
            self.overlay_check,
            "A dot shows where you're looking.",
        ))

        self.clicks_check = QCheckBox("Learn from clicks")
        self.clicks_check.toggled.connect(self.c.set_click_learning)
        beh_card.add(captioned(
            self.clicks_check,
            "Clicks quietly fine-tune accuracy over time.",
        ))

        dwell_row = QHBoxLayout()
        dwell_label = QLabel("Dwell time")
        dwell_hint = QLabel("delay before focus moves")
        dwell_hint.setObjectName("Caption")
        dwell_row.addWidget(dwell_label)
        dwell_row.addWidget(dwell_hint, 1)
        self.dwell_spin = QSpinBox()
        self.dwell_spin.setRange(200, 3000)
        self.dwell_spin.setSingleStep(50)
        self.dwell_spin.setSuffix(" ms")
        self.dwell_spin.setValue(self.c.settings.dwell_ms)
        self.dwell_spin.valueChanged.connect(self.c.set_dwell_ms)
        dwell_row.addWidget(self.dwell_spin)
        beh_card.add_layout(dwell_row)

        mouse_row = QHBoxLayout()
        mouse_label = QLabel("Mouse pause")
        mouse_hint = QLabel("pause after mouse use")
        mouse_hint.setObjectName("Caption")
        mouse_row.addWidget(mouse_label)
        mouse_row.addWidget(mouse_hint, 1)
        self.mouse_pause_spin = QSpinBox()
        self.mouse_pause_spin.setRange(0, 30)
        self.mouse_pause_spin.setSuffix(" s")
        self.mouse_pause_spin.setToolTip(
            "After you move the mouse, gaze won't switch focus for this "
            "long — the mouse outranks your eyes. 0 disables the pause."
        )
        self.mouse_pause_spin.setValue(self.c.settings.mouse_pause_s)
        self.mouse_pause_spin.valueChanged.connect(self.c.set_mouse_pause_s)
        mouse_row.addWidget(self.mouse_pause_spin)
        beh_card.add_layout(mouse_row)
        root.addWidget(beh_card)

        # -- Footer --
        footer = QHBoxLayout()
        self.notice_label = QLabel()
        self.notice_label.setObjectName("Caption")
        self.notice_label.setWordWrap(True)
        footer.addWidget(self.notice_label, 1)
        self.debug_btn = QPushButton("Debug view")
        self.debug_btn.setObjectName("Flat")
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
        color, text, kind, pill = _TRACKING.get(state, (theme.TEXT_DIM, state, "neutral", "—"))
        self.dot.set_color(color)
        self.status_label.setText(text)
        self.live_pill.set_state(kind, pill)
        self.retry_btn.setVisible(state == "error")

    def _on_clicks(self, count: int) -> None:
        self.cal_label.setText(self.c.calibration_summary())

    def _sync_calibration(self) -> None:
        calibrated = self.c.is_calibrated
        self.cal_label.setText(self.c.calibration_summary())
        self.calibrate_btn.setText("Recalibrate" if calibrated else "Calibrate")
        for box in (self.overlay_check, self.engine_check, self.clicks_check):
            box.setEnabled(calibrated)
        self.recenter_btn.setEnabled(calibrated)
        if calibrated and self.c.settings.learn_from_clicks:
            self._set_checked(self.clicks_check, True)

    # -- actions -------------------------------------------------------------

    def _retry_camera(self) -> None:
        self.notice_label.setText("Reopening camera…")
        self.c.restart_worker()

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Wrapped captions report their true height only at the fixed width.
        self.resize(self.sizeHint())

    def closeEvent(self, event) -> None:
        if self.hide_on_close:
            event.ignore()
            self.hide()
        else:
            event.accept()
