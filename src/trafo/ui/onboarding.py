"""First-run onboarding: welcome, permission checklist, then calibration."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import permissions
from . import theme
from .widgets import Card, StatusDot

_STATE_COLOR = {
    "ok": (theme.GOOD, "Granted"),
    "missing": (theme.BAD, "Not granted"),
    "unknown": (theme.WARN, "Tap Grant if needed"),
    "n/a": (theme.TEXT_DIM, "—"),
}


class _PermissionRow(QWidget):
    def __init__(self, perm: permissions.Permission):
        super().__init__()
        self.perm = perm
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        self.dot = StatusDot()
        row.addWidget(self.dot)
        text = QVBoxLayout()
        name = QLabel(perm.name)
        why = QLabel(perm.why)
        why.setObjectName("Subtle")
        text.addWidget(name)
        text.addWidget(why)
        row.addLayout(text, 1)
        self.state_label = QLabel()
        self.state_label.setObjectName("Subtle")
        row.addWidget(self.state_label)
        self.grant_btn = QPushButton("Grant")
        self.grant_btn.clicked.connect(self._grant)
        row.addWidget(self.grant_btn)

    def refresh(self) -> str:
        state = permissions.check(self.perm.key)
        color, text = _STATE_COLOR.get(state, (theme.TEXT_DIM, state))
        self.dot.set_color(color)
        self.state_label.setText(text)
        self.grant_btn.setVisible(state in ("missing", "unknown"))
        return state

    def _grant(self) -> None:
        permissions.request(self.perm.key)


class OnboardingWindow(QWidget):
    """Emits done() when the user proceeds (calibrate) or skips."""

    done = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to Trafo")
        self.setMinimumWidth(460)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        title = QLabel("Welcome to Trafo")
        title.setObjectName("Title")
        root.addWidget(title)
        intro = QLabel(
            "Trafo watches your eyes through the webcam and brings the window "
            "you look at to the front. First, grant a few permissions — Trafo "
            "needs each one for a specific job."
        )
        intro.setObjectName("Subtle")
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._rows: list[_PermissionRow] = []
        perms = permissions.permissions_for_platform()
        if perms:
            card = Card("Permissions")
            for perm in perms:
                row = _PermissionRow(perm)
                self._rows.append(row)
                card.add(row)
            hint = QLabel(
                "After granting in System Settings you may need to quit and "
                "reopen Trafo for some permissions to take effect."
            )
            hint.setObjectName("Subtle")
            hint.setWordWrap(True)
            card.add(hint)
            root.addWidget(card)
        else:
            note = QLabel("No special permissions are required on this platform.")
            note.setObjectName("Subtle")
            root.addWidget(note)

        buttons = QHBoxLayout()
        self.recheck_btn = QPushButton("Re-check")
        self.recheck_btn.clicked.connect(self._refresh)
        buttons.addWidget(self.recheck_btn)
        buttons.addStretch()
        self.skip_btn = QPushButton("Skip for now")
        self.skip_btn.clicked.connect(self._finish)
        buttons.addWidget(self.skip_btn)
        self.continue_btn = QPushButton("Continue to calibration")
        self.continue_btn.setObjectName("Primary")
        self.continue_btn.clicked.connect(self._finish)
        buttons.addWidget(self.continue_btn)
        root.addLayout(buttons)

        self._refresh()
        # Live re-poll so granting in System Settings reflects without a click.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1500)

    def _refresh(self) -> None:
        states = [row.refresh() for row in self._rows]
        ready = all(s in ("ok", "n/a", "unknown") for s in states)
        self.continue_btn.setEnabled(ready)

    def _finish(self) -> None:
        self._timer.stop()
        self.done.emit()
        self.close()
