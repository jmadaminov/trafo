"""First-run onboarding: welcome, permission checklist, then calibration."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import permissions
from . import theme
from .widgets import Card, Pill

# permission state -> (pill kind, pill text)
_STATE_PILL = {
    "ok": ("good", "Granted"),
    "missing": ("bad", "Needs access"),
    "unknown": ("neutral", "Can't verify"),
    "n/a": ("neutral", "—"),
}


class _PermissionRow(QWidget):
    def __init__(self, perm: permissions.Permission):
        super().__init__()
        self.perm = perm
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 6, 0, 6)
        row.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(1)
        name = QLabel(perm.name)
        name.setStyleSheet("font-weight: 600;")
        why = QLabel(perm.why)
        why.setObjectName("Caption")
        # No wrap: wrapped labels inside child widgets misreport their height
        # and overlap the next row. The "why" strings are short by design.
        text.addWidget(name)
        text.addWidget(why)
        row.addLayout(text, 1)

        self.pill = Pill("neutral", "Checking…")
        row.addWidget(self.pill)
        self.grant_btn = QPushButton("Grant…")
        self.grant_btn.setFixedWidth(76)
        self.grant_btn.clicked.connect(self._grant)
        row.addWidget(self.grant_btn)

    def refresh(self) -> str:
        state = permissions.check(self.perm.key)
        kind, text = _STATE_PILL.get(state, ("neutral", state))
        self.pill.set_state(kind, text)
        self.grant_btn.setVisible(state in ("missing", "unknown"))
        return state

    def _grant(self) -> None:
        permissions.request(self.perm.key)


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("Divider")
    line.setFixedHeight(1)
    return line


class OnboardingWindow(QWidget):
    """Emits done() when the user proceeds (calibrate) or skips."""

    done = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to Trafo")
        self.setFixedWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 16)
        root.setSpacing(14)

        # -- Header --
        header = QHBoxLayout()
        header.setSpacing(14)
        logo = QLabel()
        logo.setPixmap(theme.make_icon(96).pixmap(48, 48))
        header.addWidget(logo)
        head_text = QVBoxLayout()
        head_text.setSpacing(2)
        title = QLabel("Welcome to Trafo")
        title.setObjectName("Title")
        sub = QLabel(
            "Trafo watches your eyes through the webcam and brings the "
            "window you look at to the front."
        )
        sub.setObjectName("Subtle")
        sub.setWordWrap(True)
        head_text.addWidget(title)
        head_text.addWidget(sub)
        header.addLayout(head_text, 1)
        root.addLayout(header)

        # -- Permissions --
        self._rows: list[_PermissionRow] = []
        perms = permissions.permissions_for_platform()
        if perms:
            card = Card("Permissions")
            intro = QLabel("Each permission has one specific job — nothing else.")
            intro.setObjectName("Caption")
            card.add(intro)
            for i, perm in enumerate(perms):
                if i:
                    card.add(_divider())
                row = _PermissionRow(perm)
                self._rows.append(row)
                card.add(row)
            root.addWidget(card)

            steps = QLabel(
                "<b>1.</b> Grant each permission &nbsp;→&nbsp; "
                "<b>2.</b> Restart Trafo (macOS applies grants only after a "
                "restart) &nbsp;→&nbsp; <b>3.</b> Calibrate"
            )
            steps.setObjectName("Subtle")
            steps.setWordWrap(True)
            root.addWidget(steps)

            fineprint = QLabel(
                "If a permission stays “Needs access” although System "
                "Settings shows it on, remove Trafo from that list (−) and "
                "add it back — each new beta build looks like a new app to macOS."
            )
            fineprint.setObjectName("Caption")
            fineprint.setWordWrap(True)
            root.addWidget(fineprint)
        else:
            note = QLabel("No special permissions are required on this platform.")
            note.setObjectName("Subtle")
            root.addWidget(note)

        # -- Footer --
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.recheck_btn = QPushButton("Re-check")
        self.recheck_btn.setObjectName("Flat")
        self.recheck_btn.clicked.connect(self._refresh)
        buttons.addWidget(self.recheck_btn)
        self.restart_btn = QPushButton("Restart Trafo")
        self.restart_btn.setToolTip(
            "Quit and reopen Trafo — macOS applies new permission grants "
            "only to a freshly launched app."
        )
        self.restart_btn.clicked.connect(self._restart)
        if permissions.bundle_path() is None:  # dev mode: nothing to relaunch
            self.restart_btn.hide()
        buttons.addWidget(self.restart_btn)
        buttons.addStretch()
        self.skip_btn = QPushButton("Skip for now")
        self.skip_btn.setObjectName("Flat")
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
        # Refresh the pills for feedback, but never hard-block Continue on a
        # permission: calibration only needs the camera, and the rest degrade
        # gracefully (and Screen Recording often reads as "missing" for an
        # unsigned beta even when granted, which would trap the user here).
        for row in self._rows:
            row.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The word-wrapped labels only report their true height once the
        # fixed width is applied; without this the rows render compressed.
        self.resize(self.sizeHint())

    def _restart(self) -> None:
        if permissions.relaunch_app():
            QApplication.instance().quit()

    def _finish(self) -> None:
        self._timer.stop()
        self.done.emit()
        self.close()
