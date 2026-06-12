"""Per-app exclusion rules: pick apps that gaze must never auto-raise."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .controller import TrafoController


class AppRulesDialog(QDialog):
    """Checkbox list of apps; checked = never auto-raised by gaze."""

    def __init__(self, controller: TrafoController, parent=None):
        super().__init__(parent)
        self.c = controller
        self.setWindowTitle("App rules")
        self.setFixedWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(10)

        title = QLabel("App rules")
        title.setObjectName("Heading")
        root.addWidget(title)
        caption = QLabel(
            "Trafo never auto-raises checked apps. Looking at them also "
            "won't raise whatever is behind them."
        )
        caption.setObjectName("Subtle")
        caption.setWordWrap(True)
        root.addWidget(caption)

        # Running apps merged with saved exclusions (kept even if not running).
        excluded = set(self.c.settings.excluded_apps)
        names = sorted(set(self.c.running_apps()) | excluded, key=str.lower)

        self._boxes: list[QCheckBox] = []
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(2)
        if names:
            for name in names:
                box = QCheckBox(name)
                box.setChecked(name in excluded)
                self._boxes.append(box)
                list_layout.addWidget(box)
        else:
            empty = QLabel("No app windows visible right now.")
            empty.setObjectName("Caption")
            list_layout.addWidget(empty)
        list_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(list_widget)
        scroll.setMinimumHeight(220)
        root.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("Flat")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        save = QPushButton("Save")
        save.setObjectName("Primary")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        root.addLayout(buttons)

    def _save(self) -> None:
        self.c.set_excluded_apps([b.text() for b in self._boxes if b.isChecked()])
        self.accept()
