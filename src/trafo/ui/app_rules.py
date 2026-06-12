"""Per-app exclusion rules: pick apps that gaze must never auto-raise."""

from __future__ import annotations

import plistlib
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .controller import TrafoController


def app_name_from_bundle(path: str) -> str:
    """The app name as it appears as a window owner, from a .app bundle.

    Window owner names come from the bundle's CFBundleName (e.g.
    "Visual Studio Code.app" reports windows as "Code"), so prefer it over
    the file name; fall back to the executable name, then the file stem.
    """
    bundle = Path(path)
    try:
        with open(bundle / "Contents" / "Info.plist", "rb") as f:
            info = plistlib.load(f)
        name = info.get("CFBundleName") or info.get("CFBundleExecutable")
        if name:
            return str(name)
    except Exception:
        pass
    return bundle.stem


class AppRulesDialog(QDialog):
    """Editable list of apps that gaze never auto-raises."""

    def __init__(self, controller: TrafoController, parent=None):
        super().__init__(parent)
        self.c = controller
        self.setWindowTitle("App rules")
        self.setFixedWidth(400)

        self._names: list[str] = sorted(
            set(self.c.settings.excluded_apps), key=str.lower
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(10)

        title = QLabel("App rules")
        title.setObjectName("Heading")
        root.addWidget(title)
        caption = QLabel(
            "Trafo never auto-raises these apps. Looking at them also "
            "won't raise whatever is behind them."
        )
        caption.setObjectName("Subtle")
        caption.setWordWrap(True)
        root.addWidget(caption)

        self._list_layout = QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(2)
        list_widget = QWidget()
        list_widget.setLayout(self._list_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(list_widget)
        scroll.setMinimumHeight(180)
        root.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add app…")
        add_btn.setToolTip("Choose any installed application.")
        add_btn.clicked.connect(self._add_app)
        buttons.addWidget(add_btn)
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

        self._rebuild_list()

    # -- list management -------------------------------------------------------

    def _rebuild_list(self) -> None:
        while (item := self._list_layout.takeAt(0)) is not None:
            if item.widget() is not None:
                item.widget().deleteLater()
        if not self._names:
            empty = QLabel("No apps excluded yet — click “Add app…”.")
            empty.setObjectName("Caption")
            self._list_layout.addWidget(empty)
        for name in self._names:
            self._list_layout.addWidget(self._make_row(name))
        self._list_layout.addStretch()

    def _make_row(self, name: str) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.addWidget(QLabel(name), 1)
        remove = QPushButton("Remove")
        remove.setObjectName("Flat")
        remove.clicked.connect(lambda: self._remove(name))
        lay.addWidget(remove)
        return row

    def _remove(self, name: str) -> None:
        self._names = [n for n in self._names if n != name]
        self._rebuild_list()

    def _add_app(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an application", "/Applications", "Applications (*.app)"
        )
        if not path:
            return
        name = app_name_from_bundle(path)
        if name and name not in self._names:
            self._names = sorted([*self._names, name], key=str.lower)
            self._rebuild_list()

    def _save(self) -> None:
        self.c.set_excluded_apps(self._names)
        self.accept()
