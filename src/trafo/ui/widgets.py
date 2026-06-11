"""Small reusable styled widgets: status dot, card container, section header."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from . import theme


class StatusDot(QWidget):
    """A small colored circle used as a live state indicator."""

    def __init__(self, color: str = theme.TEXT_DIM):
        super().__init__()
        self.setFixedSize(12, 12)
        self._color = QColor(color)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(1, 1, 10, 10)
        p.end()


class Card(QFrame):
    """A titled rounded container."""

    def __init__(self, title: str | None = None):
        super().__init__()
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(8)
        if title:
            header = QLabel(title.upper())
            header.setObjectName("SectionHeader")
            self._layout.addWidget(header)

    def add(self, widget) -> None:
        self._layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)
