"""Small reusable styled widgets: status dot, pill, card, captioned rows."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from . import theme


class StatusDot(QWidget):
    """A small colored circle with a soft halo, used as a live state indicator."""

    def __init__(self, color: str = theme.TEXT_DIM):
        super().__init__()
        self.setFixedSize(14, 14)
        self._color = QColor(color)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        halo = QColor(self._color)
        halo.setAlpha(50)
        p.setBrush(halo)
        p.drawEllipse(0, 0, 14, 14)
        p.setBrush(self._color)
        p.drawEllipse(3, 3, 8, 8)
        p.end()


class Pill(QLabel):
    """A rounded status chip: set_state("good"|"warn"|"bad"|"neutral"|"accent", text)."""

    def __init__(self, kind: str = "neutral", text: str = ""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state(kind, text)

    def set_state(self, kind: str, text: str) -> None:
        bg, fg = theme.pill_colors(kind)
        self.setText(text)
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 9px;"
            f"padding: 2px 10px; font-size: 11px; font-weight: 600;"
        )


class Card(QFrame):
    """A titled rounded container."""

    def __init__(self, title: str | None = None):
        super().__init__()
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(10)
        if title:
            header = QLabel(title.upper())
            header.setObjectName("SectionHeader")
            self._layout.addWidget(header)

    def add(self, widget) -> None:
        self._layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


def captioned(widget: QWidget, caption: str) -> QWidget:
    """Stack a control over a dim one-line explanation, like a settings row."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(widget)
    label = QLabel(caption)
    label.setObjectName("Caption")
    # Single line on purpose: word-wrapped labels nested in child widgets
    # misreport heightForWidth and overlap neighboring rows.
    # Align the caption with checkbox text (indicator 18px + spacing 10px).
    label.setIndent(28)
    lay.addWidget(label)
    return box
