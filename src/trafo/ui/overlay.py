"""Always-on-top, click-through gaze dot that follows the estimated gaze point."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

SIZE = 44


class GazeOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # macOS hides Tool windows when the app deactivates — which happens on
        # every focus switch the engine performs. Keep the dot visible anyway.
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.resize(SIZE, SIZE)

    def set_gaze(self, point: np.ndarray) -> None:
        """Move the dot to an (already stabilized) global gaze point."""
        rect = QGuiApplication.primaryScreen().virtualGeometry()
        x = min(max(point[0], rect.left()), rect.right())
        y = min(max(point[1], rect.top()), rect.bottom())
        self.move(int(x) - SIZE // 2, int(y) - SIZE // 2)
        # Callers only feed gaze while the dot is meant to be shown; if the OS
        # hid the window (e.g. app deactivation on a focus switch), undo that.
        if not self.isVisible():
            self.show()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(80, 170, 255, 230))
        p.setBrush(QColor(80, 170, 255, 70))
        p.drawEllipse(2, 2, SIZE - 4, SIZE - 4)
        p.setBrush(QColor(80, 170, 255, 220))
        p.setPen(Qt.PenStyle.NoPen)
        c = SIZE // 2
        p.drawEllipse(c - 4, c - 4, 8, 8)
        p.end()
