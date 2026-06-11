"""Dark theme stylesheet, accent palette, and a programmatically drawn app icon."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap

# Palette (shared with the calibration/overlay widgets where useful).
BG = "#16181d"
SURFACE = "#1e2128"
SURFACE_2 = "#262a33"
BORDER = "#323743"
TEXT = "#e8eaed"
TEXT_DIM = "#9aa0ab"
ACCENT = "#50aaff"
ACCENT_DIM = "#3a7fbf"
GOOD = "#5ad17f"
WARN = "#ffac4d"
BAD = "#ff6b6b"

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QLabel#Title {{ font-size: 18px; font-weight: 600; }}
QLabel#Subtle {{ color: {TEXT_DIM}; }}
QLabel#SectionHeader {{
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QPushButton {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 14px;
}}
QPushButton:hover {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background: {SURFACE}; }}
QPushButton#Primary {{
    background: {ACCENT};
    color: #0b1420;
    border: none;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #6cb8ff; }}
QPushButton#Primary:disabled {{ background: {ACCENT_DIM}; color: #0b142080; }}
QCheckBox {{ spacing: 9px; padding: 3px 0; }}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    border: 1px solid {BORDER};
    border-radius: 5px;
    background: {SURFACE_2};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
}}
QCheckBox:disabled {{ color: {TEXT_DIM}; }}
QSpinBox {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 6px;
}}
QToolTip {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px;
}}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(STYLESHEET)


def make_icon(size: int = 256, active: bool = True) -> QIcon:
    """A stylized eye with an iris — drawn, so no asset files are required."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size

    # Rounded dark tile background.
    p.setBrush(QBrush(QColor(BG)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(0, 0, s, s), s * 0.22, s * 0.22)

    accent = QColor(ACCENT if active else TEXT_DIM)

    # Eye almond outline.
    p.setPen(QPen(accent, s * 0.045))
    p.setBrush(Qt.BrushStyle.NoBrush)
    eye = QRectF(s * 0.14, s * 0.30, s * 0.72, s * 0.40)
    path_top = QRectF(eye)
    p.drawChord(path_top, 20 * 16, 140 * 16)
    p.drawChord(path_top, 200 * 16, 140 * 16)

    # Iris + pupil.
    cx, cy = s * 0.5, s * 0.5
    r = s * 0.15
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(accent))
    p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
    p.setBrush(QBrush(QColor(BG)))
    pr = r * 0.45
    p.drawEllipse(QRectF(cx - pr, cy - pr, 2 * pr, 2 * pr))
    p.end()
    return QIcon(pm)
