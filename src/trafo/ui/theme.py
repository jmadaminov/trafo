"""Design system: palette, typography, stylesheet, and drawn icons.

Everything visual is defined here so the rest of the UI stays semantic
(object names + roles, no inline colors). Indicator artwork (checkmark,
spinbox chevrons) is rendered to the user cache dir at 1x and @2x because Qt
stylesheets can only reference image files.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, QStandardPaths, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)

# -- palette ------------------------------------------------------------------

BG = "#14161b"
SURFACE = "#1c1f26"
SURFACE_2 = "#252932"
SURFACE_HOVER = "#2b303b"
BORDER = "#2f3540"
BORDER_SOFT = "#272c35"
TEXT = "#e9ecf1"
TEXT_DIM = "#99a1ae"
TEXT_FAINT = "#646c7a"
ACCENT = "#4da3ff"
ACCENT_HOVER = "#6db5ff"
ACCENT_PRESSED = "#3b8ae6"
ACCENT_DIM = "#2f5e94"
GOOD = "#4ade80"
WARN = "#fbbf24"
BAD = "#f87171"

# Tinted backgrounds for status pills (12% alpha over SURFACE).
_PILL_BG = {
    "good": "rgba(74, 222, 128, 0.14)",
    "warn": "rgba(251, 191, 36, 0.14)",
    "bad": "rgba(248, 113, 113, 0.14)",
    "neutral": "rgba(153, 161, 174, 0.14)",
    "accent": "rgba(77, 163, 255, 0.14)",
}
_PILL_FG = {
    "good": GOOD,
    "warn": WARN,
    "bad": BAD,
    "neutral": TEXT_DIM,
    "accent": ACCENT,
}


def pill_colors(kind: str) -> tuple[str, str]:
    """(background, foreground) for a status pill kind."""
    return _PILL_BG.get(kind, _PILL_BG["neutral"]), _PILL_FG.get(kind, TEXT_DIM)


# -- generated indicator artwork ----------------------------------------------


def _assets_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    path = os.path.join(base or os.path.expanduser("~/.cache/trafo"), "ui")
    os.makedirs(path, exist_ok=True)
    return path


def _draw_check(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#0c1524"), size * 0.16, c=Qt.PenCapStyle.RoundCap,
               j=Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    s = size
    path = QPainterPath(QPointF(s * 0.22, s * 0.54))
    path.lineTo(QPointF(s * 0.42, s * 0.74))
    path.lineTo(QPointF(s * 0.78, s * 0.28))
    p.drawPath(path)
    p.end()
    return pm


def _draw_chevron(size: int, up: bool) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(TEXT_DIM), size * 0.18, c=Qt.PenCapStyle.RoundCap,
               j=Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    s = size
    if up:
        path = QPainterPath(QPointF(s * 0.2, s * 0.62))
        path.lineTo(QPointF(s * 0.5, s * 0.34))
        path.lineTo(QPointF(s * 0.8, s * 0.62))
    else:
        path = QPainterPath(QPointF(s * 0.2, s * 0.38))
        path.lineTo(QPointF(s * 0.5, s * 0.66))
        path.lineTo(QPointF(s * 0.8, s * 0.38))
    p.drawPath(path)
    p.end()
    return pm


def _write_assets() -> dict[str, str]:
    """Render indicator art at 1x and @2x (Qt picks @2x on retina) and
    return logical-path mapping for the stylesheet."""
    d = _assets_dir()
    specs = {
        "check": (12, _draw_check),
        "chevron-up": (8, lambda s: _draw_chevron(s, True)),
        "chevron-down": (8, lambda s: _draw_chevron(s, False)),
    }
    paths: dict[str, str] = {}
    for name, (size, draw) in specs.items():
        p1 = os.path.join(d, f"{name}.png")
        p2 = os.path.join(d, f"{name}@2x.png")
        draw(size).save(p1)
        draw(size * 2).save(p2)
        paths[name] = p1.replace("\\", "/")
    return paths


# -- stylesheet -----------------------------------------------------------------


def _stylesheet(assets: dict[str, str]) -> str:
    return f"""
* {{ outline: none; }}
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}

/* typography */
QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: 20px; font-weight: 600; letter-spacing: -0.2px; }}
QLabel#Heading {{ font-size: 15px; font-weight: 600; }}
QLabel#Subtle {{ color: {TEXT_DIM}; }}
QLabel#Caption {{ color: {TEXT_FAINT}; font-size: 12px; }}
QLabel#SectionHeader {{
    color: {TEXT_FAINT};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
}}

/* surfaces */
QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 12px;
}}
QFrame#Divider {{ background: {BORDER_SOFT}; border: none; max-height: 1px; }}

/* buttons */
QPushButton {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 500;
}}
QPushButton:hover {{ background: {SURFACE_HOVER}; border-color: {BORDER}; }}
QPushButton:pressed {{ background: {SURFACE}; }}
QPushButton:focus {{ border: 1px solid {ACCENT_DIM}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE}; border-color: {BORDER_SOFT}; }}

QPushButton#Primary {{
    background: {ACCENT};
    color: #0c1524;
    border: none;
    font-weight: 600;
    padding: 8px 16px;
}}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#Primary:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#Primary:disabled {{ background: {ACCENT_DIM}; color: rgba(12, 21, 36, 0.55); }}

QPushButton#Flat {{
    background: transparent;
    border: none;
    color: {TEXT_DIM};
    padding: 7px 10px;
}}
QPushButton#Flat:hover {{ color: {TEXT}; background: {SURFACE}; }}
QPushButton#Flat:pressed {{ background: {SURFACE_2}; }}

/* checkboxes */
QCheckBox {{ spacing: 10px; padding: 4px 0; background: transparent; }}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    border: 1px solid {BORDER};
    border-radius: 5px;
    background: {SURFACE_2};
}}
QCheckBox::indicator:hover {{ border-color: {TEXT_FAINT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    image: url("{assets['check']}");
}}
QCheckBox::indicator:checked:hover {{ background: {ACCENT_HOVER}; }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}
QCheckBox::indicator:disabled {{ background: {SURFACE}; border-color: {BORDER_SOFT}; }}

/* spin boxes */
QSpinBox {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 4px 6px 4px 10px;
    min-width: 76px;
}}
QSpinBox:focus {{ border-color: {ACCENT_DIM}; }}
QSpinBox::up-button, QSpinBox::down-button {{
    width: 18px;
    background: transparent;
    border: none;
}}
QSpinBox::up-arrow {{ image: url("{assets['chevron-up']}"); width: 8px; height: 8px; }}
QSpinBox::down-arrow {{ image: url("{assets['chevron-down']}"); width: 8px; height: 8px; }}

/* scrollbars */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QToolTip {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
}}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(_stylesheet(_write_assets()))


# -- icons ----------------------------------------------------------------------


def _eye_path(s: float, rect: QRectF) -> QPainterPath:
    """Almond eye outline as a closed path (two mirrored arcs)."""
    path = QPainterPath(QPointF(rect.left(), rect.center().y()))
    path.quadTo(QPointF(rect.center().x(), rect.top() - rect.height() * 0.18),
                QPointF(rect.right(), rect.center().y()))
    path.quadTo(QPointF(rect.center().x(), rect.bottom() + rect.height() * 0.18),
                QPointF(rect.left(), rect.center().y()))
    return path


def make_icon(size: int = 256, active: bool = True) -> QIcon:
    """The app icon: a stylized eye on a softly graded dark tile."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)

    # Tile with a subtle vertical gradient and inner edge highlight.
    tile = QLinearGradient(0, 0, 0, s)
    tile.setColorAt(0.0, QColor("#222b3d"))
    tile.setColorAt(1.0, QColor("#11141c"))
    p.setBrush(QBrush(tile))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(0, 0, s, s), s * 0.225, s * 0.225)
    p.setPen(QPen(QColor(255, 255, 255, 14), s * 0.008))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(s * 0.006, s * 0.006, s * 0.988, s * 0.988),
                      s * 0.22, s * 0.22)

    accent = QColor(ACCENT if active else TEXT_DIM)

    # Eye outline.
    eye = QRectF(s * 0.16, s * 0.32, s * 0.68, s * 0.36)
    p.setPen(QPen(accent, s * 0.042, c=Qt.PenCapStyle.RoundCap))
    p.drawPath(_eye_path(s, eye))

    # Iris with a soft radial sheen, then pupil and catch-light.
    cx, cy, r = s * 0.5, s * 0.5, s * 0.145
    iris = QRadialGradient(QPointF(cx - r * 0.3, cy - r * 0.35), r * 2.1)
    iris.setColorAt(0.0, QColor(ACCENT_HOVER))
    iris.setColorAt(1.0, QColor(ACCENT_PRESSED))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(iris if active else QBrush(accent)))
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.setBrush(QBrush(QColor("#11141c")))
    p.drawEllipse(QPointF(cx, cy), r * 0.44, r * 0.44)
    p.setBrush(QBrush(QColor(255, 255, 255, 200)))
    p.drawEllipse(QPointF(cx - r * 0.38, cy - r * 0.42), r * 0.14, r * 0.14)
    p.end()
    return QIcon(pm)


def make_tray_icon() -> QIcon:
    """Monochrome template glyph for the macOS menu bar (auto light/dark)."""
    size = 44  # rendered crisp, scaled down by the system
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    black = QColor(0, 0, 0)

    eye = QRectF(s * 0.08, s * 0.28, s * 0.84, s * 0.44)
    p.setPen(QPen(black, s * 0.075, c=Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(_eye_path(s, eye))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(black))
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.16, s * 0.16)
    p.end()

    icon = QIcon(pm)
    icon.setIsMask(True)  # macOS template image: adapts to menu-bar appearance
    return icon
