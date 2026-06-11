"""Fullscreen, instruction-led calibration flow.

Sequence: instruction screen -> 9 animated points -> (next display ->
instructions -> 9 points) ... -> fit + save -> summary screen.

Every connected display is visited; targets are recorded in global
virtual-desktop coordinates so one regression covers all monitors.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..gaze.calibration import ScreenLockedMapper
from ..gaze.estimator import GazeSample

# 3x3 outer grid plus 4 quarter points: 13 targets per display. The quarter
# points pin down the interior where the quadratic terms would otherwise bend.
GRID = [(gx, gy) for gy in (0.08, 0.5, 0.92) for gx in (0.08, 0.5, 0.92)] + [
    (0.29, 0.29), (0.71, 0.29), (0.29, 0.71), (0.71, 0.71),
]
SETTLE_S = 0.9  # ring shrink animation; samples ignored
SAMPLES_PER_POINT = 40
FACE_LOST_WARN_S = 1.0

BG = QColor(18, 18, 22)
FG = QColor(235, 235, 240)
DIM = QColor(150, 150, 160)
ACCENT = QColor(80, 170, 255)
WARN = QColor(255, 120, 80)

INSTRUCTIONS_FIRST = [
    "Trafo calibration",
    "",
    "• Sit as you normally do, roughly an arm's length from the screen.",
    "• A blue dot will appear at 9 spots on this display.",
    "• Follow it with your EYES — try to keep your head still.",
    "• Keep looking at each dot while its ring shrinks and fills.",
    "• Blinking is fine; the bad frames are skipped automatically.",
    "",
    "Press  SPACE  to start    ·    ESC cancels",
]

INSTRUCTIONS_NEXT = [
    "Next display",
    "",
    "Same again on this screen — follow the dot with your eyes.",
    "",
    "Press  SPACE  to start    ·    ESC cancels",
]


class CalibrationWindow(QWidget):
    """Emits finished(mapper | None): a fitted GazeMapper, or None if cancelled."""

    finished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setCursor(Qt.CursorShape.BlankCursor)

        self._screens = QGuiApplication.screens()
        self._screen_i = 0
        self._point_i = 0
        self._state = "instructions"  # instructions | settle | collect | done
        self._phase_t0 = 0.0
        self._collected: list[tuple[np.ndarray, np.ndarray, int]] = []  # (features, target, screen)
        self._point_samples = 0
        self._last_face_t = time.perf_counter()
        self._summary = ""

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(30)

    # -- flow ----------------------------------------------------------------

    def start(self) -> None:
        self._move_to_screen(0)

    def _move_to_screen(self, i: int) -> None:
        # Deliberately no showFullScreen(): on macOS a native-fullscreen window
        # lives in its own Space and silently refuses to move to another screen.
        # A frameless stays-on-top window with the screen's geometry looks the
        # same and relocates reliably.
        self._screen_i = i
        self._point_i = 0
        self._state = "instructions"
        screen = self._screens[i]
        self.hide()
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        # Re-assert geometry after show: some platforms reposition on show().
        QTimer.singleShot(50, self._show_on_current_screen)

    def _show_on_current_screen(self) -> None:
        self.setGeometry(self._screens[self._screen_i].geometry())
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def _begin_point(self) -> None:
        self._state = "settle"
        self._phase_t0 = time.perf_counter()
        self._point_samples = 0

    def _point_pos(self) -> tuple[int, int]:
        gx, gy = GRID[self._point_i]
        return int(gx * self.width()), int(gy * self.height())

    def _target_global(self) -> np.ndarray:
        x, y = self._point_pos()
        g = self.geometry().topLeft()
        return np.array([g.x() + x, g.y() + y], dtype=float)

    def _next_point(self) -> None:
        self._point_i += 1
        if self._point_i < len(GRID):
            self._begin_point()
        elif self._screen_i + 1 < len(self._screens):
            self._move_to_screen(self._screen_i + 1)
        else:
            self._finish()

    def _finish(self) -> None:
        mapper = ScreenLockedMapper()
        rects = [tuple(s.geometry().getRect()) for s in self._screens]
        stats = mapper.fit(self._collected, rects)
        mapper.save()
        self._mapper = mapper
        per_screen = "\n".join(
            f"Display {si + 1}: error ≈ {st['rms']:.0f} px"
            f"  ({st['kept']} samples, {st['dropped']} outliers dropped)"
            for si, st in stats.items()
        )
        self._summary = (
            f"Calibration saved  ·  {len(self._screens)} display(s)\n"
            f"{per_screen}\n\n"
            "Press any key to close"
        )
        self._state = "done"

    # -- data ----------------------------------------------------------------

    def on_sample(self, s: GazeSample) -> None:
        if s.features is not None:
            self._last_face_t = time.perf_counter()
        if self._state != "collect":
            return
        if s.features is None or s.blinking:
            return
        self._collected.append((s.features, self._target_global(), self._screen_i))
        self._point_samples += 1
        if self._point_samples >= SAMPLES_PER_POINT:
            self._next_point()

    # -- input ---------------------------------------------------------------

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape and self._state != "done":
            self._close_with(None)
        elif self._state == "instructions" and e.key() == Qt.Key.Key_Space:
            self._begin_point()
        elif self._state == "done":
            self._close_with(self._mapper)

    def _close_with(self, result) -> None:
        self._timer.stop()
        self.finished.emit(result)
        self.close()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        if self._state == "instructions":
            self._paint_text(p, INSTRUCTIONS_FIRST if self._screen_i == 0 else INSTRUCTIONS_NEXT)
        elif self._state == "done":
            self._paint_text(p, self._summary.split("\n"))
        else:
            self._paint_point(p)
        p.end()

    def _paint_text(self, p: QPainter, lines: list[str]) -> None:
        title_font = QFont(self.font().family(), 28, QFont.Weight.Bold)
        body_font = QFont(self.font().family(), 16)
        line_heights = [56 if i == 0 else 30 for i in range(len(lines))]
        y = (self.height() - sum(line_heights)) // 2
        for line, lh in zip(lines, line_heights):
            p.setFont(title_font if lh == 56 else body_font)
            p.setPen(QPen(FG if lh == 56 else DIM))
            p.drawText(0, y, self.width(), lh, Qt.AlignmentFlag.AlignHCenter, line)
            y += lh

    def _paint_point(self, p: QPainter) -> None:
        x, y = self._point_pos()
        now = time.perf_counter()

        if self._state == "settle":
            frac = min(1.0, (now - self._phase_t0) / SETTLE_S)
            radius = 36 - 24 * frac
            if frac >= 1.0:
                self._state = "collect"
        else:
            frac = self._point_samples / SAMPLES_PER_POINT
            radius = 12

        # progress arc while collecting
        p.setPen(QPen(ACCENT, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self._state == "collect":
            p.drawArc(x - 22, y - 22, 44, 44, 90 * 16, -int(360 * 16 * frac))
        else:
            p.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))

        p.setBrush(ACCENT)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(x - 6, y - 6, 12, 12)

        p.setPen(QPen(DIM))
        p.setFont(QFont(self.font().family(), 13))
        p.drawText(
            self.rect().adjusted(0, 0, 0, -20),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            f"Display {self._screen_i + 1}/{len(self._screens)}   ·   "
            f"Point {self._point_i + 1}/{len(GRID)}   ·   ESC to cancel",
        )

        if time.perf_counter() - self._last_face_t > FACE_LOST_WARN_S:
            p.setPen(QPen(WARN))
            p.setFont(QFont(self.font().family(), 18, QFont.Weight.Bold))
            p.drawText(
                self.rect().adjusted(0, 60, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "Face not detected — check lighting / camera",
            )
