"""Fullscreen, instruction-led smooth-pursuit calibration flow.

Sequence per display: instruction screen -> settle (dot appears, ring
shrinks while the eye acquires it) -> pursue (the dot sweeps the screen for
~35 s while features are recorded continuously) -> next display ... ->
lag-compensated pairing + fit + save -> summary screen.

Every connected display is visited; targets are recorded in global
virtual-desktop coordinates so one model covers all monitors.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..gaze.calibration import ScreenLockedMapper
from ..gaze.estimator import GazeSample
from ..gaze.pursuit import PURSUIT_DURATION_S, pair_with_lag, pursuit_path

SETTLE_S = 1.2  # ring shrink on the start point; samples ignored
FACE_LOST_WARN_S = 1.0
MIN_SAMPLES = 100  # below this the display must be redone (face was lost)

BG = QColor(18, 18, 22)
FG = QColor(235, 235, 240)
DIM = QColor(150, 150, 160)
ACCENT = QColor(80, 170, 255)
WARN = QColor(255, 120, 80)

INSTRUCTIONS_FIRST = [
    "Trafo calibration",
    "",
    "• Sit as you normally do, roughly an arm's length from the screen.",
    "• A dot will glide around this display for about half a minute.",
    "• Follow it with your EYES and let it pull your gaze smoothly.",
    "• Keep your head still-ish — eyes do the work.",
    "• Blinking is fine; bad frames are skipped automatically.",
    "",
    "Press  SPACE  to start    ·    ESC cancels",
]

INSTRUCTIONS_NEXT = [
    "Next display",
    "",
    "Same again on this screen — follow the gliding dot with your eyes.",
    "",
    "Press  SPACE  to start    ·    ESC cancels",
]

INSTRUCTIONS_RETRY = [
    "Let's redo this display",
    "",
    "Your face wasn't visible long enough — check lighting and camera angle.",
    "",
    "Press  SPACE  to retry    ·    ESC cancels",
]


class CalibrationWindow(QWidget):
    """Emits finished(mapper | None): a fitted mapper, or None if cancelled."""

    finished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setCursor(Qt.CursorShape.BlankCursor)

        self._screens = QGuiApplication.screens()
        self._screen_i = 0
        self._state = "instructions"  # instructions | settle | pursue | done
        self._retry = False
        self._phase_t0 = 0.0  # start of the current settle/pursue phase
        self._target_fn = None  # global path for the current screen
        self._times: list[float] = []  # seconds since pursue start, per sample
        self._feats: list[np.ndarray] = []
        self._collected: list[tuple[np.ndarray, np.ndarray, int]] = []  # (features, target, screen)
        self._lags: dict[int, float] = {}
        self._last_face_t = time.perf_counter()
        self._summary = ""

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)  # smooth dot motion

    # -- flow ----------------------------------------------------------------

    def start(self) -> None:
        self._move_to_screen(0)

    def _move_to_screen(self, i: int) -> None:
        # Deliberately no showFullScreen(): on macOS a native-fullscreen window
        # lives in its own Space and silently refuses to move to another screen.
        # A frameless stays-on-top window with the screen's geometry looks the
        # same and relocates reliably.
        self._screen_i = i
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

    def _begin_pursuit(self) -> None:
        g = self.geometry()
        self._target_fn = pursuit_path((g.x(), g.y(), g.width(), g.height()))
        self._times, self._feats = [], []
        self._state = "settle"
        self._phase_t0 = time.perf_counter()

    def _end_pursuit(self) -> None:
        if len(self._feats) < MIN_SAMPLES:
            # Face was lost for most of the sweep — redo this display.
            self._retry = True
            self._state = "instructions"
            return
        self._retry = False
        # Pair features with lag-shifted targets (the eye trails the dot).
        lag, targets = pair_with_lag(np.array(self._times), self._feats, self._target_fn)
        self._lags[self._screen_i] = lag
        self._collected.extend(
            (f, t, self._screen_i) for f, t in zip(self._feats, targets)
        )
        if self._screen_i + 1 < len(self._screens):
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
            f"  ({st['kept']} samples, {st['dropped']} dropped,"
            f" lag {self._lags.get(si, 0) * 1000:.0f} ms)"
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
        if self._state != "pursue":
            return
        if s.features is None or s.blinking:
            return
        rel_t = s.timestamp - self._phase_t0
        if 0.0 <= rel_t <= PURSUIT_DURATION_S:
            self._times.append(rel_t)
            self._feats.append(s.features)

    # -- input ---------------------------------------------------------------

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape and self._state != "done":
            self._close_with(None)
        elif self._state == "instructions" and e.key() == Qt.Key.Key_Space:
            self._begin_pursuit()
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
            lines = (
                INSTRUCTIONS_RETRY if self._retry
                else INSTRUCTIONS_FIRST if self._screen_i == 0
                else INSTRUCTIONS_NEXT
            )
            self._paint_text(p, lines)
        elif self._state == "done":
            self._paint_text(p, self._summary.split("\n"))
        elif self._state == "pairing":
            self._paint_text(p, ["", "Processing…"])
        else:
            self._paint_pursuit(p)
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

    def _local_dot(self, t: float) -> tuple[int, int]:
        g = self.geometry().topLeft()
        pos = self._target_fn(t)
        return int(pos[0] - g.x()), int(pos[1] - g.y())

    def _paint_pursuit(self, p: QPainter) -> None:
        now = time.perf_counter()

        if self._state == "settle":
            frac = min(1.0, (now - self._phase_t0) / SETTLE_S)
            x, y = self._local_dot(0.0)
            radius = 36 - 24 * frac
            p.setPen(QPen(ACCENT, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
            if frac >= 1.0:
                self._state = "pursue"
                self._phase_t0 = now  # pursue clock starts now
        else:
            elapsed = now - self._phase_t0
            if elapsed >= PURSUIT_DURATION_S:
                # Window moves / fits must not run inside paintEvent.
                self._state = "pairing"
                QTimer.singleShot(0, self._end_pursuit)
                return
            x, y = self._local_dot(elapsed)
            # Thin progress bar along the bottom edge.
            frac = elapsed / PURSUIT_DURATION_S
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 90))
            p.drawRect(0, self.height() - 4, int(self.width() * frac), 4)

        p.setBrush(ACCENT)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(x - 7, y - 7, 14, 14)

        p.setPen(QPen(DIM))
        p.setFont(QFont(self.font().family(), 13))
        pct = (
            0 if self._state == "settle"
            else int(100 * (now - self._phase_t0) / PURSUIT_DURATION_S)
        )
        p.drawText(
            self.rect().adjusted(0, 0, 0, -20),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            f"Display {self._screen_i + 1}/{len(self._screens)}   ·   "
            f"{pct}%   ·   follow the dot   ·   ESC to cancel",
        )

        if time.perf_counter() - self._last_face_t > FACE_LOST_WARN_S:
            p.setPen(QPen(WARN))
            p.setFont(QFont(self.font().family(), 18, QFont.Weight.Bold))
            p.drawText(
                self.rect().adjusted(0, 60, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "Face not detected — check lighting / camera",
            )
