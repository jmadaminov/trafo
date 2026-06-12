"""Live debug / quality view: camera feed with landmark overlay + live stats.

Lets a beta tester see *why* tracking is good or bad — face visibility,
landmark lock, FPS, blink state, which screen is locked, and whether the gaze
estimate is currently fixating. Enables debug frames on the worker only while
this window is open.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from . import theme
from .controller import TrafoController
from .widgets import Card


class DebugView(QWidget):
    def __init__(self, controller: TrafoController):
        super().__init__()
        self.c = controller
        self.setWindowTitle("Trafo — Debug")
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        body = QHBoxLayout()
        self.video = QLabel()
        self.video.setFixedSize(480, 360)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet(
            f"background: #000; border: 1px solid {theme.BORDER}; border-radius: 8px;"
        )
        self.video.setText("Waiting for camera…")
        body.addWidget(self.video)

        stats = Card("Live stats")
        self._stat_labels: dict[str, QLabel] = {}
        for key in ("Face", "FPS", "Blink", "Locked screen", "Gaze", "Stability", "Clicks"):
            row = QHBoxLayout()
            name = QLabel(key)
            name.setObjectName("Subtle")
            value = QLabel("—")
            self._stat_labels[key] = value
            row.addWidget(name)
            row.addStretch()
            row.addWidget(value)
            stats.add_layout(row)
        body.addWidget(stats, 1)
        root.addLayout(body)

        hint = QLabel(
            "Iris dots should sit on your pupils and move when you look around. "
            "If 'Face' drops out or FPS is low, improve lighting or move closer."
        )
        hint.setObjectName("Subtle")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.c.sample_ready.connect(self._on_sample, Qt.ConnectionType.QueuedConnection)
        self.c.gaze_point.connect(self._on_gaze, Qt.ConnectionType.QueuedConnection)
        self.c.set_debug(True)

    def _on_gaze(self, point) -> None:
        self._set("Gaze", f"{point[0]:.0f}, {point[1]:.0f}")

    def _set(self, key: str, value: str, color: str | None = None) -> None:
        label = self._stat_labels[key]
        label.setText(value)
        label.setStyleSheet(f"color: {color};" if color else "")

    def _on_sample(self, s) -> None:
        self._set("FPS", f"{s.fps:.0f}")
        if s.features is None:
            self._set("Face", "not detected", theme.BAD)
            if s.frame is not None:
                self._render(s.frame, None, None)
            return
        self._set("Face", "tracking", theme.GOOD)
        self._set("Blink", "yes" if s.blinking else "no",
                  theme.WARN if s.blinking else theme.TEXT_DIM)

        if self.c.mapper is not None:
            mapper = self.c.mapper
            if mapper.locked_screen is not None and mapper.is_fitted:
                proba = mapper.classify_proba(s.features)
                conf = float(np.max(proba))
                self._set("Locked screen",
                          f"{mapper.locked_screen + 1} · {conf:.0%}",
                          theme.GOOD if conf >= 0.8 else theme.WARN)
            else:
                self._set("Locked screen", "—")
            self._set("Clicks", str(mapper.click_count))
        stab = self.c._stabilizer
        self._set("Stability", "fixating" if stab.is_fixating else "moving",
                  theme.GOOD if stab.is_fixating else theme.TEXT_DIM)

        if s.frame is not None:
            self._render(s.frame, s.eye_points, s.iris_points)

    def _render(self, frame_bgr, eye_points, iris_points) -> None:
        import cv2

        frame = frame_bgr.copy()
        if eye_points is not None:
            for eye in eye_points:
                cv2.polylines(frame, [eye], True, (90, 220, 120), 1)
        if iris_points is not None:
            for iris in iris_points:
                cv2.circle(frame, tuple(int(v) for v in iris), 3, (255, 120, 80), -1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.video.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def closeEvent(self, event) -> None:
        self.c.set_debug(False)
        for sig, slot in ((self.c.sample_ready, self._on_sample),
                          (self.c.gaze_point, self._on_gaze)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        event.accept()
