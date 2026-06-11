"""Threaded webcam capture with latest-frame semantics.

The capture thread reads frames as fast as the camera delivers them and keeps
only the most recent one, so consumers (landmark detection, demos) never
process a stale backlog.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    # 1080p default: the iris spans only ~10 px at 720p — every extra pixel
    # of eye resolution directly reduces gaze jitter.
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080):
        self.index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera {index}. "
                "Check that the camera is connected and that this terminal has "
                "camera permission (macOS: System Settings > Privacy & Security > Camera)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_ts: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

        # Measured capture rate, updated by the reader thread.
        self.fps: float = 0.0

    def start(self) -> "Camera":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        last = time.perf_counter()
        alpha = 0.9  # EMA smoothing for the FPS estimate
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else alpha * self.fps + (1 - alpha) * inst
            with self._lock:
                self._frame = frame
                self._frame_ts = now

    def latest(self) -> tuple[np.ndarray | None, float]:
        """Return (frame, capture_timestamp). Frame is BGR, or None before first capture."""
        with self._lock:
            return self._frame, self._frame_ts

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
