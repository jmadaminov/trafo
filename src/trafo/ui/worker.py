"""Background thread that runs the gaze pipeline and emits samples as Qt signals."""

from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from ..gaze.estimator import GazePipeline, GazeSample


class GazeWorker(QThread):
    """Emits `sample` (GazeSample) ~at camera rate, `error` (str) on startup failure."""

    sample = Signal(object)
    error = Signal(str)

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._stop = False
        self._debug = False

    def set_debug(self, on: bool) -> None:
        """Toggle preview-frame/landmark output (consumed by the debug view)."""
        self._debug = on

    def run(self) -> None:
        try:
            pipeline = GazePipeline(self.camera_index)
        except Exception as exc:  # camera missing / permission denied
            self.error.emit(str(exc))
            return
        try:
            while not self._stop:
                pipeline.debug = self._debug
                s: GazeSample | None = pipeline.tick()
                if s is None:
                    time.sleep(0.005)  # no new frame yet
                    continue
                self.sample.emit(s)
        finally:
            pipeline.close()

    def stop(self) -> None:
        self._stop = True
        self.wait(3000)
