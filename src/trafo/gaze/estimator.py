"""GazePipeline: camera frame -> landmarks -> feature vector, one tick at a time.

Qt-free on purpose: the UI wraps it in a worker thread, and the focus engine
(M6) can drive it headlessly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..camera import Camera
from . import features as features_mod
from .landmarks import FaceLandmarker


@dataclass
class GazeSample:
    features: np.ndarray | None  # None when no face (or no new frame yet)
    blinking: bool
    timestamp: float  # capture time (perf_counter seconds)
    fps: float = 0.0  # camera capture rate
    # Populated only when debug mode is on (the live debug view), else None:
    frame: np.ndarray | None = None  # BGR preview frame (downscaled, mirrored)
    eye_points: tuple[np.ndarray, np.ndarray] | None = None  # (right, left) px in frame coords
    iris_points: tuple[np.ndarray, np.ndarray] | None = None  # (right, left) px


class GazePipeline:
    POST_BLINK_FRAMES = 2  # lids keep settling briefly after a blink "ends"
    DEBUG_FRAME_WIDTH = 480  # downscale preview frames to keep signal traffic light

    def __init__(self, camera_index: int = 0):
        self.camera = Camera(camera_index).start()
        self.landmarker = FaceLandmarker()
        self._t0 = time.perf_counter()
        self._last_frame_ts = 0.0
        self._post_blink = 0
        self.debug = False

    def tick(self) -> GazeSample | None:
        """Process the latest frame. Returns None if no new frame has arrived."""
        frame, ts = self.camera.latest()
        if frame is None or ts <= self._last_frame_ts:
            return None
        self._last_frame_ts = ts

        face = self.landmarker.detect(frame, int((time.perf_counter() - self._t0) * 1000))
        if face is None:
            return GazeSample(None, False, ts, self.camera.fps, *self._debug_blank(frame))
        blinking = features_mod.is_blinking(face)
        if blinking:
            self._post_blink = self.POST_BLINK_FRAMES
        elif self._post_blink > 0:
            self._post_blink -= 1
            blinking = True  # treat the settling frames as still-blinking

        dbg_frame, eye_pts, iris_pts = self._debug_overlay(frame, face)
        return GazeSample(
            features_mod.extract(face), blinking, ts, self.camera.fps, dbg_frame, eye_pts, iris_pts
        )

    def _debug_blank(self, frame):
        if not self.debug:
            return (None, None, None)
        return (self._downscale(frame), None, None)

    def _downscale(self, frame):
        import cv2

        h, w = frame.shape[:2]
        scale = self.DEBUG_FRAME_WIDTH / w
        small = cv2.resize(frame, (self.DEBUG_FRAME_WIDTH, int(h * scale)))
        return cv2.flip(small, 1)  # mirror, like a mirror

    def _debug_overlay(self, frame, face):
        if not self.debug:
            return (None, None, None)
        from .landmarks import LEFT_EYE, RIGHT_EYE

        h, w = frame.shape[:2]
        scale = self.DEBUG_FRAME_WIDTH / w
        sw = self.DEBUG_FRAME_WIDTH
        px = face.pixels(w, h) * scale
        # Mirror x to match the flipped preview frame.
        px[:, 0] = sw - px[:, 0]
        eye_pts = (px[RIGHT_EYE].astype(int), px[LEFT_EYE].astype(int))
        iris_pts = (
            px[[features_mod.RIGHT_IRIS_CENTER]].astype(int)[0],
            px[[features_mod.LEFT_IRIS_CENTER]].astype(int)[0],
        )
        return (self._downscale(frame), eye_pts, iris_pts)

    def close(self) -> None:
        self.landmarker.close()
        self.camera.stop()
