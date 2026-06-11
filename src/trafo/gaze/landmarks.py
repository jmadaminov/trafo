"""MediaPipe FaceLandmarker wrapper producing 478 face landmarks (incl. iris).

The .task model file is not bundled with the mediapipe wheel; it is downloaded
once to ~/.cache/trafo/ on first use (~3 MB).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
CACHE_DIR = Path.home() / ".cache" / "trafo"

# Landmark index sets (MediaPipe face mesh with iris refinement, 478 points).
# "Right"/"left" are the subject's anatomical sides.
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_IRIS_CENTER = 468  # 468-472: right iris (center first)
LEFT_IRIS_CENTER = 473  # 473-477: left iris (center first)
RIGHT_EYE_CORNERS = (33, 133)  # outer, inner
LEFT_EYE_CORNERS = (263, 362)  # outer, inner


def ensure_model() -> Path:
    path = CACHE_DIR / "face_landmarker.task"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmark model to {path} ...")
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
        print("Done.")
    return path


@dataclass
class FaceResult:
    """Landmarks as (478, 3) array; x, y normalized to [0,1] of the frame, z relative depth.

    `transform` is MediaPipe's 4x4 facial transformation matrix (canonical face
    space -> camera space); its 3x3 upper-left block is the head rotation.
    """

    landmarks: np.ndarray
    transform: np.ndarray | None = None

    def pixels(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Landmarks scaled to pixel coordinates, shape (478, 2)."""
        return self.landmarks[:, :2] * np.array([frame_w, frame_h])

    @property
    def right_iris(self) -> np.ndarray:
        return self.landmarks[RIGHT_IRIS_CENTER]

    @property
    def left_iris(self) -> np.ndarray:
        return self.landmarks[LEFT_IRIS_CENTER]


class FaceLandmarker:
    """Synchronous per-frame landmark detector (MediaPipe VIDEO mode)."""

    def __init__(self, model_path: Path | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path or ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceResult | None:
        """Detect landmarks in a BGR frame. Returns None if no face is found."""
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        transform = None
        if result.facial_transformation_matrixes:
            transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
        return FaceResult(
            np.array([[p.x, p.y, p.z] for p in face], dtype=np.float32),
            transform=transform,
        )

    def close(self) -> None:
        self._landmarker.close()
