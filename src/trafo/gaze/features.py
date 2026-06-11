"""Feature vector for gaze regression: head pose + normalized iris offsets.

Iris offsets are measured relative to the eye-corner midpoint and normalized by
eye width, which makes them invariant to face distance and (mostly) to head
translation. Head pose supplies the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import head_pose
from .landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS_CENTER,
    FaceResult,
)

# Eyelid landmarks (top, bottom) for vertical normalization and blink detection.
RIGHT_EYE_LIDS = (159, 145)
LEFT_EYE_LIDS = (386, 374)

FEATURE_NAMES = [
    "pitch", "yaw", "roll",
    "r_iris_dx", "r_iris_dy", "l_iris_dx", "l_iris_dy",
    "face_cx", "face_cy", "eye_dist",
]


@dataclass
class EyeState:
    iris_offset: np.ndarray  # (dx, dy) normalized by eye width
    openness: float  # lid gap / eye width; ~0 when blinking


def _eye_state(lm: np.ndarray, corners: tuple[int, int], lids: tuple[int, int], iris_idx: int) -> EyeState:
    outer, inner = lm[corners[0], :2], lm[corners[1], :2]
    center = (outer + inner) / 2
    width = np.linalg.norm(outer - inner)
    if width < 1e-6:
        return EyeState(np.zeros(2), 0.0)
    offset = (lm[iris_idx, :2] - center) / width
    gap = np.linalg.norm(lm[lids[0], :2] - lm[lids[1], :2])
    return EyeState(offset, float(gap / width))


def eye_states(face: FaceResult) -> tuple[EyeState, EyeState]:
    """(right, left) eye states."""
    lm = face.landmarks
    return (
        _eye_state(lm, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS_CENTER),
        _eye_state(lm, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS_CENTER),
    )


def is_blinking(face: FaceResult, threshold: float = 0.18) -> bool:
    """True when EITHER eye is even partially closed.

    Iris landmarks corrupt as soon as a lid starts covering the iris — well
    before the eye is fully shut — and a single bad eye is enough to throw
    the gaze mapping, so this is deliberately an OR with a high threshold.
    """
    right, left = eye_states(face)
    return right.openness < threshold or left.openness < threshold


def extract(face: FaceResult) -> np.ndarray | None:
    """Feature vector (len 10, see FEATURE_NAMES), or None if head pose is missing.

    During blinks the iris landmarks are garbage; callers should skip those
    frames via is_blinking().
    """
    if face.transform is None:
        return None
    pitch, yaw, roll = head_pose.euler_degrees(face.transform)
    right, left = eye_states(face)
    lm = face.landmarks
    r_outer, l_outer = lm[RIGHT_EYE_CORNERS[0], :2], lm[LEFT_EYE_CORNERS[0], :2]
    face_center = (r_outer + l_outer) / 2  # normalized frame coords
    eye_dist = np.linalg.norm(r_outer - l_outer)  # proxy for distance to camera
    return np.array(
        [
            pitch, yaw, roll,
            right.iris_offset[0], right.iris_offset[1],
            left.iris_offset[0], left.iris_offset[1],
            face_center[0], face_center[1], eye_dist,
        ],
        dtype=np.float32,
    )
