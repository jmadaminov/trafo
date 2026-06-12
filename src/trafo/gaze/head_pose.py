"""Head pose from MediaPipe's facial transformation matrix.

MediaPipe's canonical face frame: +x = face's left, +y = up, +z = out of the
face (toward the camera when looking straight at it). The transform maps that
frame into camera space.
"""

from __future__ import annotations

import numpy as np


def rotation(transform: np.ndarray) -> np.ndarray:
    """3x3 head rotation matrix (canonical face -> camera space)."""
    return transform[:3, :3]


def translation(transform: np.ndarray) -> tuple[float, float, float]:
    """(tx, ty, tz): head position in camera space (cm-ish units).

    A metric version of the frame-relative face center — unlike normalized
    frame coordinates, it does not change when the camera resolution or
    aspect ratio does.
    """
    t = transform[:3, 3]
    return float(t[0]), float(t[1]), float(t[2])


def euler_degrees(transform: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees.

    pitch > 0: looking up; yaw > 0: face turned to the subject's left;
    roll > 0: head tilted toward the subject's left shoulder.
    Exact signs only matter for display — the calibration regression is
    indifferent to conventions as long as they are consistent.
    """
    r = rotation(transform)
    # Forward = third column (face +z in camera space).
    fx, fy, fz = r[:, 2]
    yaw = np.degrees(np.arctan2(fx, fz))
    pitch = np.degrees(np.arctan2(fy, np.hypot(fx, fz)))
    # Roll from the up vector (second column) projected onto the camera xy plane.
    ux, uy, _ = r[:, 1]
    roll = np.degrees(np.arctan2(-ux, uy))
    return float(pitch), float(yaw), float(roll)
