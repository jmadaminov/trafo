import numpy as np

from trafo.gaze.features import is_blinking
from trafo.gaze.landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS_CENTER,
    FaceResult,
)
from trafo.gaze.features import LEFT_EYE_LIDS, RIGHT_EYE_LIDS
from trafo.gaze.stabilizer import GazeStabilizer

HZ = 30


def feed(stab, points, t0=0.0):
    out = []
    for i, p in enumerate(points):
        out.append(stab.update(np.asarray(p, dtype=float), t0 + i / HZ))
    return np.stack(out)


def test_fixation_freezes_noisy_input():
    rng = np.random.default_rng(0)
    stab = GazeStabilizer()
    target = np.array([1000.0, 600.0])
    pts = target + rng.normal(0, 25, size=(90, 2))  # 3 s of noisy fixation
    out = feed(stab, pts)
    tail = out[-30:]
    # The dot must be essentially motionless and near the true target.
    assert np.max(np.linalg.norm(tail - tail[-1], axis=1)) < 1e-6  # frozen
    assert np.linalg.norm(tail[-1] - target) < 25
    assert stab.is_fixating


def test_saccade_followed_quickly():
    rng = np.random.default_rng(1)
    stab = GazeStabilizer()
    a, b = np.array([400.0, 400.0]), np.array([2400.0, 800.0])
    feed(stab, a + rng.normal(0, 10, size=(60, 2)))
    out = feed(stab, b + rng.normal(0, 10, size=(15, 2)), t0=2.0)
    # Within 10 frames (~0.33 s) the output must be close to the new target.
    assert np.linalg.norm(out[9] - b) < 150


def test_single_frame_spike_is_ignored():
    rng = np.random.default_rng(2)
    stab = GazeStabilizer()
    target = np.array([800.0, 500.0])
    feed(stab, target + rng.normal(0, 8, size=(60, 2)))
    before = stab.update(target, 2.0)
    spiked = stab.update(target + np.array([1800.0, -900.0]), 2.0 + 1 / HZ)  # glitch frame
    assert np.linalg.norm(spiked - before) < 30


def _face_with_openness(right_open: float, left_open: float) -> FaceResult:
    lm = np.zeros((478, 3), dtype=np.float32)
    width = 0.10
    for corners, lids, iris, cx in (
        (RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS_CENTER, 0.35),
        (LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS_CENTER, 0.65),
    ):
        open_frac = right_open if cx < 0.5 else left_open
        lm[corners[0], :2] = (cx - width / 2, 0.5)
        lm[corners[1], :2] = (cx + width / 2, 0.5)
        gap = open_frac * width
        lm[lids[0], :2] = (cx, 0.5 - gap / 2)
        lm[lids[1], :2] = (cx, 0.5 + gap / 2)
        lm[iris, :2] = (cx, 0.5)
    return FaceResult(lm)


def test_blink_detected_when_either_eye_partially_closed():
    assert not is_blinking(_face_with_openness(0.30, 0.30))
    assert is_blinking(_face_with_openness(0.05, 0.05))  # full blink
    assert is_blinking(_face_with_openness(0.10, 0.30))  # one lid drooping
    assert is_blinking(_face_with_openness(0.30, 0.12))  # the other one
