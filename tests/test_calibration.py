import numpy as np

from trafo.gaze.calibration import GazeMapper
from trafo.gaze.smoothing import OneEuroFilter


def synthetic_data(n=300, seed=0):
    """Realistic feature magnitudes; gaze driven mainly by the iris offsets."""
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            rng.normal(0, 8, n),  # pitch (deg)
            rng.normal(0, 8, n),  # yaw (deg)
            rng.normal(0, 3, n),  # roll (deg)
            rng.normal(0, 0.04, n),  # r_iris_dx
            rng.normal(0, 0.04, n),  # r_iris_dy
            rng.normal(0, 0.04, n),  # l_iris_dx
            rng.normal(0, 0.04, n),  # l_iris_dy
            rng.normal(0.5, 0.05, n),  # face_cx
            rng.normal(0.4, 0.05, n),  # face_cy
            rng.normal(0.14, 0.01, n),  # eye_dist
            rng.normal(0, 2, n),  # head_tx
            rng.normal(0, 2, n),  # head_ty
            rng.normal(-45, 3, n),  # head_tz
            rng.normal(0, 0.02, n),  # cheek_dz
            rng.normal(1.0, 0.1, n),  # oval_ratio
        ]
    )
    gx = 800 + 9000 * x[:, 3] + 8000 * x[:, 5] + 25 * x[:, 1]
    gy = 500 - 7000 * x[:, 4] - 6000 * x[:, 6] - 20 * x[:, 0]
    y = np.column_stack([gx, gy]) + rng.normal(scale=5.0, size=(n, 2))
    return x, y


def test_mapper_fits_synthetic_gaze():
    x, y = synthetic_data()
    mapper = GazeMapper()
    rms = mapper.fit(x, y)
    # Well above the 5px noise floor: here the head-pose terms are independent
    # signal and the mapper deliberately shrinks them (robustness over training
    # fit). In real data head pose is collinear with iris, so the cost is far
    # smaller. The bound just catches a broken fit (unfit data is ~600+ px).
    assert rms < 120

    pred = mapper.predict(x[0])
    assert pred.shape == (2,)
    assert np.linalg.norm(pred - y[0]) < 250


def test_mapper_save_load_roundtrip(tmp_path):
    x, y = synthetic_data()
    mapper = GazeMapper()
    mapper.fit(x, y)
    path = tmp_path / "cal.json"
    mapper.save(path)

    loaded = GazeMapper.load(path)
    assert loaded is not None
    np.testing.assert_allclose(loaded.predict(x[3]), mapper.predict(x[3]), rtol=1e-6)


def test_mapper_load_missing_returns_none(tmp_path):
    assert GazeMapper.load(tmp_path / "nope.json") is None


def test_posture_shift_does_not_explode_prediction():
    """Head held still during calibration must not make head features toxic later.

    Regression test for the drift bug: with data-driven std scaling, the tiny
    head-pose variance of a held-still calibration amplified later posture
    changes into huge screen offsets.
    """
    rng = np.random.default_rng(2)
    n = 400
    x = np.column_stack(
        [
            rng.normal(0, 0.8, n),  # pitch nearly constant (head held still)
            rng.normal(0, 0.8, n),  # yaw nearly constant
            rng.normal(0, 0.3, n),  # roll nearly constant
            rng.normal(0, 0.04, n),
            rng.normal(0, 0.04, n),
            rng.normal(0, 0.04, n),
            rng.normal(0, 0.04, n),
            rng.normal(0.5, 0.01, n),  # face barely moves
            rng.normal(0.4, 0.01, n),
            rng.normal(0.14, 0.003, n),
            rng.normal(0, 0.4, n),  # head position nearly constant
            rng.normal(0, 0.4, n),
            rng.normal(-45, 0.6, n),
            rng.normal(0, 0.005, n),  # asymmetry nearly constant
            rng.normal(1.0, 0.02, n),
        ]
    )
    gx = 800 + 9000 * x[:, 3] + 8000 * x[:, 5]
    gy = 500 - 7000 * x[:, 4] - 6000 * x[:, 6]
    y = np.column_stack([gx, gy]) + rng.normal(scale=5.0, size=(n, 2))

    mapper = GazeMapper()
    mapper.fit(x, y)

    f = x[0].copy()
    before = mapper.predict(f)
    # The user slouches: pitch drops 6 deg, face shifts 10% of the frame.
    f[0] -= 6.0
    f[7] += 0.10
    f[8] += 0.05
    after = mapper.predict(f)
    assert np.linalg.norm(after - before) < 250  # an annoyance, not a different monitor


def test_one_euro_converges_and_smooths():
    f = OneEuroFilter()
    rng = np.random.default_rng(1)
    target = np.array([500.0, 300.0])
    out = None
    for i in range(120):  # 4 s at 30 Hz of noisy stationary input
        noisy = target + rng.normal(scale=15.0, size=2)
        out = f.filter(noisy, t=i / 30)
    assert np.linalg.norm(out - target) < 12  # jitter heavily damped

    # A large jump (saccade) must be followed quickly, within ~5 frames.
    jump = np.array([1500.0, 900.0])
    for i in range(120, 125):
        out = f.filter(jump, t=i / 30)
    assert np.linalg.norm(out - jump) < 150
