import numpy as np

from trafo.gaze.calibration import GazeMapper, ScreenLockedMapper
from trafo.gaze.pursuit import PURSUIT_DURATION_S, pair_with_lag, pursuit_path

RECT = (0, 0, 1920, 1080)


def features_for(rng, rect, tx, ty):
    """Synthetic features for an eye looking at global (tx, ty)."""
    fx = (tx - rect[0]) / rect[2] - 0.5
    fy = (ty - rect[1]) / rect[3] - 0.5
    f = np.zeros(15)
    f[0] = -3 - 4 * fy + rng.normal(0, 0.4)
    f[1] = 5 * fx + rng.normal(0, 0.4)
    f[2] = rng.normal(0, 1.0)
    f[3] = 0.07 * fx + rng.normal(0, 0.003)
    f[4] = -0.05 * fy + rng.normal(0, 0.003)
    f[5] = 0.07 * fx + rng.normal(0, 0.003)
    f[6] = -0.05 * fy + rng.normal(0, 0.003)
    f[7] = 0.5 + rng.normal(0, 0.005)
    f[8] = 0.4 + rng.normal(0, 0.005)
    f[9] = 0.14 + rng.normal(0, 0.002)
    f[10] = rng.normal(0, 0.3)
    f[11] = rng.normal(0, 0.3)
    f[12] = -45 + rng.normal(0, 0.5)
    f[13] = rng.normal(0, 0.003)
    f[14] = 1.0 + rng.normal(0, 0.02)
    return f


# -- path geometry ---------------------------------------------------------


def test_path_stays_inside_margins():
    path = pursuit_path(RECT)
    pts = np.stack([path(t) for t in np.arange(0, PURSUIT_DURATION_S, 0.05)])
    x0, y0, w, h = RECT
    eps = 1e-6
    assert pts[:, 0].min() >= x0 + 0.08 * w - eps
    assert pts[:, 0].max() <= x0 + 0.92 * w + eps
    assert pts[:, 1].min() >= y0 + 0.08 * h - eps
    assert pts[:, 1].max() <= y0 + 0.92 * h + eps


def test_path_covers_most_of_the_screen():
    path = pursuit_path(RECT)
    pts = np.stack([path(t) for t in np.arange(0, PURSUIT_DURATION_S, 0.05)])
    assert pts[:, 0].max() - pts[:, 0].min() >= 0.8 * RECT[2]
    assert pts[:, 1].max() - pts[:, 1].min() >= 0.8 * RECT[3]


def test_path_speed_within_smooth_pursuit_range():
    path = pursuit_path(RECT)
    dt = 0.01
    ts = np.arange(0, PURSUIT_DURATION_S - dt, dt)
    speeds = [np.linalg.norm(path(t + dt) - path(t)) / dt for t in ts]
    assert max(speeds) < 700  # px/s; ~14 deg/s on a typical setup


def test_path_clamps_time_to_endpoints():
    path = pursuit_path(RECT)
    np.testing.assert_allclose(path(-1.0), path(0.0))
    np.testing.assert_allclose(path(99.0), path(PURSUIT_DURATION_S))


# -- lag compensation --------------------------------------------------------


def test_pair_with_lag_recovers_true_lag():
    rng = np.random.default_rng(0)
    path = pursuit_path(RECT)
    true_lag = 0.12  # the eye looks where the dot was 120 ms ago
    times = np.arange(1.0, PURSUIT_DURATION_S, 1 / 30)
    feats = [features_for(rng, RECT, *path(t - true_lag)) for t in times]

    lag, targets = pair_with_lag(times, feats, path)
    assert abs(lag - true_lag) <= 0.03

    rms_best = GazeMapper().fit(np.stack(feats), targets)
    rms_zero = GazeMapper().fit(
        np.stack(feats), np.stack([path(t) for t in times])
    )
    assert rms_best < rms_zero  # compensation beats naive pairing


# -- saccade trimming through the full fit ------------------------------------


def test_residual_trim_drops_saccade_frames():
    rng = np.random.default_rng(1)
    path = pursuit_path(RECT)
    times = np.arange(0.0, PURSUIT_DURATION_S, 1 / 30)
    clean, corrupted = [], []
    n_corrupt = 0
    for i, t in enumerate(times):
        target = path(t)
        f = features_for(rng, RECT, *target)
        clean.append((f, target, 0))
        if i % 10 == 5:  # ~10% of frames: catch-up saccade (eye jumped ahead)
            f = f.copy()
            f[3:7] += 0.15
            n_corrupt += 1
        corrupted.append((f, target, 0))

    clean_rms = ScreenLockedMapper().fit(clean, [RECT])[0]["rms"]
    mapper = ScreenLockedMapper()
    stats = mapper.fit(corrupted, [RECT])
    assert stats[0]["dropped"] >= 0.6 * n_corrupt  # most saccades removed
    assert stats[0]["rms"] < clean_rms * 1.3  # fit quality close to clean data
