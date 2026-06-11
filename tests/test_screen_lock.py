import numpy as np

from trafo.gaze.calibration import GazeMapper, ScreenLockedMapper

S0 = (0, 0, 1920, 1080)
S1 = (1920, -200, 2560, 1440)
RECTS = [S0, S1]

# Head-pose regime per screen: looking at S1 means yaw is rotated ~18 deg.
SCREEN_POSE = {0: (0.0, -3.0), 1: (18.0, -1.0)}  # yaw, pitch


def sample_for(rng, screen, tx, ty, n=1):
    """Synthetic features for gazing at global (tx, ty) on `screen`."""
    rect = RECTS[screen]
    fx = (tx - rect[0]) / rect[2] - 0.5  # -0.5..0.5 within screen
    fy = (ty - rect[1]) / rect[3] - 0.5
    yaw0, pitch0 = SCREEN_POSE[screen]
    out = np.zeros((n, 10))
    out[:, 0] = pitch0 - 4 * fy + rng.normal(0, 0.8, n)
    out[:, 1] = yaw0 + 5 * fx + rng.normal(0, 0.8, n)
    out[:, 2] = rng.normal(0, 1.5, n)  # roll
    out[:, 3] = 0.07 * fx + rng.normal(0, 0.004, n)  # r_iris_dx
    out[:, 4] = -0.05 * fy + rng.normal(0, 0.004, n)  # r_iris_dy
    out[:, 5] = 0.07 * fx + rng.normal(0, 0.004, n)
    out[:, 6] = -0.05 * fy + rng.normal(0, 0.004, n)
    out[:, 7] = 0.5 + rng.normal(0, 0.01, n)
    out[:, 8] = 0.4 + rng.normal(0, 0.01, n)
    out[:, 9] = 0.14 + rng.normal(0, 0.003, n)
    return out


def calibration_samples(rng, samples_per_point=30):
    samples = []
    for si, rect in enumerate(RECTS):
        for gy in (0.08, 0.5, 0.92):
            for gx in (0.08, 0.5, 0.92):
                tx, ty = rect[0] + gx * rect[2], rect[1] + gy * rect[3]
                for f in sample_for(rng, si, tx, ty, samples_per_point):
                    samples.append((f, np.array([tx, ty]), si))
    return samples


def fitted(rng=None):
    rng = rng or np.random.default_rng(0)
    mapper = ScreenLockedMapper()
    stats = mapper.fit(calibration_samples(rng), RECTS)
    return mapper, stats


def test_fit_and_classify_screens():
    mapper, stats = fitted()
    assert all(st["rms"] < 200 for st in stats.values())
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(100):
        si = rng.integers(0, 2)
        rect = RECTS[si]
        tx = rect[0] + rng.uniform(0.1, 0.9) * rect[2]
        ty = rect[1] + rng.uniform(0.1, 0.9) * rect[3]
        f = sample_for(rng, si, tx, ty)[0]
        hits += mapper.classify(f) == si
    assert hits >= 95


def test_per_screen_model_beats_global_model():
    rng = np.random.default_rng(0)
    samples = calibration_samples(rng)
    mapper, _ = fitted()

    global_mapper = GazeMapper()
    x = np.stack([f for f, _, _ in samples])
    y = np.stack([t for _, t, _ in samples])
    global_mapper.fit(x, y)

    # Evaluate per screen in blocks (settling the lock first), the way real
    # gaze behaves — the lock hysteresis is deliberately slow to flip, so
    # alternating screens on every sample would be measuring the wrong thing.
    rng = np.random.default_rng(7)
    err_locked, err_global = [], []
    for si in (0, 1):
        rect = RECTS[si]
        for _ in range(20):  # settle the lock on this screen
            mapper.predict(sample_for(rng, si, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)[0])
        for _ in range(100):
            tx = rect[0] + rng.uniform(0.1, 0.9) * rect[2]
            ty = rect[1] + rng.uniform(0.1, 0.9) * rect[3]
            f = sample_for(rng, si, tx, ty)[0]
            err_locked.append(np.linalg.norm(mapper.predict(f) - (tx, ty)))
            err_global.append(np.linalg.norm(global_mapper.predict(f) - (tx, ty)))
    assert np.mean(err_locked) < np.mean(err_global)


def test_lock_hysteresis_ignores_brief_glances():
    mapper, _ = fitted()
    rng = np.random.default_rng(3)
    for _ in range(20):  # settle the lock on screen 0
        mapper.predict(sample_for(rng, 0, 960, 540)[0])
    assert mapper.locked_screen == 0

    for _ in range(mapper.lock_n - 1):  # brief glance at screen 1, below threshold
        p = mapper.predict(sample_for(rng, 1, 3000, 500)[0])
        assert S0[0] <= p[0] < S0[0] + S0[2]  # still clamped to screen 0
    assert mapper.locked_screen == 0

    for _ in range(mapper.lock_n + 2):  # sustained look switches the lock
        mapper.predict(sample_for(rng, 1, 3000, 500)[0])
    assert mapper.locked_screen == 1


def test_prediction_clamped_to_locked_screen():
    mapper, _ = fitted()
    rng = np.random.default_rng(4)
    for _ in range(20):
        mapper.predict(sample_for(rng, 1, 3200, 500)[0])
    p = mapper.predict(sample_for(rng, 1, 3200, 500)[0])
    assert S1[0] <= p[0] < S1[0] + S1[2]
    assert S1[1] <= p[1] < S1[1] + S1[3]


def test_save_load_roundtrip(tmp_path):
    mapper, _ = fitted()
    path = tmp_path / "cal3.json"
    mapper.save(path)
    loaded = ScreenLockedMapper.load(path)
    assert loaded is not None and len(loaded.mappers) == 2
    rng = np.random.default_rng(5)
    f = sample_for(rng, 0, 800, 400)[0]
    for _ in range(10):
        a, b = mapper.predict(f), loaded.predict(f)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_click_samples_correct_drift():
    """Simulate posture drift after calibration: iris features shift by a
    constant. Clicks at known positions must pull predictions back."""
    mapper, _ = fitted()
    rng = np.random.default_rng(8)
    drift = np.zeros(10)
    drift[3:7] = 0.012  # post-calibration posture change

    def drifted(screen, tx, ty):
        return sample_for(rng, screen, tx, ty)[0] + drift

    # Settle lock on screen 0 and measure the drift-induced error.
    for _ in range(20):
        mapper.predict(drifted(0, 960, 540))
    before = np.mean(
        [np.linalg.norm(mapper.predict(drifted(0, 960, 540)) - (960, 540)) for _ in range(20)]
    )

    # The user clicks around screen 0 for a while (ground truth at click).
    for _ in range(40):
        tx, ty = rng.uniform(200, 1700), rng.uniform(150, 950)
        mapper.add_click_sample(drifted(0, tx, ty), (tx, ty))

    after = np.mean(
        [np.linalg.norm(mapper.predict(drifted(0, 960, 540)) - (960, 540)) for _ in range(20)]
    )
    assert after < before * 0.6  # most of the drift error is gone
    assert mapper.click_count == 40


def test_click_sample_roundtrip_through_save(tmp_path):
    mapper, _ = fitted()
    rng = np.random.default_rng(9)
    for _ in range(10):
        tx, ty = rng.uniform(200, 1700), rng.uniform(150, 950)
        mapper.add_click_sample(sample_for(rng, 0, tx, ty)[0], (tx, ty))
    path = tmp_path / "cal4.json"
    mapper.save(path)
    loaded = ScreenLockedMapper.load(path)
    assert loaded is not None
    assert loaded.click_count == 10
    # Further click learning keeps working after a reload.
    tx, ty = 800.0, 600.0
    for _ in range(10):
        loaded.add_click_sample(sample_for(rng, 0, tx, ty)[0], (tx, ty))
    assert loaded.click_count == 20


def test_outliers_are_dropped():
    rng = np.random.default_rng(6)
    samples = calibration_samples(rng)
    # Corrupt a handful of samples as if mid-saccade (iris way off target).
    for i in range(0, 60, 4):
        f, t, s = samples[i]
        f = f.copy()
        f[3:7] += 0.25
        samples[i] = (f, t, s)
    mapper = ScreenLockedMapper()
    stats = mapper.fit(samples, RECTS)
    assert sum(st["dropped"] for st in stats.values()) >= 10