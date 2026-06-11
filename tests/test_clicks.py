import numpy as np

from trafo.gaze.clicks import ClickSampleGate, GazeHistoryEntry


def history(pred, n=10, t_end=5.0, hz=30, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = np.asarray(pred, dtype=float) + rng.normal(0, jitter, 2)
        out.append(GazeHistoryEntry(t_end - (n - 1 - i) / hz, np.full(10, 0.1), p))
    return out


def test_gate_accepts_stable_on_target_click():
    gate = ClickSampleGate()
    feats = gate.evaluate((1000, 500), 5.0, history((1020, 480), jitter=10))
    assert feats is not None
    assert feats.shape == (10,)


def test_gate_rejects_unstable_gaze():
    gate = ClickSampleGate()
    h = history((1000, 500), jitter=10)
    h[-1] = GazeHistoryEntry(5.0, np.full(10, 0.1), np.array([1600.0, 500.0]))  # saccade
    assert gate.evaluate((1000, 500), 5.0, h) is None


def test_gate_rejects_click_far_from_gaze():
    gate = ClickSampleGate()
    # Muscle-memory click 800px from where the user is looking.
    assert gate.evaluate((200, 500), 5.0, history((1000, 500), jitter=5)) is None


def test_gate_rejects_when_history_sparse():
    gate = ClickSampleGate()
    assert gate.evaluate((1000, 500), 5.0, history((1000, 500), n=2)) is None
    # Stale history (face was lost): entries fall outside the window.
    assert gate.evaluate((1000, 500), 50.0, history((1000, 500), t_end=5.0)) is None


def test_gate_enforces_cooldown():
    gate = ClickSampleGate()
    assert gate.evaluate((1000, 500), 5.0, history((1000, 500), jitter=5)) is not None
    assert gate.evaluate((1000, 500), 5.3, history((1000, 500), t_end=5.3, jitter=5)) is None
    assert gate.evaluate((1000, 500), 6.5, history((1000, 500), t_end=6.5, jitter=5)) is not None
