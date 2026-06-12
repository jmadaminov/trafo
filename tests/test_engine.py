import numpy as np

from trafo.engine import EngineConfig, FocusEngine
from trafo.winmgr.base import WindowInfo, WindowManager

A = WindowInfo(id=1, title="A", app="AppA", pid=11, rect=(0, 0, 500, 500))
B = WindowInfo(id=2, title="B", app="AppB", pid=22, rect=(500, 0, 500, 500))
C = WindowInfo(id=3, title="C", app="AppC", pid=33, rect=(0, 500, 1000, 300))


class FakeWM(WindowManager):
    def __init__(self, windows, focused=None):
        self.windows = list(windows)
        self.focused = focused if focused is not None else self.windows[0]
        self.focus_calls = []

    def list_windows(self):
        return list(self.windows)

    def focused_window(self):
        return self.focused

    def focus(self, window):
        self.focus_calls.append(window.id)
        self.windows.remove(window)
        self.windows.insert(0, window)  # focusing raises it to the front
        self.focused = window
        return True


def make_engine(windows=(A, B, C)):
    wm = FakeWM(windows)
    return FocusEngine(wm, EngineConfig()), wm


def run_gaze(engine, x, y, t0, t1, hz=30):
    """Feed a steady gaze point over [t0, t1); returns windows focused."""
    focused = []
    t = t0
    while t < t1:
        result = engine.update(x, y, t)
        if result:
            focused.append(result)
        t += 1 / hz
    return focused


def test_dwell_focuses_after_delay():
    engine, wm = make_engine()
    focused = run_gaze(engine, 700, 100, 0.0, 1.0)  # inside B
    assert [w.id for w in focused] == [B.id]
    assert wm.focus_calls == [B.id]


def test_no_focus_before_dwell():
    engine, wm = make_engine()
    assert run_gaze(engine, 700, 100, 0.0, 0.4) == []
    assert wm.focus_calls == []


def test_brief_flicker_does_not_reset_dwell():
    engine, wm = make_engine()
    run_gaze(engine, 700, 100, 0.0, 0.3)  # dwell on B
    run_gaze(engine, 100, 100, 0.3, 0.39)  # flicker into A, under tolerance
    focused = run_gaze(engine, 700, 100, 0.39, 0.8)  # back to B
    assert [w.id for w in focused] == [B.id]
    assert wm.focus_calls == [B.id]


def test_sustained_switch_restarts_dwell():
    engine, wm = make_engine()
    run_gaze(engine, 700, 100, 0.0, 0.3)  # dwell on B, never completes
    focused = run_gaze(engine, 200, 600, 0.3, 1.2)  # move to C and stay
    assert [w.id for w in focused] == [C.id]
    assert wm.focus_calls == [C.id]  # B was never focused


def test_frontmost_window_not_refocused():
    engine, wm = make_engine()
    focused = run_gaze(engine, 100, 100, 0.0, 2.0)  # A is already frontmost
    assert focused == []
    assert wm.focus_calls == []


def test_cooldown_blocks_immediate_second_switch():
    engine, wm = make_engine()
    run_gaze(engine, 700, 100, 0.0, 0.7)  # focus B (completes ~0.5)
    focused = run_gaze(engine, 200, 600, 0.7, 1.4)  # stare at C during cooldown
    assert focused == []  # cooldown until ~1.5
    focused = run_gaze(engine, 200, 600, 1.4, 2.5)
    assert [w.id for w in focused] == [C.id]


def test_gaze_on_empty_space_does_nothing():
    engine, wm = make_engine(windows=(A,))
    focused = run_gaze(engine, 900, 900, 0.0, 2.0)  # outside every window
    assert focused == []
    assert wm.focus_calls == []


# Small window stacked above a fullscreen window — the reported bug scenarios.
BIG = WindowInfo(id=4, title="Big", app="Editor", pid=44, rect=(0, 0, 1920, 1080))
SMALL = WindowInfo(id=5, title="Small", app="Chat", pid=55, rect=(600, 300, 400, 300))


def test_gaze_spill_around_focused_small_window_does_not_raise_big_behind():
    """Reading in a small focused window: gaze noise just outside its edge
    lands on the fullscreen window behind — that must not raise it."""
    wm = FakeWM([SMALL, BIG], focused=SMALL)
    engine = FocusEngine(wm, EngineConfig())
    focused = run_gaze(engine, 560, 450, 0.0, 3.0)  # 40px left of SMALL, inside BIG
    assert focused == []
    assert wm.focus_calls == []


def test_unfocused_window_on_top_of_focused_fullscreen_gets_focused():
    """Fullscreen window has focus; a small window floats above it in z-order.
    Looking at the small window must focus it — not be mistaken for 'already
    frontmost' (z-order is not focus)."""
    wm = FakeWM([SMALL, BIG], focused=BIG)
    engine = FocusEngine(wm, EngineConfig())
    focused = run_gaze(engine, 700, 400, 0.0, 1.0)  # inside SMALL
    assert [w.id for w in focused] == [SMALL.id]
    assert wm.focus_calls == [SMALL.id]


def test_near_miss_resolves_to_window_on_top():
    """A hit a few px outside the small top window (but inside the fullscreen
    one behind) counts toward the small window — small targets win against
    calibration error."""
    wm = FakeWM([SMALL, BIG], focused=BIG)
    engine = FocusEngine(wm, EngineConfig())
    focused = run_gaze(engine, 560, 450, 0.0, 1.0)  # 40px outside SMALL, inside BIG
    assert [w.id for w in focused] == [SMALL.id]


def test_own_window_blocks_hits_but_is_never_focused():
    own = WindowInfo(id=9, title="Trafo", app="Trafo", pid=99, rect=(100, 100, 300, 300), own=True)
    wm = FakeWM([own, BIG], focused=BIG)
    engine = FocusEngine(wm, EngineConfig())
    focused = run_gaze(engine, 200, 200, 0.0, 2.0)  # inside our own window
    assert focused == []
    assert wm.focus_calls == []


# -- mouse outranks gaze -------------------------------------------------------


def test_recent_mouse_activity_suspends_gaze_focus():
    engine, wm = make_engine()
    engine.note_mouse_activity(0.0)
    focused = run_gaze(engine, 700, 100, 0.0, 4.0)  # stare at B for 4s < pause 5s
    assert focused == []
    assert wm.focus_calls == []


def test_gaze_focus_resumes_after_mouse_pause():
    engine, wm = make_engine()
    engine.note_mouse_activity(0.0)
    focused = run_gaze(engine, 700, 100, 0.0, 6.0)  # pause ends at 5.0, dwell 0.5
    assert [w.id for w in focused] == [B.id]
    assert wm.focus_calls == [B.id]


def test_mouse_activity_drops_accumulated_dwell():
    engine, wm = make_engine()
    cfg = engine.cfg
    cfg.mouse_pause_s = 1.0
    run_gaze(engine, 700, 100, 0.0, 0.4)  # dwell on B almost complete
    engine.note_mouse_activity(0.4)
    # Pause [0.4, 1.4); dwell must restart, so nothing fires before ~1.9.
    focused = run_gaze(engine, 700, 100, 0.4, 1.85)
    assert focused == []
    focused = run_gaze(engine, 700, 100, 1.85, 2.2)
    assert [w.id for w in focused] == [B.id]


def test_zero_mouse_pause_disables_suspension():
    engine, wm = make_engine()
    engine.cfg.mouse_pause_s = 0.0
    engine.note_mouse_activity(0.0)
    focused = run_gaze(engine, 700, 100, 0.0, 1.0)
    assert [w.id for w in focused] == [B.id]


# -- keyboard outranks gaze ----------------------------------------------------


def test_recent_typing_suspends_gaze_focus():
    engine, wm = make_engine()
    engine.note_keyboard_activity(0.0)
    focused = run_gaze(engine, 700, 100, 0.0, 4.0)  # stare at B for 4s < pause 5s
    assert focused == []
    assert wm.focus_calls == []


def test_gaze_focus_resumes_after_keyboard_pause():
    engine, wm = make_engine()
    engine.note_keyboard_activity(0.0)
    focused = run_gaze(engine, 700, 100, 0.0, 6.0)  # pause ends at 5.0, dwell 0.5
    assert [w.id for w in focused] == [B.id]


def test_typing_drops_accumulated_dwell():
    engine, wm = make_engine()
    engine.cfg.keyboard_pause_s = 1.0
    run_gaze(engine, 700, 100, 0.0, 0.4)  # dwell on B almost complete
    engine.note_keyboard_activity(0.4)
    focused = run_gaze(engine, 700, 100, 0.4, 1.85)  # pause + restarted dwell
    assert focused == []
    focused = run_gaze(engine, 700, 100, 1.85, 2.2)
    assert [w.id for w in focused] == [B.id]


def test_zero_keyboard_pause_disables_suspension():
    engine, wm = make_engine()
    engine.cfg.keyboard_pause_s = 0.0
    engine.note_keyboard_activity(0.0)
    focused = run_gaze(engine, 700, 100, 0.0, 1.0)
    assert [w.id for w in focused] == [B.id]


# -- per-app exclusion rules ---------------------------------------------------


def test_excluded_app_is_never_focused():
    engine, wm = make_engine()
    engine.cfg.excluded_apps = frozenset({"appb"})
    focused = run_gaze(engine, 700, 100, 0.0, 2.0)  # stare at B (app "AppB")
    assert focused == []
    assert wm.focus_calls == []


def test_excluded_app_blocks_window_behind_it():
    """Looking at an excluded app must not raise the window underneath."""
    wm = FakeWM([SMALL, BIG], focused=BIG)
    engine = FocusEngine(wm, EngineConfig(excluded_apps=frozenset({"chat"})))
    focused = run_gaze(engine, 700, 400, 0.0, 2.0)  # inside SMALL (app "Chat")
    assert focused == []
    assert wm.focus_calls == []  # BIG behind it must not be raised either


def test_exclusion_match_is_case_insensitive():
    engine, wm = make_engine()
    engine.cfg.excluded_apps = frozenset({"appb"})  # stored lowercase
    assert run_gaze(engine, 700, 100, 0.0, 2.0) == []  # WindowInfo.app == "AppB"
    engine.cfg.excluded_apps = frozenset()
    assert [w.id for w in run_gaze(engine, 700, 100, 2.0, 4.0)] == [B.id]
