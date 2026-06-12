import sys

import pytest

if sys.platform != "darwin":  # module imports Quartz
    pytest.skip("macOS-only window manager", allow_module_level=True)

from trafo.winmgr.macos import _covers_a_display

# Mirrors a real 3-display layout (primary + one right + one above).
DISPLAYS = [
    (0.0, 0.0, 1728.0, 1117.0),
    (1728.0, -470.0, 1920.0, 1080.0),
    (-192.0, -1080.0, 1920.0, 1080.0),
]


def test_exact_display_rect_covers():
    assert _covers_a_display((1728, -470, 1920, 1080), DISPLAYS)


def test_near_full_overlay_covers():
    # e.g. fullscreen video that leaves a small inset
    assert _covers_a_display((1728, -440, 1920, 1050), DISPLAYS)


def test_half_screen_window_does_not_cover():
    assert not _covers_a_display((0, 0, 864, 1117), DISPLAYS)


def test_window_spanning_two_displays_partially_does_not_cover():
    # 90% threshold is per-display; a straddling window covers neither enough.
    assert not _covers_a_display((1000, 0, 1500, 900), DISPLAYS)


def test_no_displays_never_covers():
    assert not _covers_a_display((0, 0, 5000, 5000), [])
