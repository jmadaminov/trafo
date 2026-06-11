# Trafo

Gaze-driven window focus: look at a window and Trafo brings it to the front.

Cross-platform (macOS / Windows / Linux-X11), webcam-based — no eye-tracker hardware needed.

## Install (beta)

**macOS app (no terminal needed):** build the bundle once, then double-click:

```sh
uv sync
./scripts/build_macos.sh        # produces dist/Trafo.app
open dist/Trafo.app             # first launch: right-click ▸ Open (unsigned beta)
```

On first run Trafo shows an onboarding screen that walks you through the four
permissions and then runs calibration. It lives in the menu bar; closing the
window keeps it running in the background (Quit from the tray menu).

**From source (development):** requires Python 3.10–3.12 (MediaPipe does not
support 3.13+) and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run trafo app
```

## Permissions (macOS)

Trafo needs four permissions; the onboarding screen checks each one and links
to the right System Settings pane:

| Permission | Why |
|---|---|
| Camera | track your eye gaze |
| Screen Recording | read other apps' window titles and positions |
| Accessibility | raise the window you look at |
| Input Monitoring | learn from your clicks to stay accurate |

After granting in System Settings, quit and reopen if a permission doesn't take effect.

## Testing each milestone

| Milestone | Command | What you should see |
|---|---|---|
| M1 camera | `uv run trafo demo camera` | mirrored live preview with FPS counter; `q`/Esc quits |
| M2 landmarks | `uv run trafo demo landmarks` | green eye contours + red iris dots tracking your eyes |
| M3 gaze | `uv run trafo demo gaze` | head-pose axes on your nose + yellow gaze arrow following where you look |
| M4 app + calibration | `uv run trafo app` | app window; click "Calibrate…", follow the dots on every display |
| M4 gaze dot | `uv run trafo demo dot` | (after calibrating) blue dot follows your gaze across all displays |
| M5 windows | `uv run trafo windows list` / `uv run trafo windows focus <name>` | window table with rects; focus raises the matched window |
| M6 focus engine | `uv run trafo app` → enable "Focus follows gaze" | look at a window ~0.5 s and it comes to the front |

The **Debug view** button (or in the debug build) opens a live camera feed with
landmark overlay, FPS, blink state, locked screen and fixation status — useful
for diagnosing accuracy on a given setup.

## Platform notes

- **Linux:** X11 only. Wayland has no portable API for global window geometry or
  programmatic focus, so it is out of scope.
- **Gaze accuracy:** webcam gaze estimation is good to roughly ±2–5 cm after calibration —
  enough to pick a window, not a pixel. Dwell time + hysteresis prevent focus flapping.
  Camera capture runs at 1080p so the iris gets as many pixels as possible.
- **Stability:** the gaze point is despiked (median prefilter), smoothed (One Euro), and
  frozen during fixations, so the dot sits still when you stare and only moves on a real
  saccade. Blinks (either eye partially closing, plus a short settle hold) are dropped so
  they cannot fling the dot.
- **Multiple displays:** fully supported. Calibration visits every connected display; a
  screen classifier (with lock hysteresis) picks the display you're looking at, and a
  per-display model maps eye movement within it. Recalibrate after plugging or unplugging
  a monitor, or after moving the webcam.
- **Click learning:** you look at what you click, so stable on-target clicks are folded
  into the model as fresh ground truth (suspicious clicks are rejected). Accuracy improves
  with normal use and survives restarts. Toggle in the app window.
