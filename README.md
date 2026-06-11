# Trafo

Gaze-driven window focus: look at a window and Trafo brings it to the front.

Cross-platform (macOS / Windows / Linux-X11), webcam-based — no eye-tracker hardware needed.

## Install (macOS beta)

### 1. Get the app

Build it once (requires [uv](https://docs.astral.sh/uv/) and Python 3.10–3.12):

```sh
uv sync
./scripts/build_macos.sh
```

This produces two things in `dist/`:

- **`Trafo-0.1.0.dmg`** — open it and **drag Trafo into Applications** (the
  disk image has the usual drag-to-install layout). Share this file to install
  Trafo on another Mac.
- **`Trafo.app`** — the raw bundle, if you prefer to copy it yourself, or run
  `./scripts/build_macos.sh --install` to have the script copy it into
  /Applications for you.

If you launch Trafo from somewhere other than /Applications, it offers to move
itself there — accept; macOS remembers permission grants more reliably for
installed apps.

### 2. First launch

The beta is not notarized, so the very first time: **right-click
Trafo.app ▸ Open ▸ Open**. (A normal double-click shows a warning with no
Open button.)

Trafo lives in the **menu bar** (look for the eye icon). Closing its window
keeps it running in the background — quit from the menu-bar icon.

### 3. Permissions

On first run, the onboarding screen walks you through four permissions. Each
one has a single job:

| Permission | What Trafo uses it for |
|---|---|
| Camera | seeing your eyes to track gaze |
| Screen Recording | reading other apps' window titles and positions |
| Accessibility | bringing the window you look at to the front |
| Input Monitoring | learning from your clicks to stay accurate |

The flow is **Grant → Restart Trafo → Calibrate**. The restart matters: macOS
applies permission grants only to a freshly launched app, and onboarding has a
**Restart Trafo** button for exactly this.

### 4. Calibrate

After the restart, click **Continue to calibration** and follow the dots on
every display. That's it — enable **Focus follows gaze** in the main window
and look around.

## Troubleshooting (macOS)

**System Settings shows a permission as granted, but Trafo says "Needs
access".** The grant belongs to an older build: the beta is ad-hoc signed, and
each rebuild looks like a brand-new app to macOS. In that Settings list,
remove Trafo (select it, click **−**), add the current app with **+**, toggle
it on, then restart Trafo. If things are really tangled, reset all of Trafo's
permission records and grant fresh:

```sh
tccutil reset All com.trafo.app
```

**"Camera unavailable" in the Status card.** Another app may be holding the
camera, or Camera permission isn't granted yet. Fix the cause, then click
**Retry camera** — no app restart needed.

**Granting had no effect.** Permissions only apply after the app restarts —
use the **Restart Trafo** button in onboarding, or quit from the menu bar and
reopen.

**Beta signing caveat — make grants survive rebuilds.** One-time setup: open
**Keychain Access ▸ Certificate Assistant ▸ Create a Certificate…**, name it
`trafo-dev`, set Certificate Type to **Code Signing**, click Create. Then
build with:

```sh
TRAFO_CODESIGN_IDENTITY=trafo-dev ./scripts/build_macos.sh --install
```

You re-grant once for the new identity; after that, permission grants persist
across rebuilds.

## Run from source (development)

Requires Python 3.10–3.12 (MediaPipe does not support 3.13+) and
[uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run trafo app
```

## Testing each milestone

| Milestone | Command | What you should see |
|---|---|---|
| M1 camera | `uv run trafo demo camera` | mirrored live preview with FPS counter; `q`/Esc quits |
| M2 landmarks | `uv run trafo demo landmarks` | green eye contours + red iris dots tracking your eyes |
| M3 gaze | `uv run trafo demo gaze` | head-pose axes on your nose + yellow gaze arrow following where you look |
| M4 app + calibration | `uv run trafo app` | app window; click "Calibrate", follow the dots on every display |
| M4 gaze dot | `uv run trafo demo dot` | (after calibrating) blue dot follows your gaze across all displays |
| M5 windows | `uv run trafo windows list` / `uv run trafo windows focus <name>` | window table with rects; focus raises the matched window |
| M6 focus engine | `uv run trafo app` → enable "Focus follows gaze" | look at a window ~0.5 s and it comes to the front |

The **Debug view** button opens a live camera feed with landmark overlay, FPS,
blink state, locked screen and fixation status — useful for diagnosing
accuracy on a given setup.

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
