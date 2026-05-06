# Robo 159 Plotter
izzie
Two-link servo plotter project with:
- live hardware control through Arduino + PCA9685
- a matplotlib simulation mode with the same motion pipeline
- interactive calibration capture and calibration plotting tools
- image/SVG to stroke conversion and tracing utilities

The main runtime script is `final/plotter_v2.py`.

## What This Project Does

This repository controls a 2R (two rotary joints) pen plotter. It solves inverse kinematics for `(x, y)` targets, maps joint angles to PWM pulses via calibrated polynomials, and traces geometric patterns or image-derived strokes.

The code supports two execution modes:
- **Hardware mode**: talks to real servos via Telemetrix and PCA9685.
- **Simulation mode** (`PLOTTER_SIM=1`): replaces hardware with a live matplotlib animation while keeping command flow, IK, calibration usage, and tracing behavior aligned with real mode.

## Repository Layout

- `final/plotter_v2.py` — primary control app (keyboard control + sim UI integration)
- `final/sim_driver.py` — simulation backend (PWM interception, FK rendering, page draw state)
- `final/kinematics.py` — IK and reachable workspace / drawing-box search
- `final/polyline.py` — polyline resampling and cleanup helpers
- `final/svg_io.py` — light SVG parse/write helpers for polylines/strokes
- `final/calibration_tool.py` — reusable interactive calibration capture loop used by `plotter_v2.py`
- `final/plot_calibration.py` — plots servo calibration points and linear/cubic fit comparisons
- `final/plotter_v3.py`, `final/plotter.py` — older/alternate control variants
- `test/linedraw.py` — raster image to vector polylines (contours/hatching)
- `test/svg_to_trajectory.py` — convert polylines to angle trajectories
- `test/capture_calibration.py` — standalone calibration capture utility
- `test/move_to_xy.py` — test movement script with calibration loading
- `images/` — source images, generated SVGs, and trajectory artifacts
- `data/calibration.json` — example calibration output data

## Python Environment

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install core dependencies used across scripts:

```bash
pip install numpy matplotlib pillow
```

Optional / mode-specific dependencies:
- `telemetrix` and `telemetrix_pca9685` for hardware mode.
- `opencv-python` for enhanced edge detection in `test/linedraw.py` (falls back when OpenCV is missing).

## Hardware Assumptions

Expected setup (from current code defaults):
- Arduino running Telemetrix-compatible firmware
- PCA9685 at I2C address `0x40`
- Servo channels:
  - shoulder: channel `0`
  - elbow: channel `1`
  - pen lift: channel `2`

Main runtime constants (edit in `final/plotter_v2.py` for your rig):
- link lengths: `L1`, `L2`
- servo limits: `SERVO0_MIN_DEG`, `SERVO0_MAX_DEG`, `SERVO1_MIN_DEG`, `SERVO1_MAX_DEG`
- pen pulses: `PEN_UP_PULSE`, `PEN_DOWN_PULSE`
- calibration tables: `SERVO0_CAL`, `SERVO1_CAL`
- hysteresis compensation: `HYSTERESIS`

## Quick Start

### 1) Run in simulation mode (safe first run)

```bash
PLOTTER_SIM=1 python final/plotter_v2.py
```

This opens a matplotlib-based arm/page view and keeps terminal keyboard controls active.

### 2) Run on real hardware

```bash
python final/plotter_v2.py
```

On startup, the script:
- initializes board + PCA (or sim stand-ins)
- homes joints and lifts pen
- computes/prints a reachable drawing box (if `USE_AUTO_BOX=True`)
- prints available keyboard commands

## Runtime Controls (`plotter_v2.py`)

Terminal key controls:
- `r` home
- `x` test step
- `s` draw square (standard motion)
- `f` draw square (dynamics motion)
- `v` compare square standard vs dynamics
- `t` draw star
- `g` prompt and move to `(x, y)` in cm
- `i` trace selected image (standard)
- `o` trace selected image (dynamics)
- `p` compare image trace standard vs dynamics
- `b` trace drawing-box outline
- `u` pen up
- `d` pen down
- `c` calibration tool
- `e` erase simulated page (sim mode)
- `q` quit

## Calibration Workflow

Calibration quality has the biggest impact on accuracy. Typical flow:

1. Start `plotter_v2.py` (sim or hardware).
2. Press `c` to enter interactive calibration mode.
3. Nudge servo pulse values using provided keys.
4. Record known physical angles.
5. Copy generated `SERVO0_CAL` / `SERVO1_CAL` tables back into `final/plotter_v2.py`.
6. Re-run and refine.

The capture logic lives in `final/calibration_tool.py` and is called from `plotter_v2.py`.

### Visualize calibration fit quality

Generate plots:

```bash
python final/plot_calibration.py
```

Optional:

```bash
python final/plot_calibration.py --show
python final/plot_calibration.py --out-dir final/cal_plots
```

## Image and SVG Tracing

`plotter_v2.py` can trace image-derived strokes inside the configured drawing box.

Internally it uses:
- `test/linedraw.py` for vectorization from raster images
- `final/polyline.py` for cleanup/resampling
- `final/svg_io.py` for writing stroke preview SVGs

Generated artifacts are typically written near the input (for example `.strokes.svg` previews).

Supported image extensions in `plotter_v2.py` include:
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`, `.svg`

## Motion Modes

`plotter_v2.py` includes two motion styles:

- `standard`: incremental joint stepping
- `dynamics`: time-parameterized trajectory with limits (`DYN_VEL_MAX_DEG_S`, `DYN_ACCEL_MAX_DEG_S2`, `DYN_DT`)

Use compare commands (`v`, `p`) to benchmark speed and behavior between modes.

## Standalone Test Utilities

Useful scripts in `test/`:
- `test/linedraw.py`: convert images to polyline SVG/JSON
- `test/svg_to_trajectory.py`: map polylines to BrachioGraph-style angle trajectory JSON
- `test/capture_calibration.py`: richer standalone calibration capture to JSON
- `test/move_to_xy.py`: movement testing with loaded calibration data

These tools are helpful for iterating on tracing/calibration without editing the main runtime script.

## Troubleshooting

- **Import errors for hardware libs**
  - Install `telemetrix` and `telemetrix_pca9685`.
  - If hardware isn’t connected yet, run in sim mode first.

- **No movement / wrong movement**
  - Verify servo channel mapping and PCA I2C address.
  - Re-check `L1`, `L2`, elbow convention/sign constants (`K0`, `K1`, `ELBOW_UP`).
  - Recalibrate PWM-angle tables.

- **Drawing shifted or clipped**
  - Inspect printed drawing box bounds at startup.
  - Set `USE_AUTO_BOX=True` and confirm joint limits match hardware reality.
  - Use `b` to trace the box outline before image tracing.

- **Jitter / direction-dependent error**
  - Tune `HYSTERESIS` values and verify calibration points include both motion directions.

- **Simulation window issues on macOS**
  - Keep plotting on main thread (current script already does this).
  - If terminal state gets odd after exit, run `stty sane`.

## Safety Notes

- Start with pen lifted and low-speed motions.
- Keep hand clear of linkages during first calibration and range tests.
- Confirm mechanical limits before enabling larger/faster trajectories.
- Prefer simulation for logic checks before touching hardware constants.

## Suggested Next Improvements

- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Move runtime constants from script globals into a versioned config file.
- Save/load calibration tables from JSON in `plotter_v2.py` directly.
- Add lightweight regression tests for IK edge cases and polyline transforms.

