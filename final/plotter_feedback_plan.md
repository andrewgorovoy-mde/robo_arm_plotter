# Analog Feedback Upgrade Plan (Keep `plotter_v2.py` Clean)

This file is a build blueprint for adding analog-feedback closed-loop control and teach/replay workflow **without modifying** `plotter_v2.py` first.

Recommended approach:
- Create a new runtime file (example: `plotter_feedback_v1.py`) by copying `plotter_v2.py`
- Apply the phases below in order
- Keep `plotter_v2.py` as your stable baseline

---

## 1) New file strategy

- **Source of truth remains:** `plotter_v2.py`
- **Experimental branch file:** `plotter_feedback_v1.py`
- **Optional shared module later:** `feedback_control.py` (once stable)

Why:
- protects your known-good motion and tracing behavior
- lets you tune feedback/PID safely
- easy A/B comparison between open-loop and closed-loop behavior

---

## 2) Phase-by-phase implementation

## Phase A: Feedback plumbing + visibility

Goal: read A0/A1 continuously and print measured joint states.

### Add constants
Add near servo config:
- `FB_PIN_0 = 0`  (A0)
- `FB_PIN_1 = 1`  (A1)
- `FB_SAMPLE_HZ = 50`
- `FB_STALE_SEC = 0.25`

If your board reports analog pins by symbolic index, adjust the values to match Telemetrix expectations.

### Add runtime state
Add global state:
- `feedback_raw = [None, None]`
- `feedback_ts = [0.0, 0.0]`
- `feedback_lock = threading.Lock()`

### Add callback + accessors
Add helper functions:
- `_feedback_cb_servo0(data)` / `_feedback_cb_servo1(data)`
- `init_feedback_io()`
- `get_feedback_raw(servo_idx)`
- `get_feedback_raw_avg(servo_idx, n=8, dt=0.005)`

### Wire startup
In non-sim startup flow, call `init_feedback_io()` after board init.

### Add diagnostic command
Add key + command:
- key: `j`
- command tuple: `("feedback_status",)`

`feedback_status` prints:
- latest ADC values for servo0/servo1
- sample age
- stale warning

Acceptance criteria:
- You can press `j` and see live ADC changes when joints move.

---

## Phase B: Feedback calibration model (angle <-> ADC)

Goal: build stable `adc -> angle` conversion from measured points.

### New tables
Add:
- `SERVO0_FB_CAL = [(deg, adc), ...]`
- `SERVO1_FB_CAL = [(deg, adc), ...]`

Start with placeholder values and refine from calibration runs.

### Model builders
Add functions:
- `_build_fb_poly(cal_points)` for `angle -> adc`
- `_build_fb_inv_poly(cal_points)` for `adc -> angle`
- `_fb_to_angle(servo_idx, adc)` convenience wrapper

Prefer low-order fits (linear/cubic) only if monotonic and stable. If not, switch to piecewise linear interpolation.

### Sanity command
Add:
- key: `k`
- command: `("feedback_angles",)`

Print:
- raw ADC
- mapped angle estimate
- commanded angle (`angle0`, `angle1`)
- error estimate (`commanded - measured`)

Acceptance criteria:
- At rest, mapped angles roughly match commanded positions.

---

## Phase C: Teach/record/list/clear/play waypoints

Goal: record poses after joystick/UI positioning and replay them.

### Data structure
Add:
- `recorded_poses = []`
- pose format:
  - `{"name": str, "a0_adc": int, "a1_adc": int, "a0_deg": float, "a1_deg": float}`

### Pose capture helper
Add:
- `capture_current_pose(name=None, avg_n=12)`

Behavior:
- average A0/A1
- convert to degrees via inverse map
- append to `recorded_poses`
- print concise pose summary

### Commands + keybinds
Add:
- `("record_pose",)` key `m`
- `("list_poses",)` key `l`
- `("clear_poses",)` key `z`
- `("play_poses",)` key `y`

Playback v1 can use open-loop motion to each recorded `deg` target first.

Acceptance criteria:
- Record multiple poses, list them, replay sequence from home.

---

## Phase D: Closed-loop settle controller (outer loop PID)

Goal: improve endpoint accuracy without fighting servo internal control.

### PID state and gains
Add per-servo:
- gains (`kp`, `ki`, `kd`)
- `i_term`, `prev_error`, `prev_time`
- `i_clamp`, `out_clamp`
- deadband in ADC or degrees

### Control helper functions
Add:
- `reset_pid_states()`
- `_pid_step(servo_idx, target_deg, measured_deg, dt)`
- `settle_to_target_closed_loop(target0_deg, target1_deg, timeout=1.5)`

Recommended settle flow:
1. open-loop move to target via existing `move_both(...)`
2. run short 20-50 Hz loop:
   - read measured angles
   - compute PID trim
   - apply `set_servo_angle(..., target + trim)`
   - require N consecutive in-tolerance cycles before success

### Integrate at low risk points first
Use settle helper in:
- `goto_xy(...)` endpoint
- `play_poses` waypoint transitions

Defer full continuous closed-loop during long traces until tuned.

Acceptance criteria:
- Endpoint error decreases and oscillation remains controlled.

---

## Phase E: Teach-mode motor power workflow (optional)

Goal: allow manual arm repositioning during learning.

### Add optional power API
If hardware supports it:
- `servo_power(on: bool)`
- key `[` -> motor off
- key `]` -> motor on

If external switch only:
- keep software command as operator prompt (no GPIO action), e.g. print:
  - "Turn servo power OFF, position arm, then press m to record."

### Safety policy
- force `pen_up()` before motor-off
- disallow playback when motors are off
- require a short settle delay after motor-on before record/read

Acceptance criteria:
- teach flow works repeatably with your external power setup.

---

## 3) Function-level change map (for `plotter_feedback_v1.py`)

Use this as a checklist while editing:

- **Global config section**
  - add feedback pin constants, gains, thresholds, flags

- **Startup block (`SIM_MODE` split)**
  - add `init_feedback_io()` in hardware mode
  - add no-op/sim stubs in simulation mode

- **Calibration section**
  - add feedback calibration tables + polynomial/interpolation builders

- **Motion section**
  - keep existing `move_*` unchanged initially
  - add `settle_to_target_closed_loop(...)`

- **IK wrapper (`goto_xy`)**
  - call closed-loop settle after `move_both(...)`

- **Dispatcher (`dispatch_command`)**
  - add command handlers for status/record/list/clear/play/power

- **Keymap (`_key_to_command`)**
  - map new keys to new command tuples

- **Controls printout**
  - document all new keys clearly

---

## 4) Tuning recipe (outer-loop only)

- Start: `ki=0`, `kd=0`
- Increase `kp` until near-oscillation, then back off 20-30%
- Add small `kd` for overshoot damping
- Add tiny `ki` only for persistent steady-state bias
- Keep `dt` fixed and add:
  - integrator clamp
  - deadband near zero error
  - output clamp to avoid large correction jumps

Do tuning with:
- repeated `goto` tests near center + corners
- then pose replay
- finally image trace endpoints

---

## 5) Suggested milestone commands/tests

After each phase:

- **A:** Move arm manually, press `j`, verify ADC updates.
- **B:** Home + step, press `k`, verify measured angle tracks command.
- **C:** Record 3-5 poses with `m`, replay with `y`.
- **D:** Compare endpoint error:
  - open-loop only vs open-loop + settle
- **E:** Validate safe motor off/on teach workflow.

---

## 6) Exit criteria for promoting to main script

Move changes back into your production file only after:
- no stale-feedback read issues
- replay works for at least 20 consecutive cycles
- endpoint error reduced consistently
- no sustained oscillation/noise
- no regressions in existing shape/image modes

At that point either:
- replace `plotter_v2.py` with stable feedback version, or
- keep both and add a runtime flag (e.g. `--feedback`) for A/B operation.
