import math
import os
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kinematics  # pyright: ignore[reportMissingImports]

clamp_angle = kinematics.clamp_angle
xy_to_servo_angles = kinematics.xy_to_servo_angles


# ---------------------------------------------------------
# Runtime mode and board setup
# ---------------------------------------------------------
SIM_MODE = os.environ.get("PLOTTER_SIM", "0") == "1"

if SIM_MODE:
    import sim_driver  # pyright: ignore[reportMissingImports]

    print("Running in SIMULATION mode (PLOTTER_SIM=1).")
    board = sim_driver.SimBoard()
    pca = sim_driver.SimPCA9685()
else:
    try:
        from telemetrix import telemetrix
        from telemetrix_pca9685 import telemetrix_pca9685
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing hardware dependency. Install with:\n"
            "  pip install telemetrix telemetrix-pca9685\n"
            "Or run in simulation mode:\n"
            "  PLOTTER_SIM=1 python final/plotter_feedback_v1.py\n"
            f"\nOriginal error: {e}"
        )

    print("Connecting to Arduino...")
    board = telemetrix.Telemetrix()
    board.set_pin_mode_i2c()
    print("Initializing PCA9685 Driver...")
    pca = telemetrix_pca9685.TelemetrixPCA9685(board=board, i2c_address=0x40)

try:
    pca.set_pwm_freq(50)
except AttributeError:
    pass


# ---------------------------------------------------------
# Servo channels and home angles
# ---------------------------------------------------------
SERVO_CH_0 = 0
SERVO_CH_1 = 1
SERVO_CH_PEN = 2

HOME_0 = 90.0
HOME_1 = 90.0

PEN_UP_PULSE = 300
PEN_DOWN_PULSE = 360
PEN_DELAY = 0.15

STEP_DEGREES = 2.0
PULSE_DELAY = 0.045

PULSE_MIN = 100
PULSE_MAX = 650


# ---------------------------------------------------------
# Kinematics config copied from plotter_v2 defaults
# ---------------------------------------------------------
L1 = 10.3
L2 = 10.9
K0 = -1
K1 = -1
ELBOW_UP = True
SERVO0_MIN_DEG = 0.0
SERVO0_MAX_DEG = 90.0
SERVO1_MIN_DEG = 40.0
SERVO1_MAX_DEG = 160.0

# Cartesian square path (cm), same shape used in plotter_v2.
SQUARE_CORNERS_CM = [
    (4.0, 17.0),   # top-right (start)
    (-2.0, 17.0),  # top-left
    (-2.0, 13.0),  # bottom-left
    (4.0, 13.0),   # bottom-right
]

kinematics.configure(
    L1=L1,
    L2=L2,
    K0=K0,
    K1=K1,
    elbow_up=ELBOW_UP,
    joint_limits=[(SERVO0_MIN_DEG, SERVO0_MAX_DEG), (SERVO1_MIN_DEG, SERVO1_MAX_DEG)],
    joint_margin_deg=3.0,
)


# ---------------------------------------------------------
# Command calibration (angle -> pulse) from plotter_v2
# ---------------------------------------------------------
SERVO0_CAL = [
    (-10.0, 232), (0.0, 244), (10.0, 258), (20.0, 272), (30.0, 285),
    (40.0, 298), (50.0, 314), (60.0, 331), (70.0, 348), (80.0, 365), (90.0, 384),
]
SERVO1_CAL = [
    (40.0, 302), (50.0, 316), (60.0, 332), (70.0, 347), (80.0, 364),
    (90.0, 374), (100.0, 387), (110.0, 403), (120.0, 416), (130.0, 432),
    (140.0, 448), (150.0, 462), (160.0, 478),
]


# ---------------------------------------------------------
# Feedback calibration (angle -> adc), placeholders to tune
# ---------------------------------------------------------
# SERVO0_FB_CAL = [
#     (0.0, 220), (10.0, 270), (20.0, 320), (30.0, 370), (40.0, 420),
#     (50.0, 475), (60.0, 530), (70.0, 585), (80.0, 640), (90.0, 695),
# ]
# SERVO1_FB_CAL = [
#     (40.0, 260), (50.0, 305), (60.0, 350), (70.0, 395), (80.0, 440),
#     (90.0, 485), (100.0, 530), (110.0, 575), (120.0, 620), (130.0, 665),
#     (140.0, 710), (150.0, 755), (160.0, 800),
# ]

SERVO0_FB_CAL = [
    (0.0, 136),
    (90.0, 400),
]
SERVO1_FB_CAL = [
    (90.0, 371),
    (150.0, 555),
]


def _build_cal_poly(cal_points):
    arr = np.array(cal_points, dtype=float)
    if len(arr) >= 4:
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 3))
    if len(arr) >= 2:
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 1))
    raise ValueError("Need at least 2 calibration points per servo.")


_cal_poly_0 = _build_cal_poly(SERVO0_CAL)
_cal_poly_1 = _build_cal_poly(SERVO1_CAL)

# Use monotonic interpolation for inverse (adc -> angle) to avoid weird poly inversions.
_fb_adc_0 = np.array([p[1] for p in SERVO0_FB_CAL], dtype=float)
_fb_deg_0 = np.array([p[0] for p in SERVO0_FB_CAL], dtype=float)
_fb_adc_1 = np.array([p[1] for p in SERVO1_FB_CAL], dtype=float)
_fb_deg_1 = np.array([p[0] for p in SERVO1_FB_CAL], dtype=float)


def _set_feedback_calibration(cal0, cal1):
    global SERVO0_FB_CAL, SERVO1_FB_CAL, _fb_adc_0, _fb_deg_0, _fb_adc_1, _fb_deg_1
    SERVO0_FB_CAL = sorted([(float(a), int(v)) for a, v in cal0], key=lambda t: t[0])
    SERVO1_FB_CAL = sorted([(float(a), int(v)) for a, v in cal1], key=lambda t: t[0])
    _fb_adc_0 = np.array([p[1] for p in SERVO0_FB_CAL], dtype=float)
    _fb_deg_0 = np.array([p[0] for p in SERVO0_FB_CAL], dtype=float)
    _fb_adc_1 = np.array([p[1] for p in SERVO1_FB_CAL], dtype=float)
    _fb_deg_1 = np.array([p[0] for p in SERVO1_FB_CAL], dtype=float)


def _adc_to_angle(servo_idx, adc):
    if adc is None:
        return None
    adc_f = float(adc)
    if servo_idx == 0:
        angle = np.interp(adc_f, _fb_adc_0, _fb_deg_0)
    else:
        angle = np.interp(adc_f, _fb_adc_1, _fb_deg_1)
    return float(clamp_angle(angle))


def _angle_to_pulse(servo_idx, angle):
    poly = _cal_poly_0 if servo_idx == 0 else _cal_poly_1
    pulse = int(round(float(poly(float(clamp_angle(angle))))))
    return max(PULSE_MIN, min(PULSE_MAX, pulse))


# ---------------------------------------------------------
# Analog feedback wiring (A0/A1)
# ---------------------------------------------------------
FB_PIN_0 = 0
FB_PIN_1 = 1
FB_STALE_SEC = 0.35

_feedback_lock = threading.Lock()
_feedback_raw = [None, None]
_feedback_ts = [0.0, 0.0]


def _analog_cb(data):
    # telemetrix callback format used in existing repo: [type, pin, value, ...]
    try:
        pin = int(data[1])
        val = int(data[2])
    except Exception:
        return
    idx = 0 if pin == FB_PIN_0 else 1 if pin == FB_PIN_1 else None
    if idx is None:
        return
    now = time.time()
    with _feedback_lock:
        _feedback_raw[idx] = val
        _feedback_ts[idx] = now


def init_feedback_io():
    if SIM_MODE:
        return
    board.set_pin_mode_analog_input(FB_PIN_0, callback=_analog_cb)
    board.set_pin_mode_analog_input(FB_PIN_1, callback=_analog_cb)


def get_feedback_raw(idx):
    with _feedback_lock:
        return _feedback_raw[idx], _feedback_ts[idx]


def get_feedback_raw_avg(idx, n=8, sample_dt=0.005):
    vals = []
    for _ in range(max(1, n)):
        val, _ = get_feedback_raw(idx)
        if val is not None:
            vals.append(val)
        time.sleep(sample_dt)
    if not vals:
        return None
    return int(round(sum(vals) / len(vals)))


# ---------------------------------------------------------
# Outer-loop PID settle
# ---------------------------------------------------------
@dataclass
class PidCfg:
    kp: float
    ki: float
    kd: float
    i_limit: float
    out_limit: float


@dataclass
class PidState:
    i_term: float = 0.0
    prev_error: float = 0.0
    initialized: bool = False


PID_CFG = [
    PidCfg(kp=0.85, ki=0.015, kd=0.045, i_limit=25.0, out_limit=10.0),
    PidCfg(kp=0.85, ki=0.015, kd=0.045, i_limit=25.0, out_limit=10.0),
]
_pid_state = [PidState(), PidState()]

SETTLE_DT = 0.02
SETTLE_TIMEOUT_S = 1.2
SETTLE_TOL_DEG = 1.4
SETTLE_STABLE_CYCLES = 4


def reset_pid():
    for st in _pid_state:
        st.i_term = 0.0
        st.prev_error = 0.0
        st.initialized = False


def _pid_trim(idx, target_deg, measured_deg, dt):
    cfg = PID_CFG[idx]
    st = _pid_state[idx]
    err = target_deg - measured_deg
    st.i_term += err * dt
    st.i_term = max(-cfg.i_limit, min(cfg.i_limit, st.i_term))
    deriv = 0.0 if not st.initialized else (err - st.prev_error) / max(1e-6, dt)
    st.prev_error = err
    st.initialized = True
    out = cfg.kp * err + cfg.ki * st.i_term + cfg.kd * deriv
    return max(-cfg.out_limit, min(cfg.out_limit, out))


# ---------------------------------------------------------
# Pen + servo primitives
# ---------------------------------------------------------
def pen_up():
    pca.set_pwm(SERVO_CH_PEN, 0, PEN_UP_PULSE)
    time.sleep(PEN_DELAY)


def pen_down():
    pca.set_pwm(SERVO_CH_PEN, 0, PEN_DOWN_PULSE)
    time.sleep(PEN_DELAY)


def set_servo_angle(channel, angle):
    idx = 0 if channel == SERVO_CH_0 else 1
    pulse = _angle_to_pulse(idx, angle)
    pca.set_pwm(channel, 0, pulse)


def read_measured_angles():
    raw0, ts0 = get_feedback_raw(0)
    raw1, ts1 = get_feedback_raw(1)
    now = time.time()
    stale0 = (now - ts0) > FB_STALE_SEC if ts0 > 0 else True
    stale1 = (now - ts1) > FB_STALE_SEC if ts1 > 0 else True
    a0 = None if stale0 else _adc_to_angle(0, raw0)
    a1 = None if stale1 else _adc_to_angle(1, raw1)
    return a0, a1, raw0, raw1, stale0, stale1


def settle_closed_loop(target0, target1):
    reset_pid()
    stable = 0
    t0 = time.time()
    while (time.time() - t0) < SETTLE_TIMEOUT_S:
        m0, m1, _, _, s0, s1 = read_measured_angles()
        if s0 or s1 or m0 is None or m1 is None:
            time.sleep(SETTLE_DT)
            continue
        e0 = target0 - m0
        e1 = target1 - m1
        if abs(e0) <= SETTLE_TOL_DEG and abs(e1) <= SETTLE_TOL_DEG:
            stable += 1
            if stable >= SETTLE_STABLE_CYCLES:
                return True
        else:
            stable = 0
        t0_trim = _pid_trim(0, target0, m0, SETTLE_DT)
        t1_trim = _pid_trim(1, target1, m1, SETTLE_DT)
        set_servo_angle(SERVO_CH_0, target0 + t0_trim)
        set_servo_angle(SERVO_CH_1, target1 + t1_trim)
        time.sleep(SETTLE_DT)
    return False


def move_both_smooth(a0s, a1s, a0e, a1e, do_settle=True):
    a0s, a1s = float(clamp_angle(a0s)), float(clamp_angle(a1s))
    a0e, a1e = float(clamp_angle(a0e)), float(clamp_angle(a1e))
    d0 = abs(a0e - a0s)
    d1 = abs(a1e - a1s)
    if d0 < 0.01 and d1 < 0.01:
        return
    steps = max(1, int(round(max(d0, d1) / STEP_DEGREES)))
    for i in range(1, steps + 1):
        t = i / steps
        set_servo_angle(SERVO_CH_0, a0s + (a0e - a0s) * t)
        set_servo_angle(SERVO_CH_1, a1s + (a1e - a1s) * t)
        time.sleep(PULSE_DELAY)
    set_servo_angle(SERVO_CH_0, a0e)
    set_servo_angle(SERVO_CH_1, a1e)
    if do_settle:
        settle_closed_loop(a0e, a1e)


def goto_xy(a0, a1, x, y):
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError as e:
        print(f"  IK error: {e}")
        return a0, a1
    s0c, s1c = clamp_angle(s0), clamp_angle(s1)
    move_both_smooth(a0, a1, s0c, s1c, do_settle=True)
    return s0c, s1c


def draw_square(a0, a1):
    """Trace the Cartesian square path defined by SQUARE_CORNERS_CM."""
    if not SQUARE_CORNERS_CM:
        return a0, a1
    pen_up()
    x0, y0 = SQUARE_CORNERS_CM[0]
    a0, a1 = goto_xy(a0, a1, x0, y0)
    pen_down()
    for x, y in SQUARE_CORNERS_CM[1:] + [SQUARE_CORNERS_CM[0]]:
        a0, a1 = goto_xy(a0, a1, x, y)
    pen_up()
    return a0, a1


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_xy():
    try:
        raw = input("Enter target as 'x y' in cm (blank to cancel): ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != 2:
        print("  Expected two numbers, e.g. '10 12'.")
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("  Could not parse numbers.")
        return None


# ---------------------------------------------------------
# Teach & replay
# ---------------------------------------------------------
recorded_poses = []


def record_pose(name=None):
    raw0 = get_feedback_raw_avg(0, n=12)
    raw1 = get_feedback_raw_avg(1, n=12)
    if raw0 is None or raw1 is None:
        print("Record failed: no feedback yet from A0/A1.")
        return
    deg0 = _adc_to_angle(0, raw0)
    deg1 = _adc_to_angle(1, raw1)
    if name is None:
        name = f"pose_{len(recorded_poses)+1}"
    pose = {"name": name, "a0_adc": raw0, "a1_adc": raw1, "a0_deg": deg0, "a1_deg": deg1}
    recorded_poses.append(pose)
    print(
        f"Recorded {name}: "
        f"A0={raw0} ({deg0:.2f} deg), A1={raw1} ({deg1:.2f} deg)"
    )


def list_poses():
    if not recorded_poses:
        print("No recorded poses.")
        return
    for i, p in enumerate(recorded_poses, 1):
        print(
            f"[{i}] {p['name']}: "
            f"s0={p['a0_deg']:.2f} deg (adc={p['a0_adc']}), "
            f"s1={p['a1_deg']:.2f} deg (adc={p['a1_adc']})"
        )


def clear_poses():
    recorded_poses.clear()
    print("Cleared all recorded poses.")


def play_poses(angle0, angle1):
    if not recorded_poses:
        print("No poses recorded.")
        return angle0, angle1
    print(f"Playing {len(recorded_poses)} recorded poses...")
    for p in recorded_poses:
        t0 = float(clamp_angle(p["a0_deg"]))
        t1 = float(clamp_angle(p["a1_deg"]))
        move_both_smooth(angle0, angle1, t0, t1, do_settle=True)
        angle0, angle1 = t0, t1
        print(f"  -> {p['name']} done (cmd: {t0:.2f}, {t1:.2f})")
    return angle0, angle1


def feedback_status():
    m0, m1, r0, r1, s0, s1 = read_measured_angles()
    age0 = time.time() - get_feedback_raw(0)[1] if get_feedback_raw(0)[1] > 0 else float("inf")
    age1 = time.time() - get_feedback_raw(1)[1] if get_feedback_raw(1)[1] > 0 else float("inf")
    msg0 = f"A0 raw={r0} age={age0:.3f}s"
    msg1 = f"A1 raw={r1} age={age1:.3f}s"
    if m0 is not None:
        msg0 += f" -> {m0:.2f} deg"
    if m1 is not None:
        msg1 += f" -> {m1:.2f} deg"
    if s0:
        msg0 += " [stale]"
    if s1:
        msg1 += " [stale]"
    print(msg0)
    print(msg1)


def guided_feedback_calibration(angle0, angle1):
    if SIM_MODE:
        print("Guided feedback calibration is hardware-only (SIM mode has no analog pots).")
        return angle0, angle1

    print("\n=== GUIDED FEEDBACK CALIBRATION ===")
    print("Goal: build angle->ADC map for A0/A1.")
    print("Jog controls:")
    print("  servo0: a/s = -/+10 deg, A/S = -/+1 deg")
    print("  servo1: k/l = -/+10 deg, K/L = -/+1 deg")
    print("Capture controls:")
    print("  0 or 1 : capture point for servo0/servo1")
    print("  b      : capture both servos at once")
    print("  v      : view captured points")
    print("  ENTER  : finish and print tables")
    print("  q      : cancel")
    print("At capture prompt, enter the known physical angle in degrees.")
    print("")

    cal0 = []
    cal1 = []

    set_servo_angle(SERVO_CH_0, angle0)
    set_servo_angle(SERVO_CH_1, angle1)
    settle_closed_loop(angle0, angle1)
    feedback_status()

    while True:
        ch = read_key()
        if ch in ("\r", "\n"):
            break
        if ch in ("q", "Q", "\x03"):
            print("Calibration canceled.")
            return angle0, angle1
        if ch in ("a", "s", "A", "S", "k", "l", "K", "L"):
            if ch == "a":
                angle0 = float(clamp_angle(angle0 - 10))
            elif ch == "s":
                angle0 = float(clamp_angle(angle0 + 10))
            elif ch == "A":
                angle0 = float(clamp_angle(angle0 - 1))
            elif ch == "S":
                angle0 = float(clamp_angle(angle0 + 1))
            elif ch == "k":
                angle1 = float(clamp_angle(angle1 - 10))
            elif ch == "l":
                angle1 = float(clamp_angle(angle1 + 10))
            elif ch == "K":
                angle1 = float(clamp_angle(angle1 - 1))
            elif ch == "L":
                angle1 = float(clamp_angle(angle1 + 1))
            move_both_smooth(angle0, angle1, angle0, angle1, do_settle=False)
            set_servo_angle(SERVO_CH_0, angle0)
            set_servo_angle(SERVO_CH_1, angle1)
            time.sleep(0.12)
            print(f"Commanded: servo0={angle0:.1f} deg, servo1={angle1:.1f} deg")
            feedback_status()
            continue
        if ch in ("v", "V"):
            print(f"servo0 points: {sorted(cal0, key=lambda p: p[0])}")
            print(f"servo1 points: {sorted(cal1, key=lambda p: p[0])}")
            continue
        if ch in ("0", "1", "b", "B"):
            try:
                if ch in ("0", "b", "B"):
                    raw0 = get_feedback_raw_avg(0, n=12)
                    if raw0 is None:
                        print("A0 has no feedback sample yet.")
                    else:
                        a0 = float(input("\nKnown physical angle for servo0 (deg): ").strip())
                        cal0.append((a0, raw0))
                        print(f"Captured servo0: ({a0:.2f}, {raw0})")
                if ch in ("1", "b", "B"):
                    raw1 = get_feedback_raw_avg(1, n=12)
                    if raw1 is None:
                        print("A1 has no feedback sample yet.")
                    else:
                        a1 = float(input("Known physical angle for servo1 (deg): ").strip())
                        cal1.append((a1, raw1))
                        print(f"Captured servo1: ({a1:.2f}, {raw1})")
            except ValueError:
                print("Invalid angle input; point not saved.")
            continue

    if len(cal0) < 2 or len(cal1) < 2:
        print("Need at least 2 points per servo to activate mapping.")
        print("No calibration tables updated.")
        return angle0, angle1

    cal0 = sorted(cal0, key=lambda t: t[0])
    cal1 = sorted(cal1, key=lambda t: t[0])
    _set_feedback_calibration(cal0, cal1)

    print("\n=== PASTE-READY FEEDBACK CAL TABLES ===")
    print("SERVO0_FB_CAL = [")
    for a, v in cal0:
        print(f"    ({a:.1f}, {int(v)}),")
    print("]")
    print("SERVO1_FB_CAL = [")
    for a, v in cal1:
        print(f"    ({a:.1f}, {int(v)}),")
    print("]")
    print("Calibration applied for this session.")
    print("======================================\n")
    return angle0, angle1


def print_controls():
    print("\nControls:")
    print("  r       home position")
    print("  x       test step (servo0 +30 deg, servo1 +20 deg)")
    print("  s       draw square")
    print("  g       goto (x, y) cm")
    print("  u       pen up")
    print("  d       pen down")
    print("  j       feedback status (A0/A1 + mapped angles)")
    print("  m       record pose from feedback")
    print("  l       list recorded poses")
    print("  z       clear recorded poses")
    print("  y       play recorded poses")
    print("  c       guided feedback calibration")
    print("  q       quit")


def main():
    global_angle0 = HOME_0
    global_angle1 = HOME_1

    init_feedback_io()
    time.sleep(0.4)

    pen_up()
    set_servo_angle(SERVO_CH_0, global_angle0)
    set_servo_angle(SERVO_CH_1, global_angle1)
    settle_closed_loop(global_angle0, global_angle1)

    print_controls()

    try:
        while True:
            ch = read_key()
            if ch in ("\x03", "q", "Q"):
                break
            if ch in ("r", "R"):
                pen_up()
                move_both_smooth(global_angle0, global_angle1, HOME_0, HOME_1)
                global_angle0, global_angle1 = HOME_0, HOME_1
                print(f"Home: servo0={global_angle0:.2f}, servo1={global_angle1:.2f}")
            elif ch in ("x", "X"):
                t0 = clamp_angle(global_angle0 + 30)
                t1 = clamp_angle(global_angle1 + 20)
                move_both_smooth(global_angle0, global_angle1, t0, t1)
                global_angle0, global_angle1 = float(t0), float(t1)
                print(f"Step: servo0={global_angle0:.2f}, servo1={global_angle1:.2f}")
            elif ch in ("s", "S"):
                global_angle0, global_angle1 = draw_square(global_angle0, global_angle1)
                print(f"Square done: servo0={global_angle0:.2f}, servo1={global_angle1:.2f}")
            elif ch in ("g", "G"):
                target = prompt_xy()
                if target is not None:
                    pen_up()
                    global_angle0, global_angle1 = goto_xy(global_angle0, global_angle1, target[0], target[1])
                    print(f"Goto done: servo0={global_angle0:.2f}, servo1={global_angle1:.2f}")
            elif ch in ("u", "U"):
                pen_up()
                print("Pen Up")
            elif ch in ("d", "D"):
                pen_down()
                print("Pen Down")
            elif ch in ("j", "J"):
                feedback_status()
            elif ch in ("m", "M"):
                record_pose()
            elif ch in ("l", "L"):
                list_poses()
            elif ch in ("z", "Z"):
                clear_poses()
            elif ch in ("y", "Y"):
                global_angle0, global_angle1 = play_poses(global_angle0, global_angle1)
            elif ch in ("c", "C"):
                global_angle0, global_angle1 = guided_feedback_calibration(global_angle0, global_angle1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        try:
            pen_up()
        except Exception:
            pass
        try:
            board.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
