import bisect
import math
import os
import sys
import termios
import tty
import time
import numpy as np
from telemetrix import telemetrix
from telemetrix_pca9685 import telemetrix_pca9685

# ---------------------------------------------------------
# 1. Initialize the Main Arduino Board
# ---------------------------------------------------------
print("Connecting to Arduino...")
board = telemetrix.Telemetrix()
board.set_pin_mode_i2c()

# ---------------------------------------------------------
# 2. Initialize the PCA9685 Servo Driver
# ---------------------------------------------------------
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
HOME_0 = 90
HOME_1 = 90

# Ramp tuning
STEP_DEGREES = 2
PULSE_DELAY = 0.055

# Shape parameters
SQUARE_EDGE_DEG = 18
STAR_RADIUS_DEG = 18

# ---------------------------------------------------------
# Link lengths in cm
# ---------------------------------------------------------
L1 = 12.6
L2 = 14.5
HOME_X = 12.6
HOME_Y = 14.5
K0 = +1
K1 = +1
ELBOW_UP = True

# ---------------------------------------------------------
# Image -> waypoints feature
# ---------------------------------------------------------
DRAW_CENTER_X = 12.6
DRAW_CENTER_Y = 14.5
DRAW_BOX_CM = 6.0
MAX_WAYPOINTS = 500
DEFAULT_IMAGE_PATH = "images/banana.png"
WAYPOINTS_SVG_SUFFIX = ".waypoints.svg"

# ==========================================================
# SERVO CALIBRATION
# ==========================================================
#
# HOW TO CALIBRATE (run once, then paste results here):
#
#   1. Start the program and press 'c' to enter calibration mode.
#   2. Use keys to nudge servos to a known physical angle.
#      (Use a protractor, or align against a marked reference.)
#      Keys: a/s = servo0 coarse ±10,  A/S = servo0 fine ±1
#            k/l = servo1 coarse ±10,  K/L = servo1 fine ±1
#   3. Press SPACE to record the current pulse values at the
#      angle you measured.  Repeat for ~6-8 angles spread across
#      the full range (e.g. 0, 30, 60, 90, 120, 150, 180).
#   4. Press '0' to exit calibration — the program prints a
#      ready-to-paste CAL block.
#   5. Replace the placeholder tables below with those values.
#
# FORMAT: list of (angle_degrees, pca_pulse_count) pairs.
# The pulse range for a typical SG90 at 50 Hz is roughly 150-600
# on a 12-bit PCA9685, but your servos may differ — that is exactly
# what calibration discovers.
#
# PLACEHOLDER (linear approximation) — replace after calibration:
SERVO0_CAL = [
    (0,   150),
    (45,  262),
    (90,  375),
    (135, 490),
    (180, 608),
]

SERVO1_CAL = [
    (0,   150),
    (45,  262),
    (90,  375),
    (135, 490),
    (180, 608),
]

# ---------------------------------------------------------
# Hysteresis correction (pulse counts, one per servo)
#
# WHY: cheap servos arrive at slightly different positions
# depending on which direction they approached from.  A positive
# value nudges the pulse a little further in the direction of
# travel to compensate.
#
# HOW TO TUNE:
#   • Start at 0 for both.
#   • Draw a box with 's'.  If opposite sides are not parallel
#     (one side bows in, the other bows out), increase the value
#     for the servo that controls that axis.
#   • Typical range: 2-8 pulse counts.  Above ~12 usually means
#     something else is wrong mechanically.
#
HYSTERESIS = [4, 4]   # [servo0, servo1]  — tune after calibration

# ==========================================================
# Build calibration polynomials from the tables above
# ==========================================================
def _build_cal_poly(cal_points):
    """Fit a cubic polynomial to (angle, pulse) calibration data."""
    arr = np.array(cal_points, dtype=float)
    if len(arr) >= 4:
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 3))
    elif len(arr) >= 2:
        # Fall back to linear if we have fewer points
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 1))
    else:
        raise ValueError("Need at least 2 calibration points per servo.")

_cal_poly_0 = _build_cal_poly(SERVO0_CAL)
_cal_poly_1 = _build_cal_poly(SERVO1_CAL)

# Hysteresis state — updated by set_servo_angle on every move
_prev_pulse   = [0, 0]       # last pulse sent to each servo
_active_hyst  = [0.0, 0.0]  # currently applied correction

# Pulse safety limits — widen only if you are sure your hardware allows it
PULSE_MIN = 100
PULSE_MAX = 650


# ==========================================================
# Core servo driver — everything flows through here
# ==========================================================
def set_servo_angle(channel, angle):
    """
    Send a calibrated, hysteresis-corrected pulse to a servo channel.

    Flow:
        angle  →  cubic cal poly  →  raw pulse
               →  hysteresis correction applied based on direction
               →  safety clamp
               →  pca.set_pwm()
    """
    angle = float(clamp_angle(angle))

    # 1. Map angle → pulse via the fitted polynomial
    poly = _cal_poly_0 if channel == SERVO_CH_0 else _cal_poly_1
    raw_pulse = float(poly(angle))

    # 2. Hysteresis correction:
    #    If the pulse is increasing (arm moving in one direction) add a
    #    small positive nudge; if decreasing, subtract it.
    #    This compensates for the mechanical dead-band that causes the
    #    arm to undershoot when reversing direction.
    hyst = HYSTERESIS[channel]
    if raw_pulse > _prev_pulse[channel] + 0.01:
        _active_hyst[channel] = +hyst
    elif raw_pulse < _prev_pulse[channel] - 0.01:
        _active_hyst[channel] = -hyst
    # else: direction unchanged — keep existing correction

    _prev_pulse[channel] = raw_pulse
    corrected = raw_pulse + _active_hyst[channel]

    # 3. Safety clamp and send
    pulse = int(round(max(PULSE_MIN, min(PULSE_MAX, corrected))))
    pca.set_pwm(channel, 0, pulse)


# ==========================================================
# Calibration capture tool  (press 'c' in the main loop)
# ==========================================================
def capture_calibration():
    """
    Interactive tool to build per-servo angle→pulse calibration tables.

    Drive each servo to a known physical angle (use a protractor or a
    printed reference card), then press SPACE to record it.  When done,
    press '0' — the program prints ready-to-paste CAL tables.

    Keys
    ----
    a / s    servo0 coarse  -10 / +10 pulse counts
    A / S    servo0 fine     -1 /  +1 pulse counts
    k / l    servo1 coarse  -10 / +10 pulse counts
    K / L    servo1 fine     -1 /  +1 pulse counts
    SPACE    record current pulses — program prompts for the angle
    v        print captured data so far
    0        finish and print paste-ready CAL tables
    """
    print("\n=== CALIBRATION MODE ===")
    print("Drive each servo to a known angle, then SPACE to record.")
    print("a/s=servo0 ±10  A/S=servo0 ±1  k/l=servo1 ±10  K/L=servo1 ±1")
    print("SPACE=record  v=view so far  0=finish\n")

    # Start from wherever the servos currently are
    pw0 = int(_prev_pulse[SERVO_CH_0]) or 375
    pw1 = int(_prev_pulse[SERVO_CH_1]) or 375

    # Send raw pulses during calibration, bypassing the cal layer
    def _raw(ch, pw):
        pw = int(max(PULSE_MIN, min(PULSE_MAX, pw)))
        pca.set_pwm(ch, 0, pw)

    _raw(SERVO_CH_0, pw0)
    _raw(SERVO_CH_1, pw1)

    cal0 = []  # list of (angle, pulse) for servo 0
    cal1 = []  # list of (angle, pulse) for servo 1

    controls = {
        "a": (-10, 0), "s": (+10, 0),
        "A": (-1,  0), "S": (+1,  0),
        "k": (0, -10), "l": (0, +10),
        "K": (0,  -1), "L": (0,  +1),
    }

    while True:
        ch = read_key()

        if ch in controls:
            d0, d1 = controls[ch]
            pw0 = int(max(PULSE_MIN, min(PULSE_MAX, pw0 + d0)))
            pw1 = int(max(PULSE_MIN, min(PULSE_MAX, pw1 + d1)))
            _raw(SERVO_CH_0, pw0)
            _raw(SERVO_CH_1, pw1)
            print(f"  pulses → servo0={pw0}  servo1={pw1}")

        elif ch == " ":
            # Determine which servo was last moved
            raw = input(
                "\n  Recording. Which servo did you just position? (0 / 1 / both): "
            ).strip()
            if raw in ("0", "1", "both"):
                try:
                    angle_raw = input("  Physical angle in degrees: ").strip()
                    angle = float(angle_raw)
                except ValueError:
                    print("  Bad angle — skipping.")
                    continue
                if raw in ("0", "both"):
                    cal0.append((angle, pw0))
                    print(f"  ✓ servo0: {angle}° → pulse {pw0}")
                if raw in ("1", "both"):
                    cal1.append((angle, pw1))
                    print(f"  ✓ servo1: {angle}° → pulse {pw1}")
            else:
                print("  Enter 0, 1, or both.")

        elif ch == "v":
            print(f"\n  servo0 so far: {sorted(cal0)}")
            print(f"  servo1 so far: {sorted(cal1)}\n")

        elif ch == "0":
            break

    # Print paste-ready output
    print("\n=== CALIBRATION RESULTS — paste into your script ===\n")
    print("SERVO0_CAL = [")
    for a, p in sorted(cal0):
        print(f"    ({a:.1f}, {p}),")
    print("]\n")
    print("SERVO1_CAL = [")
    for a, p in sorted(cal1):
        print(f"    ({a:.1f}, {p}),")
    print("]\n")

    if len(cal0) >= 2 and len(cal1) >= 2:
        print("Fitting polynomials to captured data...")
        global _cal_poly_0, _cal_poly_1
        _cal_poly_0 = _build_cal_poly(cal0)
        _cal_poly_1 = _build_cal_poly(cal1)
        print("  ✓ Calibration active for this session.\n")
    else:
        print("  Not enough points to fit (need ≥ 2 per servo) — keeping old cal.\n")

    print("=== END CALIBRATION ===\n")


# ==========================================================
# Motion helpers (unchanged API, now use calibrated driver)
# ==========================================================
def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clamp_angle(a):
    return max(0, min(180, a))


def move_servo_smooth(channel, start, end):
    """Ramp a single servo from start to end degrees."""
    start = float(clamp_angle(start))
    end   = float(clamp_angle(end))
    if abs(end - start) < 0.01:
        return
    direction = 1.0 if end > start else -1.0
    current = start
    while abs(end - current) > 0.01:
        step = min(STEP_DEGREES, abs(end - current))
        current += direction * step
        current = max(end, current) if direction < 0 else min(end, current)
        set_servo_angle(channel, current)
        time.sleep(PULSE_DELAY)
    set_servo_angle(channel, end)


def move_both_smooth(a0s, a1s, a0e, a1e):
    """
    Move both servos together along a straight line in (angle0, angle1) space.

    Both servos arrive at their targets at the same time, which keeps the pen
    path as linear as possible in joint space.
    """
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


# ==========================================================
# Shapes
# ==========================================================
def draw_square(a0, a1):
    """Trace a small square in joint space and return to start."""
    edge = SQUARE_EDGE_DEG
    for d0, d1 in [(+edge, 0), (0, +edge), (-edge, 0), (0, -edge)]:
        na0 = clamp_angle(a0 + d0)
        na1 = clamp_angle(a1 + d1)
        move_both_smooth(a0, a1, na0, na1)
        a0, a1 = na0, na1
    return a0, a1


def draw_star(a0, a1):
    """5-point star (pentagram) in joint space."""
    R  = STAR_RADIUS_DEG
    cx, cy = a0, a1
    V = [
        (clamp_angle(cx + R * math.cos(math.radians(90 - k * 72))),
         clamp_angle(cy + R * math.sin(math.radians(90 - k * 72))))
        for k in range(5)
    ]
    pos = (cx, cy)
    for target in [V[0], V[2], V[4], V[1], V[3], V[0], (cx, cy)]:
        move_both_smooth(pos[0], pos[1], target[0], target[1])
        pos = target
    return cx, cy


# ==========================================================
# Inverse kinematics
# ==========================================================
def xy_to_servo_angles(x, y, elbow_up=ELBOW_UP):
    """Solve 2-link planar IK. Returns (servo0_deg, servo1_deg)."""
    r2 = x * x + y * y
    r  = math.sqrt(r2)
    reach_min = abs(L1 - L2)
    reach_max = L1 + L2
    if r < reach_min - 1e-6 or r > reach_max + 1e-6:
        raise ValueError(
            f"Target ({x:.2f}, {y:.2f}) cm unreachable: |r|={r:.2f}, "
            f"reach=[{reach_min:.2f}, {reach_max:.2f}]"
        )
    cos_phi = (r2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_phi = max(-1.0, min(1.0, cos_phi))
    sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi**2))
    if not elbow_up:
        sin_phi = -sin_phi
    phi    = math.atan2(sin_phi, cos_phi)
    theta1 = math.atan2(y, x) - math.atan2(L2 * sin_phi, L1 + L2 * cos_phi)
    servo0 = 90.0 + K0 * math.degrees(theta1)
    servo1 = 90.0 + K1 * (math.degrees(phi) - 90.0)
    return servo0, servo1


def goto_xy(a0, a1, x, y):
    """Move pen tip to (x, y) cm. Returns updated (a0, a1)."""
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError as e:
        print(f"  IK error: {e}")
        return a0, a1
    s0c, s1c = clamp_angle(s0), clamp_angle(s1)
    if (s0c, s1c) != (s0, s1):
        print(f"  WARN: clamped ({s0:.2f},{s1:.2f}) → ({s0c:.2f},{s1c:.2f})")
    print(f"  IK: ({x:.2f},{y:.2f}) cm → servo0={s0c:.2f}°, servo1={s1c:.2f}°")
    move_both_smooth(a0, a1, s0c, s1c)
    return s0c, s1c


def prompt_xy():
    """Prompt user for x,y target. Returns (x, y) or None."""
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


# ==========================================================
# Image → waypoints
# ==========================================================
def _polyline_arclen(poly):
    return sum(
        math.hypot(poly[i][0] - poly[i-1][0], poly[i][1] - poly[i-1][1])
        for i in range(1, len(poly))
    )


def _resample_polyline(poly, n):
    if n <= 1 or len(poly) <= 1:
        return [poly[0]] * max(1, n)
    cum = [0.0]
    for i in range(1, len(poly)):
        cum.append(cum[-1] + math.hypot(
            poly[i][0] - poly[i-1][0], poly[i][1] - poly[i-1][1]))
    total = cum[-1]
    if total < 1e-9:
        return [poly[0]] * n
    out = []
    for k in range(n):
        target = total * k / (n - 1)
        idx = bisect.bisect_left(cum, target)
        if idx <= 0:
            out.append(tuple(poly[0]))
        elif idx >= len(poly):
            out.append(tuple(poly[-1]))
        else:
            seg = cum[idx] - cum[idx-1]
            t = 0.0 if seg < 1e-9 else (target - cum[idx-1]) / seg
            x = poly[idx-1][0] + t * (poly[idx][0] - poly[idx-1][0])
            y = poly[idx-1][1] + t * (poly[idx][1] - poly[idx-1][1])
            out.append((x, y))
    return out


def write_waypoints_svg(waypoints, svg_path, dot_radius=0.05, padding_cm=1.0):
    if not waypoints:
        return
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    minx, maxx = min(xs) - padding_cm, max(xs) + padding_cm
    miny, maxy = min(ys) - padding_cm, max(ys) + padding_cm
    w, h = maxx - minx, maxy - miny
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{w:.2f}cm" height="{h:.2f}cm" '
        f'viewBox="{minx:.3f} {-maxy:.3f} {w:.3f} {h:.3f}">',
        f'<rect x="{minx:.3f}" y="{-maxy:.3f}" width="{w:.3f}" '
        f'height="{h:.3f}" fill="white" stroke="#cccccc" stroke-width="0.02"/>',
    ]
    for i, (x, y) in enumerate(waypoints):
        color = "red" if i == 0 else ("blue" if i == len(waypoints)-1 else "black")
        parts.append(f'<circle cx="{x:.3f}" cy="{-y:.3f}" r="{dot_radius:.3f}" fill="{color}"/>')
    parts.append("</svg>")
    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  Wrote waypoints visualization: {svg_path}")


def image_to_simple_waypoints(
    image_path,
    max_points=MAX_WAYPOINTS,
    box_cm=DRAW_BOX_CM,
    center=(DRAW_CENTER_X, DRAW_CENTER_Y),
    write_svg=True,
):
    _linedraw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")
    if _linedraw_dir not in sys.path:
        sys.path.insert(0, _linedraw_dir)
    import linedraw  # pyright: ignore[reportMissingImports]
    polylines = linedraw.vectorise(
        image_path, resolution=512,
        draw_contours=2, repeat_contours=1,
        draw_hatch=False, repeat_hatch=0,
        svg_folder="images/",
    )
    polylines = [p for p in polylines if len(p) >= 2]
    if not polylines:
        raise ValueError(f"No contours found in {image_path}.")
    tour = []
    for poly in polylines:
        tour.extend(poly)
    print(f"  Combined {len(polylines)} contours into a {len(tour)}-point tour.")
    waypoints_px = _resample_polyline(tour, max_points)
    xs = [p[0] for p in waypoints_px]
    ys = [p[1] for p in waypoints_px]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = box_cm / max(maxx - minx, maxy - miny, 1e-6)
    cx_img, cy_img = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    cx, cy = center
    out = [
        (cx + (px - cx_img) * scale, cy - (py - cy_img) * scale)
        for px, py in waypoints_px
    ]
    if write_svg:
        write_waypoints_svg(out, os.path.splitext(image_path)[0] + WAYPOINTS_SVG_SUFFIX)
    return out


def trace_waypoints(a0, a1, waypoints):
    if not waypoints:
        print("  No waypoints to trace.")
        return a0, a1
    print(f"Tracing {len(waypoints)} waypoints:")
    for i, (x, y) in enumerate(waypoints):
        print(f"  [{i+1}/{len(waypoints)}] -> ({x:.2f}, {y:.2f}) cm")
        a0, a1 = goto_xy(a0, a1, x, y)
    return a0, a1


def trace_image(a0, a1, image_path=None):
    if image_path is None:
        try:
            raw = input(f"Image path [{DEFAULT_IMAGE_PATH}]: ").strip()
        except EOFError:
            raw = ""
        image_path = raw or DEFAULT_IMAGE_PATH
    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        return a0, a1
    try:
        waypoints = image_to_simple_waypoints(image_path)
    except Exception as e:
        print(f"  Failed to vectorise image: {e}")
        return a0, a1
    return trace_waypoints(a0, a1, waypoints)


# ==========================================================
# 4. Main loop — keyboard control
# ==========================================================
angle0 = HOME_0
angle1 = HOME_1
set_servo_angle(SERVO_CH_0, angle0)
set_servo_angle(SERVO_CH_1, angle1)

print("\nControls:")
print("  r       home position")
print("  x       test step (servo0 +30°, servo1 +20°)")
print("  s       draw square")
print("  t       draw star")
print("  g       goto (x, y) cm")
print("  i       trace image waypoints")
print("  c       calibration tool  ← run this first!")
print("  q       quit")
print(f"\nHome pen: ({HOME_X:.2f}, {HOME_Y:.2f}) cm   L1={L1}  L2={L2}")
print(f"Hysteresis corrections: servo0={HYSTERESIS[0]}, servo1={HYSTERESIS[1]} pulse counts")
print("Tip: draw a box with 's', then tune HYSTERESIS until opposite sides are equal.\n")

try:
    while True:
        ch = read_key()
        if ch in ("\x03", "q", "Q"):
            break
        elif ch in ("r", "R"):
            move_servo_smooth(SERVO_CH_0, angle0, HOME_0)
            angle0 = HOME_0
            move_servo_smooth(SERVO_CH_1, angle1, HOME_1)
            angle1 = HOME_1
            print(f"Home: servo0={angle0}°, servo1={angle1}°")
        elif ch in ("x", "X"):
            target0 = clamp_angle(angle0 + 30)
            move_servo_smooth(SERVO_CH_0, angle0, target0)
            angle0 = target0
            time.sleep(2)
            target1 = clamp_angle(angle1 + 20)
            move_servo_smooth(SERVO_CH_1, angle1, target1)
            angle1 = target1
            print(f"Step: servo0={angle0}°, servo1={angle1}°")
        elif ch in ("s", "S"):
            angle0, angle1 = draw_square(angle0, angle1)
            print(f"Square done: servo0={angle0}°, servo1={angle1}°")
        elif ch in ("t", "T"):
            angle0, angle1 = draw_star(angle0, angle1)
            print(f"Star done: servo0={angle0}°, servo1={angle1}°")
        elif ch in ("g", "G"):
            target = prompt_xy()
            if target is not None:
                angle0, angle1 = goto_xy(angle0, angle1, *target)
                print(f"Goto done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
        elif ch in ("i", "I"):
            angle0, angle1 = trace_image(angle0, angle1)
            print(f"Trace done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
        elif ch in ("c", "C"):
            capture_calibration()

except KeyboardInterrupt:
    pass
finally:
    print("\nShutting down...")
    board.shutdown()
