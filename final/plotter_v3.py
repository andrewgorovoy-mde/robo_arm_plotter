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
# 2. Analog Feedback Setup (Pins A0 & A1)
# ---------------------------------------------------------
# Global array to hold the real-time analog values (0-1023)
analog_pos = [0, 0]

def analog_cb(data):
    """Callback triggered automatically when A0 or A1 changes."""
    pin = data[1]
    val = data[2]
    if pin == 0:
        analog_pos[0] = val
    elif pin == 1:
        analog_pos[1] = val

print("Initializing Analog Feedback on A0 and A1...")
board.set_pin_mode_analog_input(0, callback=analog_cb)
board.set_pin_mode_analog_input(1, callback=analog_cb)

# ---------------------------------------------------------
# 3. Initialize the PCA9685 Servo Driver
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
SERVO_CH_PEN = 2

HOME_0 = 90
HOME_1 = 90

# --- PEN CONFIGURATION ---
PEN_UP_PULSE = 300   
PEN_DOWN_PULSE = 400 
PEN_DELAY = 0.15     

# Ramp tuning
STEP_DEGREES = 2
PULSE_DELAY = 0.055

# Shape parameters
SQUARE_EDGE_DEG = 18
STAR_RADIUS_DEG = 18

# ---------------------------------------------------------
# Link lengths in cm
# ---------------------------------------------------------
L1 = 8.5
L2 = 12
HOME_X = 8.5
HOME_Y = 12
K0 = -1
K1 = 1
ELBOW_UP = True

# ---------------------------------------------------------
# Image -> strokes feature
# ---------------------------------------------------------
DRAW_CENTER_X = 12.6
DRAW_CENTER_Y = 14.5
DRAW_BOX_CM = 6.0
DEFAULT_IMAGE_PATH = "images/banana.png"
STROKES_SVG_SUFFIX = ".strokes.svg"

def _build_cal_poly(cal_points):
    """Fit a cubic polynomial to calibration data."""
    arr = np.array(cal_points, dtype=float)
    if len(arr) >= 4:
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 3))
    elif len(arr) >= 2:
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 1))
    else:
        raise ValueError("Need at least 2 calibration points per servo.")

# ==========================================================
# SERVO CALIBRATION (PWM Polynomials)
# ==========================================================
SERVO0_CAL = [
    (0.0,   150),
    (45.0,  262),
    (90.0,  375),
    (135.0, 490),
    (180.0, 608),
]

SERVO1_CAL = [
    (0.0,   150),
    (45.0,  262),
    (90.0,  375),
    (135.0, 490),
    (180.0, 608),
]

_cal_poly_0 = _build_cal_poly(SERVO0_CAL)
_cal_poly_1 = _build_cal_poly(SERVO1_CAL)

# ==========================================================
# CLOSED-LOOP ANALOG CALIBRATION (Non-linear Polynomials)
# ==========================================================
ANALOG0_CAL = [
    (0.0,   150),
    (45.0,  325),
    (90.0,  500),
    (135.0, 680),
    (180.0, 850),
]

ANALOG1_CAL = [
    (0.0,   150),
    (45.0,  325),
    (90.0,  500),
    (135.0, 680),
    (180.0, 850),
]

_analog_poly_0 = _build_cal_poly(ANALOG0_CAL)
_analog_poly_1 = _build_cal_poly(ANALOG1_CAL)

def degrees_to_analog(channel, deg):
    """Map degrees to a target analog value using a cubic polynomial fit."""
    poly = _analog_poly_0 if channel == SERVO_CH_0 else _analog_poly_1
    return float(poly(deg))

# Closed-loop gain. Higher = faster correction, but too high = jitters/oscillation.
KP_GAIN = 0.8 

PULSE_MIN = 100
PULSE_MAX = 650
_prev_pulse = [375, 375]

# ==========================================================
# Pen Control
# ==========================================================
def pen_up():
    pca.set_pwm(SERVO_CH_PEN, 0, PEN_UP_PULSE)
    time.sleep(PEN_DELAY)

def pen_down():
    pca.set_pwm(SERVO_CH_PEN, 0, PEN_DOWN_PULSE)
    time.sleep(PEN_DELAY)

# ==========================================================
# Core CLOSED-LOOP servo driver
# ==========================================================
def update_servo_closed_loop(channel, target_angle):
    target_angle = float(clamp_angle(target_angle))
    
    # 1. Feed-Forward Base: Make a strong initial guess using the polynomials
    poly = _cal_poly_0 if channel == SERVO_CH_0 else _cal_poly_1
    base_pwm = float(poly(target_angle))

    # 2. Read Target and Current Analog states
    target_analog = degrees_to_analog(channel, target_angle)
    current_analog = analog_pos[channel]

    # If the analog pin hasn't reported back yet, fall back to pure open-loop
    if current_analog == 0:
        pulse = int(round(max(PULSE_MIN, min(PULSE_MAX, base_pwm))))
        pca.set_pwm(channel, 0, pulse)
        _prev_pulse[channel] = pulse
        return

    # 3. Proportional Correction
    error = target_analog - current_analog
    
    # Apply error correction ON TOP of the polynomial guess.
    # This automatically destroys hysteresis without windup.
    corrected_pwm = base_pwm + (error * KP_GAIN)
    
    pulse = int(round(max(PULSE_MIN, min(PULSE_MAX, corrected_pwm))))
    pca.set_pwm(channel, 0, pulse)
    _prev_pulse[channel] = pulse

# ==========================================================
# Calibration capture tool
# ==========================================================
def capture_calibration():
    print("\n=== CALIBRATION MODE ===")
    print("Drive each servo to known angles (e.g. 0, 45, 90, 135, 180), then SPACE to record.")
    print("a/s=servo0 ±10  A/S=servo0 ±1  k/l=servo1 ±10  K/L=servo1 ±1")
    print("SPACE=record  v=view so far  0=finish\n")

    pw0 = _prev_pulse[SERVO_CH_0]
    pw1 = _prev_pulse[SERVO_CH_1]

    def _raw(ch, pw):
        pw = int(max(PULSE_MIN, min(PULSE_MAX, pw)))
        pca.set_pwm(ch, 0, pw)

    _raw(SERVO_CH_0, pw0)
    _raw(SERVO_CH_1, pw1)

    cal0 = []  # Stores tuples of (angle, pwm, analog)
    cal1 = []  

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
            time.sleep(0.05) # Wait for servo to move to get fresh analog read
            print(f"  servo0: PWM={pw0} | Analog={analog_pos[0]}    servo1: PWM={pw1} | Analog={analog_pos[1]}")

        elif ch == " ":
            raw = input("\n  Recording. Which servo did you just position? (0 / 1 / both): ").strip()
            if raw in ("0", "1", "both"):
                try:
                    angle_raw = input("  Physical angle in degrees: ").strip()
                    angle = float(angle_raw)
                except ValueError:
                    print("  Bad angle — skipping.")
                    continue
                if raw in ("0", "both"):
                    cal0.append((angle, pw0, analog_pos[0]))
                    print(f"  ✓ servo0: {angle}° → PWM {pw0} | Analog {analog_pos[0]}")
                if raw in ("1", "both"):
                    cal1.append((angle, pw1, analog_pos[1]))
                    print(f"  ✓ servo1: {angle}° → PWM {pw1} | Analog {analog_pos[1]}")
            else:
                print("  Enter 0, 1, or both.")

        elif ch == "v":
            print(f"\n  servo0 so far: {sorted(cal0)}")
            print(f"  servo1 so far: {sorted(cal1)}\n")

        elif ch == "0":
            break

    print("\n=== CALIBRATION RESULTS — paste into your script ===\n")
    
    print("SERVO0_CAL = [")
    for a, p, an in sorted(cal0):
        print(f"    ({a:.1f}, {p}),")
    print("]\n")
    
    print("SERVO1_CAL = [")
    for a, p, an in sorted(cal1):
        print(f"    ({a:.1f}, {p}),")
    print("]\n")

    print("ANALOG0_CAL = [")
    for a, p, an in sorted(cal0):
        print(f"    ({a:.1f}, {an}),")
    print("]\n")

    print("ANALOG1_CAL = [")
    for a, p, an in sorted(cal1):
        print(f"    ({a:.1f}, {an}),")
    print("]\n")

    if len(cal0) >= 2 and len(cal1) >= 2:
        print("Fitting polynomials to captured data...")
        global _cal_poly_0, _cal_poly_1, _analog_poly_0, _analog_poly_1
        _cal_poly_0 = _build_cal_poly([(a, p) for a, p, an in cal0])
        _cal_poly_1 = _build_cal_poly([(a, p) for a, p, an in cal1])
        _analog_poly_0 = _build_cal_poly([(a, an) for a, p, an in cal0])
        _analog_poly_1 = _build_cal_poly([(a, an) for a, p, an in cal1])
        print("  ✓ Calibration active for this session.\n")
    else:
        print("  Not enough points to fit (need ≥ 2 per servo) — keeping old cal.\n")

    print("=== END CALIBRATION ===\n")

# ==========================================================
# Motion helpers
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
        update_servo_closed_loop(channel, current)
        time.sleep(PULSE_DELAY)
    update_servo_closed_loop(channel, end)

def move_both_smooth(a0s, a1s, a0e, a1e):
    a0s, a1s = float(clamp_angle(a0s)), float(clamp_angle(a1s))
    a0e, a1e = float(clamp_angle(a0e)), float(clamp_angle(a1e))
    d0 = abs(a0e - a0s)
    d1 = abs(a1e - a1s)
    if d0 < 0.01 and d1 < 0.01:
        return
    steps = max(1, int(round(max(d0, d1) / STEP_DEGREES)))
    for i in range(1, steps + 1):
        t = i / steps
        update_servo_closed_loop(SERVO_CH_0, a0s + (a0e - a0s) * t)
        update_servo_closed_loop(SERVO_CH_1, a1s + (a1e - a1s) * t)
        time.sleep(PULSE_DELAY)
    update_servo_closed_loop(SERVO_CH_0, a0e)
    update_servo_closed_loop(SERVO_CH_1, a1e)

# ==========================================================
# Shapes
# ==========================================================
def draw_square(a0, a1):
    pen_down()
    edge = SQUARE_EDGE_DEG
    for d0, d1 in [(+edge, 0), (0, +edge), (-edge, 0), (0, -edge)]:
        na0 = clamp_angle(a0 + d0)
        na1 = clamp_angle(a1 + d1)
        move_both_smooth(a0, a1, na0, na1)
        a0, a1 = na0, na1
    pen_up()
    return a0, a1

def draw_star(a0, a1):
    pen_down()
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
    pen_up()
    return cx, cy

# ==========================================================
# Inverse kinematics
# ==========================================================
def xy_to_servo_angles(x, y, elbow_up=ELBOW_UP):
    r2 = x * x + y * y
    r  = math.sqrt(r2)
    reach_min = abs(L1 - L2)
    reach_max = L1 + L2
    if r < reach_min - 1e-6 or r > reach_max + 1e-6:
        raise ValueError(f"Target ({x:.2f}, {y:.2f}) cm unreachable: |r|={r:.2f}, reach=[{reach_min:.2f}, {reach_max:.2f}]")
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
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError as e:
        print(f"  IK error: {e}")
        return a0, a1
    s0c, s1c = clamp_angle(s0), clamp_angle(s1)
    move_both_smooth(a0, a1, s0c, s1c)
    return s0c, s1c

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

# ==========================================================
# Image → Strokes (Polylines)
# ==========================================================
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

def write_strokes_svg(strokes, svg_path, padding_cm=1.0):
    if not strokes:
        return
    all_pts = [pt for s in strokes for pt in s]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
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
    
    colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
    for i, stroke in enumerate(strokes):
        color = colors[i % len(colors)]
        pts = " ".join(f"{x:.3f},{-y:.3f}" for x, y in stroke)
        parts.append(f'<polyline points="{pts}" stroke="{color}" stroke-width="0.04" fill="none" opacity="0.8"/>')
        x0, y0 = stroke[0]
        parts.append(f'<circle cx="{x0:.3f}" cy="{-y0:.3f}" r="0.05" fill="{color}"/>')
        
    parts.append("</svg>")
    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  Wrote strokes visualization: {svg_path}")

def image_to_strokes(
    image_path,
    max_points_per_stroke=80,
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
    print(f"  Found {len(polylines)} separate strokes.")

    all_pts = [pt for poly in polylines for pt in poly]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = box_cm / max(maxx - minx, maxy - miny, 1e-6)
    cx_img, cy_img = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    cx, cy = center

    strokes_cm = []
    for poly in polylines:
        poly_cm = [
            (cx + (px - cx_img) * scale, cy - (py - cy_img) * scale)
            for px, py in poly
        ]
        if len(poly_cm) > max_points_per_stroke:
            poly_cm = _resample_polyline(poly_cm, max_points_per_stroke)
        strokes_cm.append(poly_cm)

    if write_svg:
        write_strokes_svg(strokes_cm, os.path.splitext(image_path)[0] + STROKES_SVG_SUFFIX)
    return strokes_cm

def trace_strokes(a0, a1, strokes):
    if not strokes:
        print("  No strokes to trace.")
        return a0, a1
        
    print(f"Tracing {len(strokes)} strokes...")
    for i, stroke in enumerate(strokes):
        print(f"  Stroke [{i+1}/{len(strokes)}] ({len(stroke)} pts)")
        
        pen_up()
        a0, a1 = goto_xy(a0, a1, stroke[0][0], stroke[0][1])
        
        pen_down()
        for x, y in stroke[1:]:
            a0, a1 = goto_xy(a0, a1, x, y)
            
    pen_up()
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
        strokes = image_to_strokes(image_path)
    except Exception as e:
        print(f"  Failed to vectorise image: {e}")
        return a0, a1
    return trace_strokes(a0, a1, strokes)

# ==========================================================
# 4. Main loop — keyboard control
# ==========================================================
angle0 = HOME_0
angle1 = HOME_1

# Give the telemetrix analog callbacks a moment to populate initial data
time.sleep(0.5)

# Initialize pen up and joints to home
pen_up()
update_servo_closed_loop(SERVO_CH_0, angle0)
update_servo_closed_loop(SERVO_CH_1, angle1)

print("\nControls:")
print("  r       home position")
print("  x       test step (servo0 +30°, servo1 +20°)")
print("  s       draw square")
print("  t       draw star")
print("  g       goto (x, y) cm")
print("  i       trace image waypoints")
print("  u       pen up (test/tune)")
print("  d       pen down (test/tune)")
print("  c       calibration tool  ← run this first to map Analog pins!")
print("  q       quit")
print(f"\nHome pen: ({HOME_X:.2f}, {HOME_Y:.2f}) cm   L1={L1}  L2={L2}")

try:
    while True:
        ch = read_key()
        if ch in ("\x03", "q", "Q"):
            break
        elif ch in ("r", "R"):
            pen_up()
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
                pen_up()
                angle0, angle1 = goto_xy(angle0, angle1, *target)
                print(f"Goto done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
        elif ch in ("i", "I"):
            angle0, angle1 = trace_image(angle0, angle1)
            print(f"Trace done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
        elif ch in ("u", "U"):
            pen_up()
            print("Pen Up")
        elif ch in ("d", "D"):
            pen_down()
            print("Pen Down")
        elif ch in ("c", "C"):
            capture_calibration()

except KeyboardInterrupt:
    pass
finally:
    print("\nShutting down...")
    pen_up()
    board.shutdown()