import math
import os
import sys
import termios
import threading
import tty
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kinematics            # pyright: ignore[reportMissingImports]
import polyline              # pyright: ignore[reportMissingImports]
import svg_io                # pyright: ignore[reportMissingImports]
from calibration_tool import capture_calibration as _capture_calibration  # pyright: ignore[reportMissingImports]

# Re-export so the rest of the file can keep its old names.
clamp_angle         = kinematics.clamp_angle
xy_to_servo_angles  = kinematics.xy_to_servo_angles
is_xy_reachable     = kinematics.is_xy_reachable
compute_drawing_box = kinematics.compute_drawing_box

# ---------------------------------------------------------
# Simulation flag — set PLOTTER_SIM=1 in env to skip the Arduino and run
# the live matplotlib simulation instead. Everything else (calibration,
# IK, image tracing, hysteresis) runs identically in either mode.
# ---------------------------------------------------------
SIM_MODE = os.environ.get("PLOTTER_SIM", "0") == "1"

if SIM_MODE:
    import queue
    import sim_driver  # pyright: ignore[reportMissingImports]
    print("Running in SIMULATION mode (no hardware).")
    board = sim_driver.SimBoard()
    pca = sim_driver.SimPCA9685()
    command_queue = queue.Queue()
else:
    command_queue = None
    from telemetrix import telemetrix
    from telemetrix_pca9685 import telemetrix_pca9685
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

HOME_0 = 90
HOME_1 = 90

# --- PEN CONFIGURATION ---
# Tune these raw pulse values to match your physical setup!
PEN_UP_PULSE = 300   
PEN_DOWN_PULSE = 400 
PEN_DELAY = 0.15     # Seconds to wait for the pen to finish moving

# Ramp tuning (PULSE_DELAY = seconds between micro-steps; bigger = slower)
STEP_DEGREES = 2
PULSE_DELAY = 0.077

# Shape parameters
STAR_RADIUS_DEG = 18

# Cartesian corners traced by `draw_square` (cm), in stroke order. A 6x4 cm
# rectangle centered on the auto-computed safe-box centroid; every corner
# stays comfortably inside the [0, 90] / [40, 160] joint range.
SQUARE_CORNERS_CM = [
    ( 4.0, 17.0),    # top-right (start)
    (-2.0, 17.0),    # top-left
    (-2.0, 13.0),    # bottom-left
    ( 4.0, 13.0),    # bottom-right
]

# ---------------------------------------------------------
# Link lengths in cm
# ---------------------------------------------------------
L1 = 10.3
L2 = 10.9
HOME_X = 10.3
HOME_Y = 10.9
K0 = -1
K1 = -1
ELBOW_UP = True

# ---------------------------------------------------------
# Servo joint limits (the safe mechanical workspace)
# ---------------------------------------------------------
SERVO0_MIN_DEG = 0.0     # shoulder
SERVO0_MAX_DEG = 90.0
SERVO1_MIN_DEG = 40.0    # elbow
SERVO1_MAX_DEG = 160.0
JOINT_MARGIN_DEG = 3.0   # stay this far inside the limits when sizing the box

kinematics.configure(
    L1=L1, L2=L2, K0=K0, K1=K1, elbow_up=ELBOW_UP,
    joint_limits=[(SERVO0_MIN_DEG, SERVO0_MAX_DEG),
                  (SERVO1_MIN_DEG, SERVO1_MAX_DEG)],
    joint_margin_deg=JOINT_MARGIN_DEG,
)

# ---------------------------------------------------------
# Image -> strokes feature
# ---------------------------------------------------------
# Set USE_AUTO_BOX=True to derive the drawing box from the joint limits
# above. Set False to use the hand-tuned values below verbatim.
USE_AUTO_BOX = True
DRAW_CENTER_X = 6.0
DRAW_CENTER_Y = 12.0
DRAW_BOX_W_CM = 6.0
DRAW_BOX_H_CM = 6.0

IMAGES_DIR = "images"
DEFAULT_IMAGE_PATH = "images/banana.png"
STROKES_SVG_SUFFIX = ".strokes.svg"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg")

# ==========================================================
# SERVO CALIBRATION
# ==========================================================
# PLACEHOLDER (linear approximation) — replace after calibration:
SERVO0_CAL = [
    (-10.0, 232),
    (0.0, 244),
    (10.0, 258),
    (20.0, 272),
    (30.0, 285),
    (40.0, 298),
    (50.0, 314),
    (60.0, 331),
    (70.0, 348),
    (80.0, 365),
    (90.0, 384),
]

SERVO1_CAL = [
    (40.0, 302),
    (50.0, 316),
    (60.0, 332),
    (70.0, 347),
    (80.0, 364),
    (90.0, 374),
    (100.0, 387),
    (110.0, 403),
    (120.0, 416),
    (130.0, 432),
    (140.0, 448),
    (150.0, 462),
    (160.0, 478),
]

# Hysteresis correction (pulse counts, one per servo)
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
        return np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 1))
    else:
        raise ValueError("Need at least 2 calibration points per servo.")

_cal_poly_0 = _build_cal_poly(SERVO0_CAL)
_cal_poly_1 = _build_cal_poly(SERVO1_CAL)

_prev_pulse   = [0, 0]       
_active_hyst  = [0.0, 0.0]  

PULSE_MIN = 100
PULSE_MAX = 650

# ---------------------------------------------------------
# Hand off all the constants to the simulation driver so it can invert the
# calibration polynomials, run forward kinematics, and label its plot
# panels with the right joint limits and drawing-box rectangle.
# ---------------------------------------------------------
if SIM_MODE:
    sim_driver.configure(
        L1=L1, L2=L2, K0=K0, K1=K1,
        home_angles=(HOME_0, HOME_1),
        servo_ch_0=SERVO_CH_0, servo_ch_1=SERVO_CH_1, servo_ch_pen=SERVO_CH_PEN,
        pen_up_pulse=PEN_UP_PULSE, pen_down_pulse=PEN_DOWN_PULSE,
        servo_cal=[SERVO0_CAL, SERVO1_CAL],
        joint_limits=[(SERVO0_MIN_DEG, SERVO0_MAX_DEG),
                      (SERVO1_MIN_DEG, SERVO1_MAX_DEG)],
        # Tell the sim what hysteresis the production driver is adding so it
        # can be subtracted before inverse calibration -> no fake jitter
        # on direction reversals.
        prod_hysteresis_pulse=list(HYSTERESIS),
        # Realism dials — tweak to study errors. 0 = ideal sim.
        sim_hysteresis_pulse=[0, 0],
        sim_servo_noise_deg=0.0,
        sim_tau_s=0.0,
    )

# ==========================================================
# Pen Control
# ==========================================================
def pen_up():
    """Lift the pen off the paper."""
    pca.set_pwm(SERVO_CH_PEN, 0, PEN_UP_PULSE)
    time.sleep(PEN_DELAY)

def pen_down():
    """Drop the pen onto the paper."""
    pca.set_pwm(SERVO_CH_PEN, 0, PEN_DOWN_PULSE)
    time.sleep(PEN_DELAY)

# ==========================================================
# Core servo driver
# ==========================================================
def set_servo_angle(channel, angle):
    angle = float(clamp_angle(angle))

    poly = _cal_poly_0 if channel == SERVO_CH_0 else _cal_poly_1
    raw_pulse = float(poly(angle))

    hyst = HYSTERESIS[channel]
    if raw_pulse > _prev_pulse[channel] + 0.01:
        _active_hyst[channel] = +hyst
    elif raw_pulse < _prev_pulse[channel] - 0.01:
        _active_hyst[channel] = -hyst

    _prev_pulse[channel] = raw_pulse
    corrected = raw_pulse + _active_hyst[channel]

    pulse = int(round(max(PULSE_MIN, min(PULSE_MAX, corrected))))
    pca.set_pwm(channel, 0, pulse)

# ==========================================================
# Calibration capture tool — thin wrapper over calibration_tool module.
# ==========================================================
def capture_calibration():
    def _install(p0, p1):
        global _cal_poly_0, _cal_poly_1
        _cal_poly_0 = p0
        _cal_poly_1 = p1
    _capture_calibration(
        pca=pca, read_key=read_key, build_cal_poly=_build_cal_poly,
        servo_ch_0=SERVO_CH_0, servo_ch_1=SERVO_CH_1,
        prev_pulse=_prev_pulse,
        pulse_min=PULSE_MIN, pulse_max=PULSE_MAX,
        update_polynomials=_install,
    )

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
        set_servo_angle(channel, current)
        time.sleep(PULSE_DELAY)
    set_servo_angle(channel, end)

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
        set_servo_angle(SERVO_CH_0, a0s + (a0e - a0s) * t)
        set_servo_angle(SERVO_CH_1, a1s + (a1e - a1s) * t)
        time.sleep(PULSE_DELAY)
    set_servo_angle(SERVO_CH_0, a0e)
    set_servo_angle(SERVO_CH_1, a1e)

# ==========================================================
# Shapes
# ==========================================================
def draw_square(a0, a1):
    """Trace the rectangle in `SQUARE_CORNERS_CM`. The pen is lifted while
    travelling to the first corner so no line is drawn between home (or
    wherever the pen was) and the rectangle."""
    if not SQUARE_CORNERS_CM:
        return a0, a1
    pen_up()
    x0, y0 = SQUARE_CORNERS_CM[0]
    a0, a1 = goto_xy(a0, a1, x0, y0)
    pen_down()
    # Walk the remaining corners, then close back to the starting corner.
    for x, y in SQUARE_CORNERS_CM[1:] + [SQUARE_CORNERS_CM[0]]:
        a0, a1 = goto_xy(a0, a1, x, y)
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
# Inverse kinematics & drawing-box live in `kinematics.py`.
# Convenience wrapper that handles command-loop error reporting:
# ==========================================================
def goto_xy(a0, a1, x, y):
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError as e:
        print(f"  IK error: {e}")
        return a0, a1
    s0c, s1c = clamp_angle(s0), clamp_angle(s1)
    move_both_smooth(a0, a1, s0c, s1c)
    return s0c, s1c

def trace_box_outline(a0, a1):
    """Move the pen around the perimeter of the current drawing box
    so the user can verify that the workspace is correctly mapped."""
    cx, cy = DRAW_CENTER_X, DRAW_CENTER_Y
    hw, hh = DRAW_BOX_W_CM / 2, DRAW_BOX_H_CM / 2
    corners = [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
        (cx - hw, cy - hh),
    ]
    pen_up()
    a0, a1 = goto_xy(a0, a1, *corners[0])
    pen_down()
    for x, y in corners[1:]:
        a0, a1 = goto_xy(a0, a1, x, y)
    pen_up()
    return a0, a1

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
# Polyline math + SVG I/O live in `polyline.py` and `svg_io.py`. Aliases
# below keep the existing call sites in `image_to_strokes` unchanged.
_resample_polyline   = polyline.resample_polyline
_remove_backtracking = polyline.remove_backtracking
write_strokes_svg    = svg_io.write_strokes_svg
_polylines_from_svg  = svg_io.polylines_from_svg


def image_to_strokes(
    image_path,
    max_points_per_stroke=80,
    box_w_cm=None,
    box_h_cm=None,
    center=None,
    write_svg=True,
):
    if box_w_cm is None:
        box_w_cm = DRAW_BOX_W_CM
    if box_h_cm is None:
        box_h_cm = DRAW_BOX_H_CM
    if center is None:
        center = (DRAW_CENTER_X, DRAW_CENTER_Y)

    if image_path.lower().endswith(".svg"):
        print(f"  Loading waypoints directly from SVG: {image_path}")
        polylines = _polylines_from_svg(image_path)
    else:
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
    img_w = max(maxx - minx, 1e-6)
    img_h = max(maxy - miny, 1e-6)
    # Preserve aspect ratio while fitting inside the (w x h) drawing box.
    scale = min(box_w_cm / img_w, box_h_cm / img_h)
    cx_img, cy_img = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    cx, cy = center

    strokes_cm = []
    n_clipped = 0
    for poly in polylines:
        poly_cm = []
        for px, py in poly:
            x = cx + (px - cx_img) * scale
            y = cy - (py - cy_img) * scale
            if not is_xy_reachable(x, y, margin_deg=0.0):
                n_clipped += 1
                # Clamp to the box just in case rounding pushes us out
                x = max(cx - box_w_cm / 2, min(cx + box_w_cm / 2, x))
                y = max(cy - box_h_cm / 2, min(cy + box_h_cm / 2, y))
            poly_cm.append((x, y))
        strokes_cm.append(poly_cm)
    if n_clipped:
        print(f"  Note: clamped {n_clipped} point(s) to stay inside the workspace.")

    # Split at U-turns and drop sections that retrace already-drawn segments.
    pts_before = sum(len(s) for s in strokes_cm)
    deduped = []
    for poly in strokes_cm:
        deduped.extend(_remove_backtracking(poly, angle_threshold_deg=150.0,
                                            overlap_eps=0.2))
    strokes_cm = deduped
    pts_after = sum(len(s) for s in strokes_cm)
    if pts_after != pts_before:
        print(f"  Removed backtracking: {pts_before - pts_after} redundant point(s) "
              f"-> {len(strokes_cm)} strokes.")

    # Resample any oversize strokes.
    for i, poly in enumerate(strokes_cm):
        if len(poly) > max_points_per_stroke:
            strokes_cm[i] = _resample_polyline(poly, max_points_per_stroke)

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

def list_available_images(images_dir=IMAGES_DIR):
    """Return a sorted list of source image files in `images_dir`.
    Excludes generated artifacts like `*.strokes.svg`."""
    if not os.path.isdir(images_dir):
        return []
    files = []
    for name in os.listdir(images_dir):
        low = name.lower()
        if not low.endswith(IMAGE_EXTS):
            continue
        if low.endswith(".strokes.svg"):
            continue
        files.append(os.path.join(images_dir, name))
    return sorted(files)

def select_image_path():
    """Prompt user to pick an image from `images/` by number, or type a path."""
    images = list_available_images()
    if images:
        print("\nAvailable images:")
        for i, path in enumerate(images, 1):
            marker = " (default)" if path == DEFAULT_IMAGE_PATH else ""
            print(f"  {i}. {path}{marker}")
        prompt = f"Select # or path [{DEFAULT_IMAGE_PATH}]: "
    else:
        print(f"\n(No images found in {IMAGES_DIR}/)")
        prompt = f"Image path [{DEFAULT_IMAGE_PATH}]: "

    try:
        raw = input(prompt).strip()
    except EOFError:
        raw = ""

    if not raw:
        return DEFAULT_IMAGE_PATH

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(images):
            return images[idx - 1]
        print(f"  Number out of range (1..{len(images)}).")
        return None

    return raw

def trace_image(a0, a1, image_path=None):
    if image_path is None:
        image_path = select_image_path()
        if image_path is None:
            return a0, a1
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

# Initialize pen up and joints to home
pen_up()
set_servo_angle(SERVO_CH_0, angle0)
set_servo_angle(SERVO_CH_1, angle1)

# ---------------------------------------------------------
# Compute the safe drawing box from the joint limits
# ---------------------------------------------------------
if USE_AUTO_BOX:
    print("\nComputing safe drawing box from joint limits...")
    print(f"  shoulder (servo0): [{SERVO0_MIN_DEG:.0f}°, {SERVO0_MAX_DEG:.0f}°]")
    print(f"  elbow    (servo1): [{SERVO1_MIN_DEG:.0f}°, {SERVO1_MAX_DEG:.0f}°]")
    box = compute_drawing_box()
    if box is not None:
        cx_box, cy_box, hw_box, hh_box = box
        DRAW_CENTER_X = cx_box
        DRAW_CENTER_Y = cy_box
        DRAW_BOX_W_CM = 2 * hw_box
        DRAW_BOX_H_CM = 2 * hh_box
    else:
        print("  Warning: empty reachable area, keeping hand-tuned box.")

print("\nDrawing box (cm):")
print(f"  center=({DRAW_CENTER_X:.2f}, {DRAW_CENTER_Y:.2f})  "
      f"size={DRAW_BOX_W_CM:.2f} x {DRAW_BOX_H_CM:.2f}")
print(f"  x: [{DRAW_CENTER_X - DRAW_BOX_W_CM/2:.2f}, {DRAW_CENTER_X + DRAW_BOX_W_CM/2:.2f}]")
print(f"  y: [{DRAW_CENTER_Y - DRAW_BOX_H_CM/2:.2f}, {DRAW_CENTER_Y + DRAW_BOX_H_CM/2:.2f}]")

if SIM_MODE:
    sim_driver.configure(
        draw_box=(DRAW_CENTER_X, DRAW_CENTER_Y, DRAW_BOX_W_CM, DRAW_BOX_H_CM),
    )

print("\nControls:")
print("  r       home position")
print("  x       test step (servo0 +30°, servo1 +20°)")
print("  s       draw square")
print("  t       draw star")
print("  g       goto (x, y) cm")
print("  i       trace image waypoints (inside drawing box)")
print("  b       trace drawing-box outline (verify workspace)")
print("  u       pen up (test/tune)")
print("  d       pen down (test/tune)")
print("  c       calibration tool  ← run this first!")
print("  e       erase the simulated page (sim only)")
print("  q       quit")
print(f"\nHome pen: ({HOME_X:.2f}, {HOME_Y:.2f}) cm   L1={L1}  L2={L2}")
print(f"Hysteresis corrections: servo0={HYSTERESIS[0]}, servo1={HYSTERESIS[1]} pulse counts")

# ==========================================================
# Command dispatcher — used by both keyboard and the matplotlib UI
# ==========================================================
def dispatch_command(cmd):
    """Execute a single command tuple. Returns False if the command was
    'quit' (signalling the caller to break its loop)."""
    global angle0, angle1
    kind = cmd[0]
    args = cmd[1:]

    if kind == "quit":
        return False
    elif kind == "home":
        pen_up()
        move_servo_smooth(SERVO_CH_0, angle0, HOME_0); angle0 = HOME_0
        move_servo_smooth(SERVO_CH_1, angle1, HOME_1); angle1 = HOME_1
        print(f"Home: servo0={angle0}°, servo1={angle1}°")
    elif kind == "test_step":
        target0 = clamp_angle(angle0 + 30)
        move_servo_smooth(SERVO_CH_0, angle0, target0); angle0 = target0
        time.sleep(2)
        target1 = clamp_angle(angle1 + 20)
        move_servo_smooth(SERVO_CH_1, angle1, target1); angle1 = target1
        print(f"Step: servo0={angle0}°, servo1={angle1}°")
    elif kind == "square":
        angle0, angle1 = draw_square(angle0, angle1)
        print(f"Square done: servo0={angle0}°, servo1={angle1}°")
    elif kind == "star":
        angle0, angle1 = draw_star(angle0, angle1)
        print(f"Star done: servo0={angle0}°, servo1={angle1}°")
    elif kind == "goto":
        x, y = args
        pen_up()
        angle0, angle1 = goto_xy(angle0, angle1, x, y)
        print(f"Goto done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
    elif kind == "trace_image":
        path = args[0] if args else None
        angle0, angle1 = trace_image(angle0, angle1, image_path=path)
        print(f"Trace done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
    elif kind == "trace_box":
        angle0, angle1 = trace_box_outline(angle0, angle1)
        print(f"Box outline done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
    elif kind == "pen_up":
        pen_up()
        print("Pen Up")
    elif kind == "pen_down":
        pen_down()
        print("Pen Down")
    elif kind == "calibrate":
        capture_calibration()
    elif kind == "erase":
        if SIM_MODE:
            sim_driver.get_recorder().erase()
            print("Erased the page.")
        else:
            print("Erase: only meaningful in sim mode (real paper "
                  "isn't quite that magical yet).")
    else:
        print(f"  Unknown command: {cmd!r}")
    return True


def _key_to_command(ch):
    """Translate a single keystroke into a command tuple. May call input()
    for keys that need extra arguments (g, i). Returns None if the key
    is not bound or the user aborted the prompt."""
    if ch in ("\x03", "q", "Q"):
        return ("quit",)
    if ch in ("r", "R"):
        return ("home",)
    if ch in ("x", "X"):
        return ("test_step",)
    if ch in ("s", "S"):
        return ("square",)
    if ch in ("t", "T"):
        return ("star",)
    if ch in ("g", "G"):
        target = prompt_xy()
        if target is None:
            return None
        return ("goto", target[0], target[1])
    if ch in ("i", "I"):
        path = select_image_path()
        if path is None:
            return None
        return ("trace_image", path)
    if ch in ("b", "B"):
        return ("trace_box",)
    if ch in ("u", "U"):
        return ("pen_up",)
    if ch in ("d", "D"):
        return ("pen_down",)
    if ch in ("c", "C"):
        return ("calibrate",)
    if ch in ("e", "E"):
        return ("erase",)
    return None


def run_keyboard_loop():
    """Read keystrokes and either dispatch directly (real mode) or push to
    the command queue (sim mode, where the UI also produces commands)."""
    try:
        while True:
            ch = read_key()
            cmd = _key_to_command(ch)
            if cmd is None:
                continue
            if SIM_MODE:
                command_queue.put(cmd)
                if cmd[0] == "quit":
                    break
            else:
                if not dispatch_command(cmd):
                    break
    except KeyboardInterrupt:
        if SIM_MODE:
            command_queue.put(("quit",))
    finally:
        if not SIM_MODE:
            print("\nShutting down...")
            try: pen_up()
            except Exception: pass
            try: board.shutdown()
            except Exception: pass


def run_command_worker():
    """Sim-mode only: consume commands from the queue and execute them."""
    try:
        while True:
            cmd = command_queue.get()
            try:
                if not dispatch_command(cmd):
                    break
            except Exception as e:
                print(f"  Command error ({cmd[0]!r}): {e}")
            finally:
                command_queue.task_done()
    finally:
        print("\nShutting down...")
        try: pen_up()
        except Exception: pass
        try: board.shutdown()
        except Exception: pass
        sim_driver.request_shutdown()


if SIM_MODE:
    # In sim mode: keyboard thread + worker thread + main thread for matplotlib.
    print("\n[sim] Window will open with controls. Click buttons in the "
          "figure or type commands in the terminal.")
    kb_thread = threading.Thread(target=run_keyboard_loop, daemon=True)
    worker_thread = threading.Thread(target=run_command_worker, daemon=True)
    kb_thread.start()
    worker_thread.start()
    try:
        sim_driver.start_animation(
            command_queue=command_queue,
            available_images=list_available_images(),
            default_image=DEFAULT_IMAGE_PATH,
        )
    finally:
        sim_driver.request_shutdown()
        # Nudge the worker to exit if it's blocked on Queue.get()
        try:
            command_queue.put_nowait(("quit",))
        except Exception:
            pass
        try:
            os.system("stty sane 2>/dev/null")
        except Exception:
            pass
else:
    run_keyboard_loop()