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
SQUARE_EDGE_DEG = 18
STAR_RADIUS_DEG = 18

# ---------------------------------------------------------
# Link lengths in cm
# ---------------------------------------------------------
L1 = 10.2
L2 = 11.0
HOME_X = 10.2
HOME_Y = 11.0
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
# Calibration capture tool
# ==========================================================
def capture_calibration():
    print("\n=== CALIBRATION MODE ===")
    print("Drive each servo to a known angle, then SPACE to record.")
    print("a/s=servo0 ±10  A/S=servo0 ±1  k/l=servo1 ±10  K/L=servo1 ±1")
    print("SPACE=record  v=view so far  0=finish\n")

    pw0 = int(_prev_pulse[SERVO_CH_0]) or 375
    pw1 = int(_prev_pulse[SERVO_CH_1]) or 375

    def _raw(ch, pw):
        pw = int(max(PULSE_MIN, min(PULSE_MAX, pw)))
        pca.set_pwm(ch, 0, pw)

    _raw(SERVO_CH_0, pw0)
    _raw(SERVO_CH_1, pw1)

    cal0 = []  
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
            print(f"  pulses → servo0={pw0}  servo1={pw1}")

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

def is_xy_reachable(x, y, margin_deg=JOINT_MARGIN_DEG):
    """A point is reachable when IK returns servo angles inside the joint
    limits (with a safety margin) and the geometry is solvable."""
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError:
        return False
    return (SERVO0_MIN_DEG + margin_deg <= s0 <= SERVO0_MAX_DEG - margin_deg
            and SERVO1_MIN_DEG + margin_deg <= s1 <= SERVO1_MAX_DEG - margin_deg)

def _largest_true_rect(grid):
    """Largest axis-aligned all-True rectangle in a 2-D bool array.
    Returns (i0, j0, i1, j1) with inclusive indices, or None if grid is all False.
    Standard histogram method, O(rows*cols)."""
    rows, cols = grid.shape
    heights = [0] * cols
    best = None  # (area, i0, j0, i1, j1)
    for i in range(rows):
        for j in range(cols):
            heights[j] = heights[j] + 1 if grid[i, j] else 0
        # Largest rectangle in current histogram row
        stack = []  # entries: (start_col, height)
        for j in range(cols + 1):
            cur_h = heights[j] if j < cols else 0
            start = j
            while stack and stack[-1][1] > cur_h:
                left, h = stack.pop()
                area = h * (j - left)
                if best is None or area > best[0]:
                    best = (area, i - h + 1, left, i, j - 1)
                start = left
            stack.append((start, cur_h))
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]

def compute_drawing_box(margin_deg=JOINT_MARGIN_DEG, grid_step=0.2,
                        x_range=(-8.0, 22.0), y_range=(-2.0, 22.0)):
    """Find the largest axis-aligned rectangle (in cm) that lies entirely
    inside the reachable workspace defined by the joint limits.
    Returns (cx, cy, half_w, half_h)."""
    xs = np.arange(x_range[0], x_range[1] + grid_step, grid_step)
    ys = np.arange(y_range[0], y_range[1] + grid_step, grid_step)
    grid = np.zeros((len(ys), len(xs)), dtype=bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid[i, j] = is_xy_reachable(x, y, margin_deg)
    rect = _largest_true_rect(grid)
    if rect is None:
        return DRAW_CENTER_X, DRAW_CENTER_Y, DRAW_BOX_W_CM / 2, DRAW_BOX_H_CM / 2
    i0, j0, i1, j1 = rect
    x0, x1 = xs[j0], xs[j1]
    y0, y1 = ys[i0], ys[i1]
    return (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2

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
def _point_seg_dist(p, a, b):
    """Shortest distance from point `p` to the segment `a`–`b`."""
    apx, apy = p[0] - a[0], p[1] - a[1]
    abx, aby = b[0] - a[0], b[1] - a[1]
    abl2 = abx * abx + aby * aby
    if abl2 < 1e-12:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / abl2))
    cx, cy = a[0] + t * abx, a[1] + t * aby
    return math.hypot(p[0] - cx, p[1] - cy)

def _direction_change_deg(p_prev, p_cur, p_next):
    """Angle change (deg) between edges (p_prev->p_cur) and (p_cur->p_next).
    0° = colinear-forward, 180° = exact reverse (U-turn)."""
    v1x, v1y = p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]
    v2x, v2y = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(c))

def _remove_backtracking(poly, angle_threshold_deg=150.0, overlap_eps=0.2):
    """Split a polyline wherever consecutive edges reverse direction by more
    than `angle_threshold_deg` (a U-turn). For each part after a split, drop
    leading vertices that lie within `overlap_eps` of any previously emitted
    segment — those points are retracing what the pen has already drawn.
    Returns a list of cleaned polylines (each with len ≥ 2)."""
    if len(poly) < 3:
        return [list(poly)] if len(poly) >= 2 else []

    # 1. Split at U-turns.
    parts = []
    cur = [poly[0], poly[1]]
    for i in range(2, len(poly)):
        ang = _direction_change_deg(poly[i - 2], poly[i - 1], poly[i])
        if ang > angle_threshold_deg:
            if len(cur) >= 2:
                parts.append(cur)
            cur = [poly[i - 1], poly[i]]
        else:
            cur.append(poly[i])
    if len(cur) >= 2:
        parts.append(cur)

    # 2. Trim each part's leading overlap with all already-emitted segments.
    cleaned = []
    for part in parts:
        if not cleaned:
            cleaned.append(part)
            continue
        prior_segs = [(p[j], p[j + 1]) for p in cleaned for j in range(len(p) - 1)]
        k = 0
        while k < len(part) - 1:
            pt = part[k]
            if any(_point_seg_dist(pt, a, b) < overlap_eps for a, b in prior_segs):
                k += 1
            else:
                break
        trimmed = part[k:]
        if len(trimmed) >= 2:
            cleaned.append(trimmed)

    return cleaned

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

def _polylines_from_svg(svg_path):
    """Parse a simple SVG and return a list of polylines (in SVG units, with
    the SVG y-axis pointing down). Supports <polyline>, <polygon>, <line>,
    and <path> elements with only M/L/Z commands. This is intentionally tiny
    so that hand-authored "waypoint" SVGs work without extra dependencies."""
    import re
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns_match = re.match(r"\{(.*)\}", root.tag)
    ns = ns_match.group(1) if ns_match else ""
    def tag(name):
        return f"{{{ns}}}{name}" if ns else name

    def parse_points(s):
        nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)]
        return list(zip(nums[0::2], nums[1::2]))

    polylines = []
    for el in root.iter():
        t = el.tag.split("}")[-1]
        if t == "polyline":
            pts = parse_points(el.attrib.get("points", ""))
            if len(pts) >= 2:
                polylines.append(pts)
        elif t == "polygon":
            pts = parse_points(el.attrib.get("points", ""))
            if len(pts) >= 2:
                if pts[0] != pts[-1]:
                    pts = pts + [pts[0]]
                polylines.append(pts)
        elif t == "line":
            x1 = float(el.attrib.get("x1", 0)); y1 = float(el.attrib.get("y1", 0))
            x2 = float(el.attrib.get("x2", 0)); y2 = float(el.attrib.get("y2", 0))
            polylines.append([(x1, y1), (x2, y2)])
        elif t == "path":
            d = el.attrib.get("d", "")
            tokens = re.findall(r"[MLmlZz]|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", d)
            i = 0
            cur = None
            start = None
            current_poly = []
            while i < len(tokens):
                tok = tokens[i]
                if tok in ("M", "m"):
                    if current_poly and len(current_poly) >= 2:
                        polylines.append(current_poly)
                    current_poly = []
                    rel = tok == "m"
                    i += 1
                    x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                    if rel and cur is not None:
                        x += cur[0]; y += cur[1]
                    cur = (x, y); start = cur
                    current_poly.append(cur)
                    while i + 1 < len(tokens) and tokens[i] not in "MLZmlz":
                        x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                        if rel:
                            x += cur[0]; y += cur[1]
                        cur = (x, y)
                        current_poly.append(cur)
                elif tok in ("L", "l"):
                    rel = tok == "l"
                    i += 1
                    while i + 1 < len(tokens) and tokens[i] not in "MLZmlz":
                        x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                        if rel and cur is not None:
                            x += cur[0]; y += cur[1]
                        cur = (x, y)
                        current_poly.append(cur)
                elif tok in ("Z", "z"):
                    if start is not None and current_poly and current_poly[-1] != start:
                        current_poly.append(start)
                    i += 1
                else:
                    i += 1
            if current_poly and len(current_poly) >= 2:
                polylines.append(current_poly)

    return polylines

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
    cx_box, cy_box, hw_box, hh_box = compute_drawing_box()
    DRAW_CENTER_X = cx_box
    DRAW_CENTER_Y = cy_box
    DRAW_BOX_W_CM = 2 * hw_box
    DRAW_BOX_H_CM = 2 * hh_box

print("\nDrawing box (cm):")
print(f"  center=({DRAW_CENTER_X:.2f}, {DRAW_CENTER_Y:.2f})  "
      f"size={DRAW_BOX_W_CM:.2f} x {DRAW_BOX_H_CM:.2f}")
print(f"  x: [{DRAW_CENTER_X - DRAW_BOX_W_CM/2:.2f}, {DRAW_CENTER_X + DRAW_BOX_W_CM/2:.2f}]")
print(f"  y: [{DRAW_CENTER_Y - DRAW_BOX_H_CM/2:.2f}, {DRAW_CENTER_Y + DRAW_BOX_H_CM/2:.2f}]")

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
print("  q       quit")
print(f"\nHome pen: ({HOME_X:.2f}, {HOME_Y:.2f}) cm   L1={L1}  L2={L2}")
print(f"Hysteresis corrections: servo0={HYSTERESIS[0]}, servo1={HYSTERESIS[1]} pulse counts")

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
        elif ch in ("b", "B"):
            angle0, angle1 = trace_box_outline(angle0, angle1)
            print(f"Box outline done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
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