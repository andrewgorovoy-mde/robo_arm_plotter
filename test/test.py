import bisect
import math
import os
import sys
import termios
import tty
import time
from telemetrix import telemetrix
from telemetrix_pca9685 import telemetrix_pca9685

# ---------------------------------------------------------
# 1. Initialize the Main Arduino Board
# ---------------------------------------------------------
print("Connecting to Arduino...")
board = telemetrix.Telemetrix()

# Initialize the I2C bus on the Arduino (Required for PCA9685)
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
# Servo channels and home angles (tweak if needed)
# ---------------------------------------------------------
SERVO_CH_0 = 0
SERVO_CH_1 = 1
HOME_0 = 90
HOME_1 = 90

# Ramp tuning: smaller steps + longer delay = smoother motion (less pen skip)
STEP_DEGREES = 2
PULSE_DELAY = 0.055

# Square: edge length in degrees per side (servo0 = one axis, servo1 = other)
SQUARE_EDGE_DEG = 18

# Star: radius from center to each tip (degrees in joint space)
STAR_RADIUS_DEG = 18

# ---------------------------------------------------------
# Basic 2-DOF inverse kinematics
# ---------------------------------------------------------
# Link lengths in cm.
L1 = 12.6  # inner (shoulder -> elbow)
L2 = 14.5  # outer (elbow -> pen tip)

# At home (servo0=90, servo1=90) the pen sits at (HOME_X, HOME_Y).
# This means the inner arm points along +x and the outer arm points along +y
# (90 deg counter-clockwise from the inner arm).
HOME_X = 12.6
HOME_Y = 14.5

# Sign convention: depends on how each servo is mounted.
#   K0 = +1 -> increasing servo0 rotates the inner arm CCW (toward +y).
#   K1 = +1 -> increasing servo1 opens the elbow (outer arm rotates CCW
#              relative to the inner arm).
# Flip to -1 if a positive xy move drives the arm the wrong way.
K0 = +1
K1 = +1

# Use elbow-up solution (outer arm bends toward +y at home). Set False for
# the mirrored elbow-down branch.
ELBOW_UP = True

# ---------------------------------------------------------
# Image -> waypoints feature
# ---------------------------------------------------------
# Bounding box (cm) the image gets scaled into. Centred near home so the
# whole sketch stays inside the reachable annulus and we don't need to lift
# a pen between strokes.
DRAW_CENTER_X = 12.6
DRAW_CENTER_Y = 14.5
DRAW_BOX_CM = 6.0  # max edge length of the drawing
MAX_WAYPOINTS = 500
DEFAULT_IMAGE_PATH = "images/banana.png"
WAYPOINTS_SVG_SUFFIX = ".waypoints.svg"  # written next to the source image

# ---------------------------------------------------------
# 3. Define the Servo Helper Function
# ---------------------------------------------------------
def set_servo_angle(channel, angle):
    """
    Sets a servo on a specific PCA9685 channel to a specific angle (0-180).
    """
    min_pulse = 150
    max_pulse = 600
    pulse = int(min_pulse + (angle / 180.0) * (max_pulse - min_pulse))
    pca.set_pwm(channel, 0, pulse)


def read_key():
    """Single keypress without Enter (macOS/Linux)."""
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
    """Ramp from start to end in STEP_DEGREES steps with PULSE_DELAY between each."""
    start = float(clamp_angle(start))
    end = float(clamp_angle(end))
    if abs(end - start) < 0.01:
        return
    direction = 1.0 if end > start else -1.0
    current = start
    while abs(end - current) > 0.01:
        remaining = abs(end - current)
        step = min(STEP_DEGREES, remaining)
        current += direction * step
        if direction > 0 and current > end:
            current = end
        elif direction < 0 and current < end:
            current = end
        set_servo_angle(channel, current)
        time.sleep(PULSE_DELAY)
    set_servo_angle(channel, end)


def move_both_smooth(a0s, a1s, a0e, a1e):
    """Move both servos together along a straight line in (angle0, angle1) space."""
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


def draw_square(a0, a1):
    """
    Trace a small square from the current pose: +servo0, +servo1, -servo0, -servo1.
    Returns updated (a0, a1).
    """
    edge = SQUARE_EDGE_DEG
    # Edge 1: +X (servo 0)
    t = clamp_angle(a0 + edge)
    move_servo_smooth(SERVO_CH_0, a0, t)
    a0 = t
    # Edge 2: +Y (servo 1)
    t = clamp_angle(a1 + edge)
    move_servo_smooth(SERVO_CH_1, a1, t)
    a1 = t
    # Edge 3: -X
    t = clamp_angle(a0 - edge)
    move_servo_smooth(SERVO_CH_0, a0, t)
    a0 = t
    # Edge 4: -Y (back to start)
    t = clamp_angle(a1 - edge)
    move_servo_smooth(SERVO_CH_1, a1, t)
    a1 = t
    return a0, a1


def xy_to_servo_angles(x, y, elbow_up=ELBOW_UP):
    """Solve 2-link planar IK for the pen tip at (x, y) cm.

    Returns (servo0_deg, servo1_deg).

    Convention:
        theta1 = absolute angle of inner arm from +x axis
        phi    = angle of outer arm relative to inner arm
        At home: theta1 = 0, phi = 90  ->  pen at (L1, L2).

    Servo mapping (chosen so home maps to 90/90):
        servo0 = 90 + K0 * theta1
        servo1 = 90 + K1 * (phi - 90)

    Raises ValueError if the target is outside the reachable annulus.
    """
    r2 = x * x + y * y
    r = math.sqrt(r2)
    reach_min = abs(L1 - L2)
    reach_max = L1 + L2
    if r < reach_min - 1e-6 or r > reach_max + 1e-6:
        raise ValueError(
            f"Target ({x:.2f}, {y:.2f}) cm unreachable: |r|={r:.2f}, "
            f"reach=[{reach_min:.2f}, {reach_max:.2f}]"
        )

    cos_phi = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_phi = max(-1.0, min(1.0, cos_phi))
    sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
    if not elbow_up:
        sin_phi = -sin_phi

    phi = math.atan2(sin_phi, cos_phi)
    theta1 = math.atan2(y, x) - math.atan2(L2 * sin_phi, L1 + L2 * cos_phi)

    theta1_deg = math.degrees(theta1)
    phi_deg = math.degrees(phi)

    servo0 = 90.0 + K0 * theta1_deg
    servo1 = 90.0 + K1 * (phi_deg - 90.0)
    return servo0, servo1


def goto_xy(a0, a1, x, y):
    """Move pen tip to (x, y) cm using simple IK. Returns updated (a0, a1)."""
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError as e:
        print(f"  IK error: {e}")
        return a0, a1

    s0_clamped = clamp_angle(s0)
    s1_clamped = clamp_angle(s1)
    if (s0_clamped, s1_clamped) != (s0, s1):
        print(
            f"  WARN: servo angles clamped to 0-180: "
            f"raw=({s0:.2f}, {s1:.2f}) -> ({s0_clamped:.2f}, {s1_clamped:.2f})"
        )
    print(
        f"  IK: x={x:.2f} y={y:.2f} -> servo0={s0_clamped:.2f}°, "
        f"servo1={s1_clamped:.2f}°"
    )
    move_both_smooth(a0, a1, s0_clamped, s1_clamped)
    return s0_clamped, s1_clamped


def prompt_xy():
    """Prompt the user for an x,y target. Returns (x, y) or None on cancel."""
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


def _polyline_arclen(poly):
    return sum(
        math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1])
        for i in range(1, len(poly))
    )


def _resample_polyline(poly, n):
    """Resample a polyline to exactly n points, evenly spaced by arc length."""
    if n <= 1 or len(poly) <= 1:
        return [poly[0]] * max(1, n)
    cum = [0.0]
    for i in range(1, len(poly)):
        cum.append(
            cum[-1]
            + math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1])
        )
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
            seg = cum[idx] - cum[idx - 1]
            t = 0.0 if seg < 1e-9 else (target - cum[idx - 1]) / seg
            x = poly[idx - 1][0] + t * (poly[idx][0] - poly[idx - 1][0])
            y = poly[idx - 1][1] + t * (poly[idx][1] - poly[idx - 1][1])
            out.append((x, y))
    return out


def write_waypoints_svg(waypoints, svg_path, dot_radius=0.05, padding_cm=1.0):
    """Write an SVG showing only waypoints as dots (no connecting lines).

    Coordinates are in cm; viewBox is the waypoints' bounding box (with
    padding) flipped so +y points up like in the robot frame.
    """
    if not waypoints:
        return
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    minx, maxx = min(xs) - padding_cm, max(xs) + padding_cm
    miny, maxy = min(ys) - padding_cm, max(ys) + padding_cm
    width = maxx - minx
    height = maxy - miny

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{width:.2f}cm" height="{height:.2f}cm" '
        f'viewBox="{minx:.3f} {-maxy:.3f} {width:.3f} {height:.3f}">',
        f'<rect x="{minx:.3f}" y="{-maxy:.3f}" width="{width:.3f}" '
        f'height="{height:.3f}" fill="white" stroke="#cccccc" '
        f'stroke-width="0.02"/>',
    ]
    for i, (x, y) in enumerate(waypoints):
        color = "red" if i == 0 else ("blue" if i == len(waypoints) - 1 else "black")
        parts.append(
            f'<circle cx="{x:.3f}" cy="{-y:.3f}" r="{dot_radius:.3f}" '
            f'fill="{color}"/>'
        )
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
    """Convert an image into <= max_points (x, y) cm waypoints.

    Pipeline (mirrors camera_linedraw): linedraw.vectorise -> pick the longest
    polyline -> uniformly resample by arc length -> rescale into a box_cm
    square centred on `center`. Image y is flipped so the sketch faces +y
    (forward).
    """
    import linedraw

    polylines = linedraw.vectorise(
        image_path,
        resolution=512,
        draw_contours=2,
        repeat_contours=1,
        draw_hatch=False,
        repeat_hatch=0,
        svg_folder="images/",
    )
    polylines = [p for p in polylines if len(p) >= 2]
    if not polylines:
        raise ValueError(f"No contours found in {image_path}.")

    # Concatenate every contour into one tour so the trace covers the whole
    # silhouette (outer edge, inner curve, stem, etc.), not just the longest
    # one. linedraw.sortlines already orders them nearest-neighbour so the
    # connector jumps between strokes are short.
    tour = []
    for poly in polylines:
        tour.extend(poly)
    print(
        f"  Combined {len(polylines)} contours into a {len(tour)}-point tour "
        f"(total arc length ~{_polyline_arclen(tour):.0f} px)."
    )
    waypoints_px = _resample_polyline(tour, max_points)

    xs = [p[0] for p in waypoints_px]
    ys = [p[1] for p in waypoints_px]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = max(maxx - minx, 1e-6)
    bh = max(maxy - miny, 1e-6)
    scale = box_cm / max(bw, bh)
    cx_img = (minx + maxx) / 2.0
    cy_img = (miny + maxy) / 2.0
    cx, cy = center

    out = []
    for px, py in waypoints_px:
        rx = cx + (px - cx_img) * scale
        ry = cy - (py - cy_img) * scale  # flip y: image y-down -> robot y-up
        out.append((rx, ry))

    if write_svg:
        base = os.path.splitext(image_path)[0]
        write_waypoints_svg(out, base + WAYPOINTS_SVG_SUFFIX)

    return out


def trace_waypoints(a0, a1, waypoints):
    """Drive the arm through a list of (x, y) cm waypoints in order."""
    if not waypoints:
        print("  No waypoints to trace.")
        return a0, a1
    print(f"Tracing {len(waypoints)} waypoints:")
    for i, (x, y) in enumerate(waypoints):
        print(f"  [{i + 1}/{len(waypoints)}] -> ({x:.2f}, {y:.2f}) cm")
        a0, a1 = goto_xy(a0, a1, x, y)
    return a0, a1


def trace_image(a0, a1, image_path=None):
    """Vectorise an image to <=10 waypoints and trace them with the arm."""
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


def draw_star(a0, a1):
    """
    5-point star (pentagram) in joint space: center at current pose, then trace
    V0->V2->V4->V1->V3->V0 and return to center.
    """
    R = STAR_RADIUS_DEG
    cx, cy = a0, a1
    V = []
    for k in range(5):
        deg = 90 - k * 72
        rad = math.radians(deg)
        V.append(
            (
                clamp_angle(cx + R * math.cos(rad)),
                clamp_angle(cy + R * math.sin(rad)),
            )
        )
    targets = [V[0], V[2], V[4], V[1], V[3], V[0], (cx, cy)]
    pos = (cx, cy)
    for target in targets:
        move_both_smooth(pos[0], pos[1], target[0], target[1])
        pos = target
    return cx, cy


# ---------------------------------------------------------
# 4. Main loop — keyboard control
# ---------------------------------------------------------
angle0 = HOME_0
angle1 = HOME_1
set_servo_angle(SERVO_CH_0, angle0)
set_servo_angle(SERVO_CH_1, angle1)

print(
    "Servos: channel 0 and 1. "
    "r = home, x = servo0 +30° then servo1 +20°, s = square, t = star, "
    "g = goto (x, y) cm, i = trace image waypoints, q = quit"
)
print(f"Home pen position: ({HOME_X:.2f}, {HOME_Y:.2f}) cm  L1={L1}, L2={L2}")

try:
    while True:
        ch = read_key()
        if ch in ("\x03", "q", "Q"):  # Ctrl+C or q
            break
        if ch in ("r", "R"):
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
                x, y = target
                angle0, angle1 = goto_xy(angle0, angle1, x, y)
                print(f"Goto done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")
        elif ch in ("i", "I"):
            angle0, angle1 = trace_image(angle0, angle1)
            print(f"Trace done: servo0={angle0:.2f}°, servo1={angle1:.2f}°")

except KeyboardInterrupt:
    pass
finally:
    print("\nShutting down and cleaning up...")
    board.shutdown()
