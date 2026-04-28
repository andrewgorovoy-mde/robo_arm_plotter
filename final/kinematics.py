"""Inverse kinematics, joint-limit reachability, and drawing-box search
for the 2R arm. Pure functions; configure once via `configure(...)` and the
module reads the current values whenever it's called.

Coordinate conventions (matching plotter_v2.py):
  * Shoulder origin at (0, 0).
  * `theta1` is the upper-arm angle from +X. `phi` is the elbow joint
    angle (relative). With `K0 = K1 = -1` the conversions are:
        servo0 = 90 + K0 * theta1_deg
        servo1 = 90 + K1 * (phi_deg - 90)
"""

import math

import numpy as np


# ---------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------
_cfg = {
    "L1": 10.3,
    "L2": 10.9,
    "K0": -1,
    "K1": -1,
    "elbow_up": True,
    "joint_limits": [(0.0, 90.0), (40.0, 160.0)],
    "joint_margin_deg": 3.0,
}


def configure(**kwargs):
    """Set arm geometry / joint limits. Recognized keys: L1, L2, K0, K1,
    elbow_up, joint_limits (list of two (min, max) tuples), joint_margin_deg."""
    _cfg.update(kwargs)


# ---------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------
def clamp_angle(a):
    """Clamp to the absolute servo range [0, 180]."""
    return max(0, min(180, a))


# ---------------------------------------------------------
# Forward / inverse kinematics
# ---------------------------------------------------------
def xy_to_servo_angles(x, y, elbow_up=None):
    """Solve IK for a 2-link arm. Raises ValueError if the target is
    geometrically out of reach (joint *limits* are NOT enforced here -- use
    `is_xy_reachable` for that)."""
    L1, L2 = _cfg["L1"], _cfg["L2"]
    K0, K1 = _cfg["K0"], _cfg["K1"]
    if elbow_up is None:
        elbow_up = _cfg["elbow_up"]

    r2 = x * x + y * y
    r = math.sqrt(r2)
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


def is_xy_reachable(x, y, margin_deg=None):
    """A point is reachable when IK is solvable AND both servo angles fall
    inside the configured joint limits with `margin_deg` padding."""
    if margin_deg is None:
        margin_deg = _cfg["joint_margin_deg"]
    try:
        s0, s1 = xy_to_servo_angles(x, y)
    except ValueError:
        return False
    (s0min, s0max), (s1min, s1max) = _cfg["joint_limits"]
    return (s0min + margin_deg <= s0 <= s0max - margin_deg
            and s1min + margin_deg <= s1 <= s1max - margin_deg)


# ---------------------------------------------------------
# Drawing-box search
# ---------------------------------------------------------
def _largest_true_rect(grid):
    """Largest axis-aligned all-True rectangle in a 2-D bool array.
    Standard histogram method, O(rows*cols).
    Returns (i0, j0, i1, j1) inclusive, or None."""
    rows, cols = grid.shape
    heights = [0] * cols
    best = None
    for i in range(rows):
        for j in range(cols):
            heights[j] = heights[j] + 1 if grid[i, j] else 0
        stack = []
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


def compute_drawing_box(margin_deg=None, grid_step=0.2,
                        x_range=(-8.0, 22.0), y_range=(-2.0, 22.0)):
    """Find the largest axis-aligned rectangle (in cm) that lies entirely
    inside the reachable workspace. Returns `(cx, cy, half_w, half_h)`,
    or `None` if the grid is empty."""
    xs = np.arange(x_range[0], x_range[1] + grid_step, grid_step)
    ys = np.arange(y_range[0], y_range[1] + grid_step, grid_step)
    grid = np.zeros((len(ys), len(xs)), dtype=bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid[i, j] = is_xy_reachable(x, y, margin_deg)
    rect = _largest_true_rect(grid)
    if rect is None:
        return None
    i0, j0, i1, j1 = rect
    x0, x1 = xs[j0], xs[j1]
    y0, y1 = ys[i0], ys[i1]
    return (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
