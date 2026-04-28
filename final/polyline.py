"""Polyline math: resampling, U-turn splitting, backtrack removal.

Pure functions; no dependency on the plotter, hardware, or matplotlib.
Coordinates are tuples of `(x, y)` floats; units are whatever the caller is
working in (cm, pixels, etc.) — the only unit-aware parameter is
`overlap_eps` in `remove_backtracking`.
"""

import bisect
import math


def resample_polyline(poly, n):
    """Return `n` points spaced uniformly by arc length along `poly`.
    The first and last points of the output match the input."""
    if n <= 1 or len(poly) <= 1:
        return [poly[0]] * max(1, n)
    cum = [0.0]
    for i in range(1, len(poly)):
        cum.append(cum[-1] + math.hypot(
            poly[i][0] - poly[i - 1][0],
            poly[i][1] - poly[i - 1][1],
        ))
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


def _point_seg_dist(p, a, b):
    """Shortest distance from point `p` to the segment `a`-`b`."""
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
    0° = straight, 180° = exact reverse (U-turn)."""
    v1x, v1y = p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]
    v2x, v2y = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(c))


def remove_backtracking(poly, angle_threshold_deg=150.0, overlap_eps=0.2):
    """Split `poly` wherever consecutive edges reverse direction by more
    than `angle_threshold_deg` (a U-turn). For each split part, drop the
    leading vertices that lie within `overlap_eps` of any segment already
    emitted -- those points retrace what's already been drawn.

    Returns a list of cleaned polylines (each with len >= 2)."""
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
