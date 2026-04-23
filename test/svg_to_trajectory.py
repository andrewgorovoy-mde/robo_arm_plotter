#!/usr/bin/env python3
"""
Convert BrachioGraph-style polylines (SVG or JSON from linedraw.py) into a
robot trajectory: servo angles for a two-link arm (shoulder + elbow).

Pipeline, mirroring BrachioGraph (https://github.com/evildmp/BrachioGraph):
    1. Load polylines (pixel space, y-down, as produced by linedraw.py).
    2. Auto-fit into a target drawing area in cm (y-up, plotter space),
       preserving the source aspect ratio.
    3. Run BrachioGraph's inverse kinematics on every waypoint to get
       (shoulder_motor_angle, elbow_motor_angle) in degrees.
    4. Emit a JSON trajectory.

Angle convention is BrachioGraph's motor-angle convention:
    - shoulder_motor_angle = asin(x / r) - acos((r^2 + L1^2 - L2^2)/(2*r*L1))
    - elbow_motor_angle    = pi - acos((L1^2 + L2^2 - r^2)/(2*L1*L2))
Map these into your servo frame (e.g. the 0-180 space used by test.py) with
an offset / sign per servo; see notes at the bottom of the emitted JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Polyline = List[Point]

DEFAULT_BOUNDS = (-8.0, 6.0, 7.0, 14.0)  # (xmin, xmax, ymin, ymax) in cm
SVG_NS = "{http://www.w3.org/2000/svg}"


# -------------- loading --------------


def load_polylines(path: str) -> Tuple[List[Polyline], Optional[Tuple[float, float]]]:
    """Return (polylines, (svg_width, svg_height)) or (polylines, None) for JSON."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        polylines = [[(float(p[0]), float(p[1])) for p in line] for line in raw]
        return polylines, None

    tree = ET.parse(path)
    root = tree.getroot()
    try:
        w = float(root.attrib.get("width", 0))
        h = float(root.attrib.get("height", 0))
    except ValueError:
        w, h = 0.0, 0.0

    polylines: List[Polyline] = []
    for pl in root.iter(SVG_NS + "polyline"):
        pts_attr = pl.attrib.get("points", "").replace(",", " ").split()
        try:
            vals = [float(v) for v in pts_attr]
        except ValueError:
            continue
        coords: Polyline = [
            (vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)
        ]
        if len(coords) >= 2:
            polylines.append(coords)

    return polylines, (w, h) if (w and h) else None


# -------------- geometry helpers --------------


def bounding_box(polylines: Sequence[Polyline]) -> Tuple[float, float, float, float]:
    xs = [p[0] for line in polylines for p in line]
    ys = [p[1] for line in polylines for p in line]
    return min(xs), max(xs), min(ys), max(ys)


def auto_fit(
    polylines: Sequence[Polyline],
    target_bounds: Tuple[float, float, float, float],
    margin: float = 0.0,
    flip_y: bool = True,
) -> List[Polyline]:
    """Fit polylines inside target_bounds preserving aspect; optionally flip Y."""
    tx_min, tx_max, ty_min, ty_max = target_bounds
    tw = (tx_max - tx_min) - 2 * margin
    th = (ty_max - ty_min) - 2 * margin
    if tw <= 0 or th <= 0:
        raise ValueError("Target bounds (minus margin) have zero or negative size.")

    x_min, x_max, y_min, y_max = bounding_box(polylines)
    sw = x_max - x_min
    sh = y_max - y_min
    if sw <= 0 or sh <= 0:
        return [list(line) for line in polylines]

    scale = min(tw / sw, th / sh)
    draw_w = sw * scale
    draw_h = sh * scale
    ox = tx_min + margin + (tw - draw_w) / 2.0
    oy = ty_min + margin + (th - draw_h) / 2.0

    fitted: List[Polyline] = []
    for line in polylines:
        mapped: Polyline = []
        for x, y in line:
            fx = ox + (x - x_min) * scale
            if flip_y:
                fy = oy + (y_max - y) * scale
            else:
                fy = oy + (y - y_min) * scale
            mapped.append((fx, fy))
        fitted.append(mapped)
    return fitted


# -------------- BrachioGraph inverse kinematics --------------


def xy_to_angles(
    x: float, y: float, inner: float = 8.0, outer: float = 8.0
) -> Tuple[float, float]:
    """BrachioGraph's IK: returns (shoulder_motor_deg, elbow_motor_deg)."""
    hypotenuse = math.hypot(x, y)
    max_reach = inner + outer
    min_reach = abs(inner - outer)
    if hypotenuse > max_reach or hypotenuse < min_reach:
        raise ValueError(
            f"Point ({x:.3f},{y:.3f}) is unreachable with arms "
            f"{inner}/{outer}: |r|={hypotenuse:.3f}, "
            f"allowed [{min_reach:.3f}, {max_reach:.3f}]."
        )

    ratio = x / hypotenuse
    ratio = max(-1.0, min(1.0, ratio))
    hypotenuse_angle = math.asin(ratio)

    inner_cos = (hypotenuse ** 2 + inner ** 2 - outer ** 2) / (2 * hypotenuse * inner)
    outer_cos = (inner ** 2 + outer ** 2 - hypotenuse ** 2) / (2 * inner * outer)
    inner_cos = max(-1.0, min(1.0, inner_cos))
    outer_cos = max(-1.0, min(1.0, outer_cos))

    inner_angle = math.acos(inner_cos)
    outer_angle = math.acos(outer_cos)

    shoulder_motor_angle = hypotenuse_angle - inner_angle
    elbow_motor_angle = math.pi - outer_angle



    print("Shoulder: ", shoulder_motor_angle, "\nElbow: ", elbow_motor_angle)
    return math.degrees(shoulder_motor_angle), math.degrees(elbow_motor_angle)


def polylines_to_angles(
    polylines: Sequence[Polyline], inner: float, outer: float
) -> Tuple[List[List[dict]], List[Tuple[float, float]]]:
    """Returns per-polyline waypoints (dicts) and a flat list of unreachable coords."""
    out: List[List[dict]] = []
    unreachable: List[Tuple[float, float]] = []
    for line in polylines:
        waypoints: List[dict] = []
        for x, y in line:
            try:
                s_deg, e_deg = xy_to_angles(x, y, inner, outer)
            except ValueError:
                unreachable.append((x, y))
                continue
            waypoints.append(
                {
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "shoulder": round(s_deg, 3),
                    "elbow": round(e_deg, 3),
                }
            )
        if len(waypoints) >= 2:
            out.append(waypoints)
    return out, unreachable


# -------------- output --------------


def flatten_waypoints(polyline_angles: Sequence[Sequence[dict]]) -> List[dict]:
    """Join every polyline into a single sequence (no pen-ups, per user choice)."""
    flat: List[dict] = []
    for line in polyline_angles:
        flat.extend(line)
    return flat


def build_trajectory(
    source: str,
    svg_size: Optional[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
    inner: float,
    outer: float,
    polyline_angles: Sequence[Sequence[dict]],
    unreachable: Sequence[Tuple[float, float]],
) -> dict:
    return {
        "source": source,
        "units": "cm",
        "angle_convention": "brachiograph_motor_angles_deg",
        "arm": {"inner_cm": inner, "outer_cm": outer},
        "drawing_bounds_cm": {
            "x_min": bounds[0],
            "x_max": bounds[1],
            "y_min": bounds[2],
            "y_max": bounds[3],
        },
        "svg_size_px": (
            {"width": svg_size[0], "height": svg_size[1]} if svg_size else None
        ),
        "counts": {
            "polylines": len(polyline_angles),
            "waypoints": sum(len(p) for p in polyline_angles),
            "unreachable_points": len(unreachable),
        },
        "polylines": [list(line) for line in polyline_angles],
        "flat_waypoints": flatten_waypoints(polyline_angles),
        "notes": [
            "Angles are BrachioGraph motor angles in degrees.",
            "shoulder_motor_angle=0 corresponds to the inner arm pointing along",
            " +y from the shoulder when the arm is fully extended (r = inner+outer).",
            "elbow_motor_angle=0 means the arm is fully extended.",
            "To drive servos like test.py (which uses 0-180° with HOME=90),",
            " apply a linear map: servo_deg = home + k * (motor_deg - motor_home),",
            " with k=+/-1 depending on servo orientation. Calibrate by measuring",
            " motor_deg at your physical home pose and comparing to servo deg.",
        ],
    }


# -------------- CLI --------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an SVG/JSON polyline file (from linedraw.py) into a "
            "BrachioGraph-style servo-angle trajectory JSON."
        )
    )
    parser.add_argument(
        "input",
        help="Path to polyline SVG or linedraw JSON (pixel space).",
    )
    parser.add_argument(
        "--inner",
        type=float,
        default=8.0,
        help="Inner arm length in cm (shoulder to elbow). Default 8.0.",
    )
    parser.add_argument(
        "--outer",
        type=float,
        default=8.0,
        help="Outer arm length in cm (elbow to pen). Default 8.0.",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=list(DEFAULT_BOUNDS),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help=(
            "Target drawing rectangle in cm. "
            f"Default {DEFAULT_BOUNDS} (BrachioGraph default)."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Inner margin (cm) inside bounds. Default 0.25.",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help=(
            "Do not flip SVG Y axis. By default Y is flipped so SVG (y-down) "
            "becomes plotter (y-up)."
        ),
    )
    parser.add_argument(
        "--out",
        help=(
            "Output trajectory JSON path. Defaults to "
            "<input-basename>.trajectory.json next to input."
        ),
    )
    args = parser.parse_args()

    polylines, svg_size = load_polylines(args.input)
    if not polylines:
        parser.error(f"No polylines found in {args.input}")

    bounds = tuple(args.bounds)  # type: ignore[assignment]
    fitted = auto_fit(
        polylines,
        bounds,
        margin=args.margin,
        flip_y=not args.no_flip_y,
    )

    polyline_angles, unreachable = polylines_to_angles(
        fitted, inner=args.inner, outer=args.outer
    )
    if not polyline_angles:
        parser.error(
            "All points were unreachable with the current arm/bounds config."
        )
    if unreachable:
        print(
            f"Warning: skipped {len(unreachable)} unreachable point(s). "
            "Consider expanding arm lengths or shrinking bounds."
        )

    traj = build_trajectory(
        source=os.path.abspath(args.input),
        svg_size=svg_size,
        bounds=bounds,
        inner=args.inner,
        outer=args.outer,
        polyline_angles=polyline_angles,
        unreachable=unreachable,
    )

    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.input)
        out_path = base + ".trajectory.json"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)

    print(
        f"Wrote {out_path}: {traj['counts']['polylines']} polylines, "
        f"{traj['counts']['waypoints']} waypoints."
    )
    shoulders = [w["shoulder"] for w in traj["flat_waypoints"]]
    elbows = [w["elbow"] for w in traj["flat_waypoints"]]
    if shoulders:
        print(
            f"Shoulder range: [{min(shoulders):.2f}, {max(shoulders):.2f}]°; "
            f"Elbow range: [{min(elbows):.2f}, {max(elbows):.2f}]°"
        )


if __name__ == "__main__":
    main()
