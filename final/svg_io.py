"""Tiny SVG reader/writer for plotter waypoints.

`write_strokes_svg` produces a multi-color preview of a stroke list (cm
coordinates with y-axis pointing up).

`polylines_from_svg` parses `<polyline>`, `<polygon>`, `<line>`, and simple
`<path>` (M/L/Z) elements -- enough for hand-authored waypoint SVGs without
needing a full SVG library.
"""

import os
import re
import xml.etree.ElementTree as ET


_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def write_strokes_svg(strokes, svg_path, padding_cm=1.0):
    """Render `strokes` (list of [(x, y), ...] in cm) into an SVG with the
    plotter convention (y-up, so SVG coordinates are flipped)."""
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
    for i, stroke in enumerate(strokes):
        color = _PALETTE[i % len(_PALETTE)]
        pts = " ".join(f"{x:.3f},{-y:.3f}" for x, y in stroke)
        parts.append(
            f'<polyline points="{pts}" stroke="{color}" '
            f'stroke-width="0.04" fill="none" opacity="0.8"/>'
        )
        x0, y0 = stroke[0]
        parts.append(
            f'<circle cx="{x0:.3f}" cy="{-y0:.3f}" r="0.05" fill="{color}"/>'
        )
    parts.append("</svg>")

    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  Wrote strokes visualization: {svg_path}")


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_PATH_TOKEN_RE = re.compile(r"[MLmlZz]|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _parse_points(s):
    nums = [float(t) for t in _NUM_RE.findall(s)]
    return list(zip(nums[0::2], nums[1::2]))


def _parse_path_d(d):
    """Parse an SVG `d` attribute restricted to M/L/Z (case-insensitive)
    commands and return a list of polylines."""
    tokens = _PATH_TOKEN_RE.findall(d)
    polylines = []
    cur = None
    start = None
    current_poly = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("M", "m"):
            if current_poly and len(current_poly) >= 2:
                polylines.append(current_poly)
            current_poly = []
            rel = tok == "m"
            i += 1
            x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
            if rel and cur is not None:
                x += cur[0]; y += cur[1]
            cur = (x, y); start = cur
            current_poly.append(cur)
            while i + 1 < len(tokens) and tokens[i] not in "MLZmlz":
                x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
                if rel:
                    x += cur[0]; y += cur[1]
                cur = (x, y)
                current_poly.append(cur)
        elif tok in ("L", "l"):
            rel = tok == "l"
            i += 1
            while i + 1 < len(tokens) and tokens[i] not in "MLZmlz":
                x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
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


def polylines_from_svg(svg_path):
    """Return a list of polylines (in raw SVG coordinates, y-down) from
    `<polyline>`, `<polygon>`, `<line>`, and simple `<path>` elements."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    polylines = []
    for el in root.iter():
        t = el.tag.split("}")[-1]
        if t == "polyline":
            pts = _parse_points(el.attrib.get("points", ""))
            if len(pts) >= 2:
                polylines.append(pts)
        elif t == "polygon":
            pts = _parse_points(el.attrib.get("points", ""))
            if len(pts) >= 2:
                if pts[0] != pts[-1]:
                    pts = pts + [pts[0]]
                polylines.append(pts)
        elif t == "line":
            x1 = float(el.attrib.get("x1", 0))
            y1 = float(el.attrib.get("y1", 0))
            x2 = float(el.attrib.get("x2", 0))
            y2 = float(el.attrib.get("y2", 0))
            polylines.append([(x1, y1), (x2, y2)])
        elif t == "path":
            polylines.extend(_parse_path_d(el.attrib.get("d", "")))

    return polylines
