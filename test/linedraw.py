# Derived from BrachioGraph linedraw (https://github.com/evildmp/BrachioGraph/blob/main/linedraw.py)
# which is based on https://github.com/LingDong-/linedraw — Lingdong Huang.

"""
Convert a raster image into polylines for pen plotters: each line is a list of
(x, y) points in pixel space (same convention as BrachioGraph).

Dependencies: Pillow; optional numpy + OpenCV for Canny edges (falls back to Sobel).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Sequence, Tuple

from PIL import Image, ImageOps

# Default output locations (BrachioGraph-style)
SVG_FOLDER = "images/"
JSON_FOLDER = "images/"

no_cv = False

try:
    import numpy as np
    import cv2
except ImportError:
    print("Cannot import numpy/openCV. Switching to NO_CV mode.")
    no_cv = True

Point = Tuple[float, float]
Polyline = List[Point]


# -------------- output --------------


def makesvg(lines: Sequence[Polyline]) -> str:
    """Build a minimal SVG of polylines (coordinates halved, as in upstream)."""
    print("Generating svg file...")
    if not lines:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            'width="1" height="1"></svg>'
        )
    width = math.ceil(max(max(p[0] * 0.5 for p in l) for l in lines))
    height = math.ceil(max(max(p[1] * 0.5 for p in l) for l in lines))
    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" width="%s" height="%s">'
            % (width, height)
        )
    ]
    for line in lines:
        pts = ",".join(str(p[0] * 0.5) + "," + str(p[1] * 0.5) for p in line)
        parts.append(
            '<polyline points="%s" stroke="black" stroke-width="1" fill="none" />'
            % pts
        )
    parts.append("</svg>")
    return "\n".join(parts)


def lines_to_file(lines: Sequence[Polyline], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=4)


def image_to_json(
    image_filename: str,
    resolution: int = 1024,
    draw_contours: bool = False,
    repeat_contours: int = 1,
    draw_hatch: bool = False,
    repeat_hatch: int = 1,
    svg_folder: str = SVG_FOLDER,
    json_folder: str = JSON_FOLDER,
) -> None:
    lines = vectorise(
        image_filename,
        resolution,
        draw_contours,
        repeat_contours,
        draw_hatch,
        repeat_hatch,
        svg_folder=svg_folder,
    )
    base = os.path.splitext(os.path.basename(image_filename))[0]
    out_json = os.path.join(json_folder, base + ".json")
    os.makedirs(json_folder, exist_ok=True)
    lines_to_file(lines, out_json)


# -------------- turtle preview (optional) --------------


def draw(lines: Sequence[Polyline]) -> None:
    from tkinter import Tk
    from turtle import Canvas, RawTurtle, TurtleScreen

    root = Tk()
    canvas = Canvas(root, width=800, height=800)
    canvas.pack()
    s = TurtleScreen(canvas)
    t = RawTurtle(canvas)
    t.speed(0)
    t.width(1)
    for line in lines:
        x, y = line[0]
        t.up()
        t.goto(x * 800 / 1024 - 400, -(y * 800 / 1024 - 400))
        for point in line:
            t.down()
            t.goto(point[0] * 800 / 1024 - 400, -(point[1] * 800 / 1024 - 400))
    s.mainloop()


# -------------- vectorisation --------------


def vectorise(
    image_filename: str,
    resolution: int = 1024,
    draw_contours: bool = False,
    repeat_contours: int = 1,
    draw_hatch: bool = False,
    repeat_hatch: int = 1,
    svg_folder: str = SVG_FOLDER,
) -> List[Polyline]:
    image = None
    possible = [
        image_filename,
        os.path.join("images", image_filename),
        os.path.join("images", image_filename + ".jpg"),
        os.path.join("images", image_filename + ".png"),
        os.path.join("images", image_filename + ".tif"),
    ]
    for p in possible:
        try:
            image = Image.open(p)
            break
        except OSError:
            pass
    if image is None:
        raise FileNotFoundError(
            "Could not open image; tried: " + ", ".join(possible)
        )

    image = image.convert("L")
    image = ImageOps.autocontrast(image, 5, preserve_tone=True)

    lines: List[Polyline] = []

    if draw_contours and repeat_contours:
        contours = getcontours(
            resize_image(image, resolution, draw_contours), draw_contours
        )
        contours = sortlines(contours)
        contours = join_lines(contours)
        for _ in range(repeat_contours):
            lines += contours

    if draw_hatch and repeat_hatch:
        hatches = hatch(resize_image(image, resolution), line_spacing=draw_hatch)
        hatches = sortlines(hatches)
        hatches = join_lines(hatches)
        for _ in range(repeat_hatch):
            lines += hatches

    segments = sum(len(line) - 1 for line in lines)
    print(len(lines), "lines,", segments, "segments.")

    os.makedirs(svg_folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_filename))[0]
    svg_path = os.path.join(svg_folder, base + ".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(makesvg(lines))

    return lines


def resize_image(image: Image.Image, resolution: int, divider: int = 1) -> Image.Image:
    w, h = image.size
    return image.resize(
        (
            int(resolution / divider),
            int(resolution / divider * h / w),
        )
    )


# -------------- contours & hatch --------------


def getcontours(image: Image.Image, draw_contours: int = 2) -> List[Polyline]:
    print("Generating contours...")
    image = find_edges(image)
    im1 = image.copy()
    im2 = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    dots1 = getdots(im1)
    contours1 = connectdots(dots1)
    dots2 = getdots(im2)
    contours2 = connectdots(dots2)

    for i in range(len(contours2)):
        contours2[i] = [(c[1], c[0]) for c in contours2[i]]
    contours = contours1 + contours2

    for i in range(len(contours)):
        for j in range(len(contours)):
            if len(contours[i]) > 0 and len(contours[j]) > 0:
                if distsum(contours[j][0], contours[i][-1]) < 8:
                    contours[i] = contours[i] + contours[j]
                    contours[j] = []

    for i in range(len(contours)):
        contours[i] = [contours[i][j] for j in range(0, len(contours[i]), 8)]

    contours = [c for c in contours if len(c) > 1]

    for i in range(0, len(contours)):
        contours[i] = [
            (v[0] * draw_contours, v[1] * draw_contours) for v in contours[i]
        ]

    return contours


E = (1, 0)
S = (0, 1)
SE = (1, 1)
NE = (1, -1)


def hatch(image: Image.Image, line_spacing: int = 16) -> List[Polyline]:
    lines: List[Polyline] = []
    lines.extend(get_lines(image, "y", E, line_spacing, 160))
    lines.extend(get_lines(image, "x", S, line_spacing, 80))
    lines.extend(get_lines(image, "y", SE, line_spacing, 40))
    lines.extend(get_lines(image, "x", SE, line_spacing, 40))
    lines.extend(get_lines(image, "y", NE, line_spacing, 20))
    lines.extend(get_lines(image, "x", NE, line_spacing, 20))
    return lines


def get_lines(
    image: Image.Image,
    scan: str,
    direction: Tuple[int, int],
    line_spacing: int,
    level: int,
) -> List[Polyline]:
    pixels = image.load()
    width, height = image.size[0], image.size[1]
    i_start = 0
    j_start = 0
    lines: List[Polyline] = []

    if scan == "y":
        i_range = height
    elif scan == "x":
        i_range = width
    else:
        raise ValueError("scan must be 'x' or 'y'")

    if direction == SE:
        i_start = line_spacing
    elif direction == NE:
        i_start = line_spacing - ((height - 1) % line_spacing)
        j_start = height - 1

    for i in range(i_start, i_range, line_spacing):
        start_point = None

        if scan == "y":
            x, y = j_start, i
        else:
            x, y = i, j_start

        while (0 <= x < width) and (0 <= y < height):
            if not start_point:
                if pixels[x, y] < level:
                    start_point = (x, y)
            else:
                if pixels[x, y] >= level:
                    end_point = (x, y)
                    lines.append([start_point, end_point])
                    start_point = None

            end_point = (x, y)
            x += direction[0]
            y += direction[1]

        if start_point:
            lines.append([start_point, end_point])

    return lines


def join_segments(line_groups):
    print("Making segments into lines...")
    for line_group in line_groups:
        for lines in line_group:
            for lines2 in line_group:
                if lines and lines2:
                    if lines[-1] == lines2[0]:
                        lines.extend(lines2[1:])
                        lines2.clear()
        saved_lines = [[line[0], line[-1]] for line in line_group if line]
        line_group.clear()
        line_group.extend(saved_lines)
    return [item for group in line_groups for item in group]


# -------------- edge / contour helpers --------------


def find_edges(image: Image.Image) -> Image.Image:
    print("Finding edges...")
    if no_cv:
        appmask(image, [F_SobelX, F_SobelY])
    else:
        im = np.array(image)
        im = cv2.GaussianBlur(im, (3, 3), 0)
        im = cv2.Canny(im, 100, 200)
        image = Image.fromarray(im)
    return image.point(lambda p: p > 128 and 255)


def getdots(im: Image.Image):
    print("Getting contour points...")
    px = im.load()
    dots = []
    w, h = im.size
    for y in range(h - 1):
        row = []
        for x in range(1, w):
            if px[x, y] == 255:
                if len(row) > 0:
                    if x - row[-1][0] == row[-1][-1] + 1:
                        row[-1] = (row[-1][0], row[-1][-1] + 1)
                    else:
                        row.append((x, 0))
                else:
                    row.append((x, 0))
        dots.append(row)
    return dots


def connectdots(dots):
    print("Connecting contour points...")
    contours = []
    for y in range(len(dots)):
        for x, v in dots[y]:
            if v > -1:
                if y == 0:
                    contours.append([(x, y)])
                else:
                    closest = -1
                    cdist = 100
                    for x0, _ in dots[y - 1]:
                        if abs(x0 - x) < cdist:
                            cdist = abs(x0 - x)
                            closest = x0

                    if cdist > 3:
                        contours.append([(x, y)])
                    else:
                        found = 0
                        for i in range(len(contours)):
                            if contours[i][-1] == (closest, y - 1):
                                contours[i].append((x, y))
                                found = 1
                                break
                        if found == 0:
                            contours.append([(x, y)])
    last_y = len(dots) - 1
    contours = [
        c for c in contours if not (c[-1][1] < last_y - 1 and len(c) < 4)
    ]
    return contours


# -------------- line ordering --------------


def sortlines(lines: List[Polyline]) -> List[Polyline]:
    print("Optimising line sequence...")
    if not lines:
        return []
    clines = lines[:]
    slines = [clines.pop(0)]
    while clines:
        best, best_dist, reverse = None, 1000000.0, False
        for ln in clines:
            d = distsum(ln[0], slines[-1][-1])
            dr = distsum(ln[-1], slines[-1][-1])
            if d < best_dist:
                best, best_dist, reverse = ln[:], d, False
            if dr < best_dist:
                best, best_dist, reverse = ln[:], dr, True
        clines.remove(best)
        if reverse:
            best = best[::-1]
        slines.append(best)
    return slines


def join_lines(lines: List[Polyline], closeness: int = 128) -> List[Polyline]:
    previous_line = None
    new_lines: List[Polyline] = []

    for line in lines:
        if not previous_line:
            new_lines.append(line)
            previous_line = line
        else:
            xdiff = abs(previous_line[-1][0] - line[0][0])
            ydiff = abs(previous_line[-1][1] - line[0][1])
            if xdiff**2 + ydiff**2 <= closeness:
                previous_line.extend(line)
            else:
                new_lines.append(line)
                previous_line = line

    print(f"Reduced {len(lines)} lines to {len(new_lines)} lines.")
    return new_lines


# -------------- math helpers --------------


def midpt(*args: Point) -> Point:
    xs, ys = 0.0, 0.0
    for p in args:
        xs += p[0]
        ys += p[1]
    n = len(args)
    return xs / n, ys / n


def distsum(*args: Point) -> float:
    return sum(
        (
            (args[i][0] - args[i - 1][0]) ** 2 + (args[i][1] - args[i - 1][1]) ** 2
        )
        ** 0.5
        for i in range(1, len(args))
    )


# -------------- Sobel (no OpenCV) --------------


def appmask(im: Image.Image, masks):
    px = im.load()
    w, h = im.size
    npx = {}
    for x in range(0, w):
        for y in range(0, h):
            a = [0] * len(masks)
            for i in range(len(masks)):
                for p in masks[i].keys():
                    if 0 < x + p[0] < w and 0 < y + p[1] < h:
                        a[i] += px[x + p[0], y + p[1]] * masks[i][p]
                if sum(masks[i].values()) != 0:
                    a[i] = a[i] / sum(masks[i].values())
            npx[x, y] = int(sum(v**2 for v in a) ** 0.5)
    for x in range(0, w):
        for y in range(0, h):
            px[x, y] = npx[x, y]


F_Blur = {
    (-2, -2): 2,
    (-1, -2): 4,
    (0, -2): 5,
    (1, -2): 4,
    (2, -2): 2,
    (-2, -1): 4,
    (-1, -1): 9,
    (0, -1): 12,
    (1, -1): 9,
    (2, -1): 4,
    (-2, 0): 5,
    (-1, 0): 12,
    (0, 0): 15,
    (1, 0): 12,
    (2, 0): 5,
    (-2, 1): 4,
    (-1, 1): 9,
    (0, 1): 12,
    (1, 1): 9,
    (2, 1): 4,
    (-2, 2): 2,
    (-1, 2): 4,
    (0, 2): 5,
    (1, 2): 4,
    (2, 2): 2,
}
F_SobelX = {
    (-1, -1): 1,
    (0, -1): 0,
    (1, -1): -1,
    (-1, 0): 2,
    (0, 0): 0,
    (1, 0): -2,
    (-1, 1): 1,
    (0, 1): 0,
    (1, 1): -1,
}
F_SobelY = {
    (-1, -1): 1,
    (0, -1): 2,
    (1, -1): 1,
    (-1, 0): 0,
    (0, 0): 0,
    (1, 0): 0,
    (-1, 1): -1,
    (0, 1): -2,
    (1, 1): -1,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an image to polylines (JSON + SVG), BrachioGraph-style."
    )
    parser.add_argument(
        "image",
        help="Image path or basename (also searches ./images/).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Working resolution for the long edge (default 1024).",
    )
    parser.add_argument(
        "--contours",
        type=int,
        default=0,
        metavar="N",
        help="Contour simplify factor; 0 disables contours (default 0).",
    )
    parser.add_argument(
        "--repeat-contours",
        type=int,
        default=1,
        help="How many times to repeat contour strokes.",
    )
    parser.add_argument(
        "--hatch",
        type=int,
        default=0,
        metavar="SPACING",
        help="Hatch line spacing; 0 disables hatching (default 0).",
    )
    parser.add_argument(
        "--repeat-hatch",
        type=int,
        default=1,
        help="How many times to repeat hatch strokes.",
    )
    parser.add_argument(
        "--svg-dir",
        default=SVG_FOLDER,
        help="Directory for output SVG (default: images/).",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Also write polylines to this JSON path.",
    )
    args = parser.parse_args()

    draw_c = args.contours > 0
    draw_h = args.hatch > 0
    if not draw_c and not draw_h:
        parser.error("Enable at least one of --contours N or --hatch SPACING.")

    lines = vectorise(
        args.image,
        resolution=args.resolution,
        draw_contours=draw_c,
        repeat_contours=args.repeat_contours,
        draw_hatch=draw_h,
        repeat_hatch=args.repeat_hatch,
        svg_folder=args.svg_dir,
    )
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        lines_to_file(lines, args.json)
    print("Done.")


if __name__ == "__main__":
    main()
