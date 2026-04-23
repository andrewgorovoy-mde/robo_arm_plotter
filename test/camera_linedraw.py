#!/usr/bin/env python3
"""
Live webcam preview: press Space to capture the current frame and run linedraw
vectorisation (SVG + optional JSON under images/).

Requires OpenCV (cv2) for the camera. Install: pip install opencv-python

Processing runs in a background thread so the preview window stays responsive.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import linedraw


def main() -> None:
    try:
        import cv2
    except ImportError:
        print(
            "OpenCV is required for the camera. Install with: pip install opencv-python",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Webcam capture → linedraw SVG/JSON")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default 0).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=640,
        metavar="PX",
        help="Resize captured frame so the long edge is at most this many pixels "
        "before vectorising (default 640; smaller = faster). Use 0 to disable.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="linedraw internal working resolution along the long edge (default 512; "
        "1024 is slower).",
    )
    parser.add_argument(
        "--contours",
        type=int,
        default=2,
        metavar="N",
        help="Contour simplify factor; 0 disables (default 2).",
    )
    parser.add_argument(
        "--repeat-contours",
        type=int,
        default=1,
        help="Repeat contour passes.",
    )
    parser.add_argument(
        "--hatch",
        type=int,
        default=0,
        metavar="SPACING",
        help="Hatch spacing; 0 disables (default 0 — much faster than hatching).",
    )
    parser.add_argument(
        "--repeat-hatch",
        type=int,
        default=1,
        help="Repeat hatch passes.",
    )
    parser.add_argument(
        "--svg-dir",
        default=linedraw.SVG_FOLDER,
        help=f"SVG output directory (default {linedraw.SVG_FOLDER}).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write a JSON file alongside the SVG.",
    )
    args = parser.parse_args()

    draw_c = args.contours > 0
    draw_h = args.hatch > 0
    if not draw_c and not draw_h:
        parser.error("Enable at least one of --contours N or --hatch SPACING.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.svg_dir, exist_ok=True)
    os.makedirs(linedraw.JSON_FOLDER, exist_ok=True)

    window = "Camera — Space: capture & vectorise | Q: quit"
    print(window)
    print("Outputs go under:", os.path.abspath(args.svg_dir))
    if args.hatch > 0:
        print(
            "Note: hatching is slow on large images; use --hatch 0 for outline-only "
            "or increase spacing (e.g. --hatch 32).",
        )

    busy = threading.Event()
    status_lock = threading.Lock()
    status_text = ""

    def run_vectorise(png_path: str, base: str) -> None:
        nonlocal status_text
        t0 = time.perf_counter()
        try:
            with status_lock:
                status_text = "Processing…"
            lines = linedraw.vectorise(
                png_path,
                resolution=args.resolution,
                draw_contours=draw_c,
                repeat_contours=args.repeat_contours,
                draw_hatch=draw_h,
                repeat_hatch=args.repeat_hatch,
                svg_folder=args.svg_dir,
            )
            out_dir = linedraw.JSON_FOLDER
            if not args.no_json:
                json_path = os.path.join(out_dir, base + ".json")
                linedraw.lines_to_file(lines, json_path)
                print(f"Wrote {json_path}")
            svg_path = os.path.join(args.svg_dir, base + ".svg")
            elapsed = time.perf_counter() - t0
            print(
                f"Done in {elapsed:.1f}s. SVG: {svg_path} ({len(lines)} polylines)"
            )
            with status_lock:
                status_text = f"Done {elapsed:.1f}s"
        except Exception as e:
            print(f"linedraw failed: {e}", file=sys.stderr)
            with status_lock:
                status_text = "Error (see terminal)"
        finally:
            busy.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.", file=sys.stderr)
                break

            display = frame
            with status_lock:
                label = status_text
            if label:
                cv2.putText(
                    display,
                    label,
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if not label.startswith("Error") else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if busy.is_set():
                cv2.putText(
                    display,
                    "(busy — wait or press Q to quit)",
                    (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key != ord(" "):
                continue

            if busy.is_set():
                print("Still processing previous capture — ignored Space.")
                continue

            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = f"capture_{stamp}"
            png_name = base + ".png"
            out_dir = linedraw.JSON_FOLDER
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, png_name)

            to_save = frame
            if args.max_dimension and args.max_dimension > 0:
                h, w = to_save.shape[:2]
                m = max(h, w)
                if m > args.max_dimension:
                    scale = args.max_dimension / m
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    to_save = cv2.resize(
                        to_save, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

            cv2.imwrite(png_path, to_save)
            print(f"Saved {png_path}, starting linedraw in background…")

            busy.set()
            threading.Thread(
                target=run_vectorise,
                args=(png_path, base),
                daemon=True,
            ).start()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
