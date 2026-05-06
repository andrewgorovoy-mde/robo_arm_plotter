"""
Plot the servo calibration tables (SERVO0_CAL, SERVO1_CAL) from plotter_v2.py.

Generates:
  - A scatter of the raw calibration points for each servo
  - Linear vs cubic fit comparison for each servo
  - A combined chart with both servos overlaid

Outputs PNG files next to this script (or shows them interactively with --show).

Usage:
    python final/plot_calibration.py
    python final/plot_calibration.py --show
    python final/plot_calibration.py --out-dir final/cal_plots
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTTER_PATH = os.path.join(THIS_DIR, "plotter_v2.py")


def _load_cal_tables(path=PLOTTER_PATH):
    """Parse SERVO0_CAL / SERVO1_CAL from plotter_v2.py without importing it.

    plotter_v2 pulls in hardware-only deps (telemetrix) at import time, so we
    extract the literal lists via the AST instead.
    """
    with open(path, "r") as f:
        tree = ast.parse(f.read(), filename=path)

    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in ("SERVO0_CAL", "SERVO1_CAL"):
                found[tgt.id] = ast.literal_eval(node.value)

    missing = {"SERVO0_CAL", "SERVO1_CAL"} - found.keys()
    if missing:
        raise RuntimeError(f"could not find {missing} in {path}")
    return found["SERVO0_CAL"], found["SERVO1_CAL"]


def _build_cal_poly(cal_points, deg=3):
    """Polynomial fit of (angle, pulse)."""
    angles = np.array([p[0] for p in cal_points], dtype=float)
    pulses = np.array([p[1] for p in cal_points], dtype=float)
    fit_deg = min(deg, len(cal_points) - 1)
    coeffs = np.polyfit(angles, pulses, fit_deg)
    return np.poly1d(coeffs)


SERVO0_CAL, SERVO1_CAL = _load_cal_tables()


def _curve(cal_points, deg=3, num=400):
    angles = np.array([p[0] for p in cal_points], dtype=float)
    pulses = np.array([p[1] for p in cal_points], dtype=float)
    poly = _build_cal_poly(cal_points, deg=deg)
    a_dense = np.linspace(angles.min(), angles.max(), num)
    p_dense = np.array([poly(a) for a in a_dense])
    return angles, pulses, a_dense, p_dense


def plot_single(name, cal_points, color, out_path):
    angles, pulses, a_dense, p_cubic_dense = _curve(cal_points, deg=3)
    _, _, _, p_linear_dense = _curve(cal_points, deg=1)

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_main.scatter(angles, pulses, color=color, zorder=3, label=f"{name} samples")
    ax_main.plot(
        a_dense, p_linear_dense, color="tab:gray", linestyle="--", label=f"{name} linear fit"
    )
    ax_main.plot(a_dense, p_cubic_dense, color=color, alpha=0.8, label=f"{name} cubic fit")
    ax_main.set_ylabel("Pulse (counts)")
    ax_main.set_title(f"{name} calibration: linear vs cubic fit")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    poly_linear = _build_cal_poly(cal_points, deg=1)
    poly_cubic = _build_cal_poly(cal_points, deg=3)
    residuals_linear = pulses - np.array([poly_linear(a) for a in angles])
    residuals_cubic = pulses - np.array([poly_cubic(a) for a in angles])
    ax_resid.axhline(0, color="black", linewidth=0.8)
    ax_resid.scatter(angles, residuals_linear, color="tab:gray", zorder=2, label="linear residual")
    ax_resid.scatter(angles, residuals_cubic, color=color, zorder=3, label="cubic residual")
    ax_resid.set_xlabel("Angle (deg)")
    ax_resid.set_ylabel("Residual\n(pulse)")
    ax_resid.grid(True, alpha=0.3)
    ax_resid.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return fig


def plot_combined(out_path):
    a0, p0, ad0, pd0 = _curve(SERVO0_CAL)
    a1, p1, ad1, pd1 = _curve(SERVO1_CAL)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(a0, p0, color="tab:blue", zorder=3, label="servo0 samples")
    ax.plot(ad0, pd0, color="tab:blue", alpha=0.7, label="servo0 cubic fit")
    ax.scatter(a1, p1, color="tab:red", zorder=3, label="servo1 samples")
    ax.plot(ad1, pd1, color="tab:red", alpha=0.7, label="servo1 cubic fit")

    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Pulse (counts)")
    ax.set_title("Servo calibration curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(THIS_DIR, "cal_plots"),
        help="directory to write PNG charts into",
    )
    parser.add_argument(
        "--show", action="store_true", help="display the charts interactively"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    plot_single(
        "servo0", SERVO0_CAL, "tab:blue",
        os.path.join(args.out_dir, "servo0_calibration.png"),
    )
    plot_single(
        "servo1", SERVO1_CAL, "tab:red",
        os.path.join(args.out_dir, "servo1_calibration.png"),
    )
    plot_combined(os.path.join(args.out_dir, "servos_calibration_combined.png"))

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
