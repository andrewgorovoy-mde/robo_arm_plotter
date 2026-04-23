#!/usr/bin/env python3
"""Move a two-link arm to a single (x, y) cm waypoint using BrachioGraph IK.

Computes inverse kinematics, maps motor angles into the servo (0-180°) frame,
and ramps both joints smoothly using calibrated angle->pulse-width curves
(loaded from `calibration.json`, falling back to a linear mapping).

Examples:
    # Compute angles only, no hardware
    python3 move_to_xy.py -3 10 --dry-run

    # Custom calibration
    python3 move_to_xy.py -3 10 \\
        --inner 9 --outer 9 \\
        --servo0-home 90 --servo1-home 90 \\
        --motor0-home -45 --motor1-home 90 \\
        --k0 1 --k1 -1
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Callable, Optional

from svg_to_trajectory import xy_to_angles

try:
    import numpy as _np
except ImportError:
    _np = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PW_MIN = 100
PW_MAX = 650
PW_LINEAR_LO = 150.0
PW_LINEAR_HI = 600.0

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

START_HOME0 = 90.0
START_HOME1 = 90.0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def clamp_angle(a: float, lo: float = SERVO_MIN_DEG, hi: float = SERVO_MAX_DEG) -> float:
    return max(lo, min(hi, a))


def motor_to_servo(motor_deg: float, servo_home: float, motor_home: float, k: int) -> float:
    """Map a BrachioGraph motor angle into servo (0-180°) space."""
    return servo_home + k * (motor_deg - motor_home)


def servo_to_motor(servo_deg: float, servo_home: float, motor_home: float, k: int) -> float:
    """Inverse of `motor_to_servo`."""
    return motor_home + (servo_deg - servo_home) / k


def linear_angle_to_pw(angle: float) -> float:
    """Fallback 0-180° -> PCA9685 pulse counts (~150..600)."""
    return PW_LINEAR_LO + (angle / SERVO_MAX_DEG) * (PW_LINEAR_HI - PW_LINEAR_LO)


@dataclass
class JointFrame:
    """Per-joint mapping between BrachioGraph motor angles and servo angles."""
    servo_home: float
    motor_home: float
    k: int


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def _fit(angles: list[float], pws: list[float]) -> Callable[[float], float]:
    """Fit angle -> pw. Cubic via numpy.polyfit if available, else piecewise linear."""
    if len(angles) < 2:
        constant = float(pws[0]) if pws else 0.0
        return lambda _a: constant

    if _np is not None:
        degree = min(3, len(angles) - 1)
        poly = _np.poly1d(_np.polyfit(angles, pws, degree))
        return lambda a: float(poly(a))

    pairs = sorted(zip(angles, pws))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def piecewise(angle: float) -> float:
        if angle <= xs[0]:
            return ys[0]
        if angle >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= angle <= xs[i + 1]:
                t = (angle - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    return piecewise


def _parse_samples(block: dict) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split a calibration block into (cw, acw) angle/pw lists."""
    cw: list[tuple[float, float]] = []
    acw: list[tuple[float, float]] = []

    if "samples" in block:
        for s in block["samples"]:
            try:
                angle = float(s["angle"])
                pw = float(s["pw"])
                direction = s.get("direction", "cw")
            except (KeyError, TypeError, ValueError):
                continue
            (cw if direction == "cw" else acw).append((angle, pw))
    elif "angle_pws_bidi" in block:
        for angle_str, pws in block["angle_pws_bidi"].items():
            try:
                angle = float(angle_str)
            except ValueError:
                continue
            if pws.get("cw") is not None:
                cw.append((angle, float(pws["cw"])))
            if pws.get("acw") is not None:
                acw.append((angle, float(pws["acw"])))

    return cw, acw


def load_calibration(path: Optional[str]) -> dict:
    """Build per-servo angle->pw curves from a calibration JSON.

    Returns ``{logical_idx: cal_dict}`` where logical_idx is 0=shoulder, 1=elbow.

    cal_dict keys:
        convention:    "brachiograph_motor_deg" or "servo_deg"
        to_pw_cw:      callable(angle) -> pw, or None
        to_pw_acw:     callable(angle) -> pw, or None
        to_pw_mean:    callable(angle) -> pw  (always present)
        n_cw, n_acw:   sample counts
        angle_range:   (lo, hi) of fitted angles
        borrowed_from: int (only if curves were copied from the other servo)

    Supports two on-disk formats from `capture_calibration.py`:
        - new "samples":      list of {direction, angle, pw, xy_cm}
        - legacy "angle_pws_bidi": {angle: {cw, acw}}
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"No calibration file at {path}; using linear fallback.")
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Calibration load failed ({e}); using linear fallback.")
        return {}

    result: dict = {}
    for key, idx in (("servo_0", 0), ("servo_1", 1)):
        block = data.get(key, {})
        cw, acw = _parse_samples(block)
        all_samples = sorted(cw + acw)
        if len(all_samples) < 2:
            continue

        xs = [a for a, _ in all_samples]
        ys = [p for _, p in all_samples]
        result[idx] = {
            "convention": block.get("angle_convention", "servo_deg"),
            "to_pw_cw": _fit([a for a, _ in cw], [p for _, p in cw]) if len(cw) >= 2 else None,
            "to_pw_acw": _fit([a for a, _ in acw], [p for _, p in acw]) if len(acw) >= 2 else None,
            "to_pw_mean": _fit(xs, ys),
            "n_cw": len(cw),
            "n_acw": len(acw),
            "angle_range": (xs[0], xs[-1]),
        }

    for idx, cal in result.items():
        lo, hi = cal["angle_range"]
        print(
            f"Calibration servo_{idx} ({cal['convention']}): "
            f"cw={cal['n_cw']} acw={cal['n_acw']} samples, "
            f"angle range [{lo:.2f}, {hi:.2f}]°"
        )

    # Cross-fallback: if one servo has no calibration, borrow the other's curves.
    for missing, source in ((0, 1), (1, 0)):
        if missing not in result and source in result:
            print(
                f"servo_{missing} has no calibration; "
                f"reusing servo_{source}'s curves as fallback."
            )
            result[missing] = {**result[source], "borrowed_from": source}

    return result


# ---------------------------------------------------------------------------
# Servo controller
# ---------------------------------------------------------------------------

class ServoController:
    """Drives a PCA9685 with calibrated, hysteresis-aware angle->pulse mapping."""

    def __init__(
        self,
        pca,
        channels: tuple[int, int],
        frames: dict[int, JointFrame],
        calibration: dict,
        step_deg: float,
        pulse_delay: float,
    ) -> None:
        self.pca = pca
        self.channels = channels  # (shoulder_ch, elbow_ch)
        self.frames = frames
        self.calibration = calibration
        self.step_deg = step_deg
        self.pulse_delay = pulse_delay
        self._channel_to_logical = {channels[0]: 0, channels[1]: 1}
        self._previous_pw: dict[int, int] = {}

    def _angle_to_pw(self, channel: int, angle: float) -> float:
        logical = self._channel_to_logical.get(channel)
        cal = self.calibration.get(logical) if logical is not None else None
        if cal is None:
            return linear_angle_to_pw(angle)

        if cal["convention"] == "brachiograph_motor_deg":
            f = self.frames[logical]
            cal_input = servo_to_motor(angle, f.servo_home, f.motor_home, f.k)
        else:
            cal_input = angle

        # A borrowed calibration only models the source servo's sampled range.
        # Extrapolating a cubic far outside that range produces nonsense pulse
        # widths (e.g. servo_1 home with shoulder curves -> hits PW_MIN). Use
        # the linear servo->pw mapping when we'd be extrapolating.
        if cal.get("borrowed_from") is not None:
            lo, hi = cal["angle_range"]
            if cal_input < lo - 1.0 or cal_input > hi + 1.0:
                return linear_angle_to_pw(angle)

        mean_pw = cal["to_pw_mean"](cal_input)
        prev = self._previous_pw.get(channel)
        if prev is None:
            return mean_pw
        if mean_pw > prev and cal["to_pw_cw"] is not None:
            return cal["to_pw_cw"](cal_input)
        if mean_pw < prev and cal["to_pw_acw"] is not None:
            return cal["to_pw_acw"](cal_input)
        return mean_pw

    def set_angle(self, channel: int, angle: float, use_calibration: bool = True) -> None:
        pw = self._angle_to_pw(channel, angle) if use_calibration else linear_angle_to_pw(angle)
        pw_int = max(PW_MIN, min(PW_MAX, int(round(pw))))
        self._previous_pw[channel] = pw_int
        self.pca.set_pwm(channel, 0, pw_int)

    def move_smooth(
        self,
        a0_start: float,
        a1_start: float,
        a0_end: float,
        a1_end: float,
        use_calibration: bool = True,
    ) -> None:
        """Linearly ramp both joints from (a0_start, a1_start) to (a0_end, a1_end).

        Set ``use_calibration=False`` to bypass the calibrated polynomial and
        use the linear servo-angle -> pulse-width mapping end-to-end. Useful
        for home moves where the servos sit mid-range and the untrusted
        calibration doesn't help.
        """
        ch0, ch1 = self.channels
        a0_start, a1_start = clamp_angle(a0_start), clamp_angle(a1_start)
        a0_end, a1_end = clamp_angle(a0_end), clamp_angle(a1_end)

        max_delta = max(abs(a0_end - a0_start), abs(a1_end - a1_start))
        if max_delta < 0.01:
            return

        steps = max(1, int(round(max_delta / self.step_deg)))
        for i in range(1, steps + 1):
            t = i / steps
            self.set_angle(ch0, a0_start + (a0_end - a0_start) * t, use_calibration)
            self.set_angle(ch1, a1_start + (a1_end - a1_start) * t, use_calibration)
            time.sleep(self.pulse_delay)
        self.set_angle(ch0, a0_end, use_calibration)
        self.set_angle(ch1, a1_end, use_calibration)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def wait_for_key(key: str, prompt: str = "") -> None:
    """Block until `key` (single char) is pressed. Falls back to line input over pipes."""
    if prompt:
        print(prompt)
    target = key.lower()

    if not sys.stdin.isatty():
        while True:
            line = input().strip().lower()
            if line == target:
                return
            print(f"Type '{target}' then Enter.")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready and sys.stdin.read(1).lower() == target:
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Interactive home calibration
# ---------------------------------------------------------------------------

CAL_HELP = """
Interactive calibration mode
Tune motor home angles so the tip lands on the plotted point.
Commands (press key then Enter):
  a/d : motor0-home -/+ step
  j/l : motor1-home -/+ step
  z/x : step /2 or *2
  m   : preview move to target
  h   : move to startup home (90,90)
  p   : finish + print recommended args
  q   : quit calibration without changes
""".strip()


def interactive_calibration(
    controller: ServoController,
    args: argparse.Namespace,
    shoulder_deg: float,
    elbow_deg: float,
    home0: float,
    home1: float,
) -> tuple[float, float]:
    motor0_home = args.motor0_home
    motor1_home = args.motor1_home
    step = max(0.1, float(args.cal_step))
    at_target = False

    def target_for(m0: float, m1: float) -> tuple[float, float]:
        s0 = motor_to_servo(shoulder_deg, args.servo0_home, m0, args.k0)
        s1 = motor_to_servo(elbow_deg, args.servo1_home, m1, args.k1)
        return clamp_angle(s0), clamp_angle(s1)

    print("\n" + CAL_HELP)

    while True:
        target0, target1 = target_for(motor0_home, motor1_home)
        print(
            f"\nmotor0-home={motor0_home:.2f}, motor1-home={motor1_home:.2f}, "
            f"step={step:.2f}, target=({target0:.2f}, {target1:.2f})"
        )
        cmd = input("cal> ").strip().lower()
        if not cmd:
            continue

        if cmd in ("a", "d", "j", "l"):
            if cmd == "a":
                motor0_home -= step
            elif cmd == "d":
                motor0_home += step
            elif cmd == "j":
                motor1_home -= step
            elif cmd == "l":
                motor1_home += step
            if at_target:
                # Re-preview after each tweak.
                controller.move_smooth(target0, target1, home0, home1, use_calibration=False)
                target0, target1 = target_for(motor0_home, motor1_home)
                controller.move_smooth(home0, home1, target0, target1)
        elif cmd == "z":
            step = max(0.1, step / 2.0)
        elif cmd == "x":
            step = min(45.0, step * 2.0)
        elif cmd == "h":
            if at_target:
                controller.move_smooth(target0, target1, home0, home1, use_calibration=False)
            else:
                controller.set_angle(args.servo0_ch, home0, use_calibration=False)
                controller.set_angle(args.servo1_ch, home1, use_calibration=False)
            at_target = False
        elif cmd == "m":
            if at_target:
                controller.move_smooth(target0, target1, home0, home1, use_calibration=False)
            controller.move_smooth(home0, home1, target0, target1)
            at_target = True
        elif cmd == "p":
            if at_target:
                controller.move_smooth(target0, target1, home0, home1, use_calibration=False)
            print("\nCalibration complete.")
            print(
                "Use these args next run: "
                f"--motor0-home {motor0_home:.3f} --motor1-home {motor1_home:.3f}"
            )
            return motor0_home, motor1_home
        elif cmd == "q":
            if at_target:
                controller.move_smooth(target0, target1, home0, home1, use_calibration=False)
            print("\nCalibration cancelled; keeping existing values.")
            return args.motor0_home, args.motor1_home
        else:
            print("Unknown command. Use a/d/j/l/z/x/m/h/p/q.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move the arm to a single (x, y) cm waypoint using BrachioGraph IK."
    )
    p.add_argument("x", type=float, help="Target X in cm (shoulder-frame).")
    p.add_argument("y", type=float, help="Target Y in cm (shoulder-frame).")

    p.add_argument("--inner", type=float, default=12.6, help="Inner arm length cm.")
    p.add_argument("--outer", type=float, default=14.5, help="Outer arm length cm.")

    p.add_argument("--servo0-ch", type=int, default=0, help="PCA9685 channel for shoulder.")
    p.add_argument("--servo1-ch", type=int, default=1, help="PCA9685 channel for elbow.")

    p.add_argument("--servo0-home", type=float, default=90.0,
                   help="Shoulder servo angle at physical home pose (0-180).")
    p.add_argument("--servo1-home", type=float, default=90.0,
                   help="Elbow servo angle at physical home pose (0-180).")
    p.add_argument("--motor0-home", type=float, default=-45.0,
                   help="BrachioGraph shoulder motor angle at physical home.")
    p.add_argument("--motor1-home", type=float, default=90.0,
                   help="BrachioGraph elbow motor angle at physical home.")
    p.add_argument("--k0", type=int, choices=[-1, 1], default=-1,
                   help="Shoulder direction (+1 or -1).")
    p.add_argument("--k1", type=int, choices=[-1, 1], default=-1,
                   help="Elbow direction (+1 or -1).")

    p.add_argument("--step-deg", type=float, default=0.2, help="Ramp step per tick (deg).")
    p.add_argument("--pulse-delay", type=float, default=0.01, help="Delay between ticks (s).")

    p.add_argument("--calibration", default="calibration.json",
                   help="Path to calibration JSON from capture_calibration.py.")
    p.add_argument("--no-calibration", action="store_true",
                   help="Ignore calibration file and use linear mapping.")

    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print; do not move hardware.")
    p.add_argument("--calibrate", action="store_true",
                   help="Interactive live calibration for motor0-home/motor1-home.")
    p.add_argument("--cal-step", type=float, default=2.0,
                   help="Initial step size in degrees for interactive calibration.")
    p.add_argument("--home-start0", type=float, default=START_HOME0,
                   help="Assumed current shoulder angle before startup homing ramp.")
    p.add_argument("--home-start1", type=float, default=START_HOME1,
                   help="Assumed current elbow angle before startup homing ramp.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _solve_ik_and_map(args: argparse.Namespace) -> tuple[float, float, float, float, float, float]:
    """Run IK and convert to clamped servo angles.

    Returns (shoulder_motor_deg, elbow_motor_deg,
             servo0_raw, servo1_raw, servo0_clamped, servo1_clamped).
    """
    shoulder_deg, elbow_deg = xy_to_angles(args.x, args.y, args.inner, args.outer)
    servo0 = motor_to_servo(shoulder_deg, args.servo0_home, args.motor0_home, args.k0)
    servo1 = motor_to_servo(elbow_deg, args.servo1_home, args.motor1_home, args.k1)
    return shoulder_deg, elbow_deg, servo0, servo1, clamp_angle(servo0), clamp_angle(servo1)


def _print_target_summary(
    args: argparse.Namespace,
    shoulder_deg: float,
    elbow_deg: float,
    servo0: float,
    servo1: float,
    s0_clamped: float,
    s1_clamped: float,
) -> None:
    print(f"Target (x, y) = ({args.x:.3f}, {args.y:.3f}) cm")
    print(f"IK motor angles  : shoulder={shoulder_deg:+.3f}°, elbow={elbow_deg:+.3f}°")
    print(f"Mapped servo deg : ch{args.servo0_ch}={servo0:+.3f}°, ch{args.servo1_ch}={servo1:+.3f}°")
    if servo0 != s0_clamped or servo1 != s1_clamped:
        print(
            f"WARNING: servo angle(s) clamped to 0-180: "
            f"ch{args.servo0_ch}={s0_clamped:.3f}°, ch{args.servo1_ch}={s1_clamped:.3f}°"
        )


def _connect_hardware():
    """Initialize Telemetrix + PCA9685. Returns (board, pca)."""
    from telemetrix import telemetrix
    from telemetrix_pca9685 import telemetrix_pca9685

    print("Connecting to Arduino...")
    board = telemetrix.Telemetrix()
    board.set_pin_mode_i2c()
    print("Initializing PCA9685...")
    pca = telemetrix_pca9685.TelemetrixPCA9685(board=board, i2c_address=0x40)
    try:
        pca.set_pwm_freq(50)
    except AttributeError:
        pass
    return board, pca


def main() -> int:
    args = parse_args()

    try:
        shoulder_deg, elbow_deg, servo0, servo1, s0c, s1c = _solve_ik_and_map(args)
    except ValueError as e:
        print(f"IK error: {e}", file=sys.stderr)
        return 2

    _print_target_summary(args, shoulder_deg, elbow_deg, servo0, servo1, s0c, s1c)

    if args.dry_run:
        print("Dry run: not moving hardware.")
        return 0

    try:
        board, pca = _connect_hardware()
    except ImportError as e:
        print(
            f"telemetrix/telemetrix_pca9685 not available ({e}). "
            "Run with --dry-run to just compute the angles.",
            file=sys.stderr,
        )
        return 3

    calibration = load_calibration(None if args.no_calibration else args.calibration)
    frames = {
        0: JointFrame(args.servo0_home, args.motor0_home, args.k0),
        1: JointFrame(args.servo1_home, args.motor1_home, args.k1),
    }
    controller = ServoController(
        pca=pca,
        channels=(args.servo0_ch, args.servo1_ch),
        frames=frames,
        calibration=calibration,
        step_deg=args.step_deg,
        pulse_delay=args.pulse_delay,
    )

    try:
        print(
            f"Startup home pose (fixed, linear): ch{args.servo0_ch}={START_HOME0}°, "
            f"ch{args.servo1_ch}={START_HOME1}°"
        )
        controller.move_smooth(
            args.home_start0, args.home_start1, START_HOME0, START_HOME1,
            use_calibration=False,
        )
        time.sleep(0.2)

        if args.calibrate:
            interactive_calibration(
                controller, args, shoulder_deg, elbow_deg, START_HOME0, START_HOME1
            )
        else:
            print(f"Moving to target: ch{args.servo0_ch}={s0c:.2f}°, ch{args.servo1_ch}={s1c:.2f}°")
            controller.move_smooth(START_HOME0, START_HOME1, s0c, s1c)
            print("At target. Holding position.")
            wait_for_key("p", "Press 'p' to return to home (Ctrl+C to stop).")
            print("Returning to home...")
            controller.move_smooth(
                s0c, s1c, START_HOME0, START_HOME1, use_calibration=False
            )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Shutting down...")
        board.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
