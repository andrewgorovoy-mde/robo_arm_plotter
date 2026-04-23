#!/usr/bin/env python3
"""
Interactive servo calibration capture (PCA9685 + telemetrix).

For each capture you enter the measured pen-tip position (x, y) in cm
relative to the shoulder axis. The script back-solves the relevant joint's
BrachioGraph motor angle from law-of-cosines using known link lengths
(--inner, --outer) and stores (motor_angle_deg, pulse_width, direction)
samples per servo.

Assumption: only one servo moves at a time (the other is held static).
The script captures for the servo that you most-recently nudged.

Output (calibration.json):
{
  "servo_0": {
    "angle_convention": "brachiograph_motor_deg",
    "link_lengths_cm": {"L1": 12.6, "L2": 14.5},
    "samples": [
      {"direction": "cw",  "angle": -45.123, "pw": 412, "xy_cm": [2.3, 3.2]},
      {"direction": "acw", "angle": -45.012, "pw": 408, "xy_cm": [2.31, 3.18]}
    ]
  },
  "servo_1": { ... }
}

"cw"  = pulse-width was INCREASING when this sample was captured
"acw" = pulse-width was DECREASING when this sample was captured

move_to_xy.py loads this file and per servo:
  * fits an angle->pw polynomial for cw samples and another for acw samples
    (falls back to a combined polynomial if a direction is sparse)
  * picks the cw or acw polynomial at runtime based on direction of travel

Controls (single-key, no Enter in a normal TTY):
  Shoulder (servo 0):  a=-5  A=-1  s=+5  S=+1
  Elbow    (servo 1):  k=-5  K=-1  l=+5  L=+1
  c : MARK capture -> records (servo, direction, pw); pen tip stays where it is
      You then mark a dot on the paper next to the printed capture #N.
  z : enter (x, y) coordinates for all pending marks, one at a time
  v : view current pending + finalized samples
  w : write calibration to disk
  u : undo last capture (pending first, then finalized)
  r : reset all samples (pending + finalized)
  h : show this help
  q or 0 : save + quit (warns if there are unfinished pending marks)

Suggested workflow:
  1. Set link lengths via --inner / --outer (defaults 12.6 / 14.5).
  2. Set up a paper coordinate grid with origin under the shoulder pivot.
  3. Drive servo 0 (shoulder) to a position. Press 'c' to MARK -- the
     console prints "capture #N". Mark a dot on the paper and label it N.
  4. Continue moving in the same direction (increasing pw) and pressing 'c'
     for each new pose -> these become "cw" samples for servo 0.
  5. Reverse direction (decreasing pw) and capture "acw" samples.
  6. Switch to servo 1 (elbow) and repeat.
  7. Press 'z' and enter the (x, y) of each labeled dot in order.
  8. Press 'w' (save) or 'q' (save + quit).

Aim for ~6-12 samples per servo per direction, spanning the useful range.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from typing import Optional


HELP = """
Controls:
  Shoulder (servo 0):  a=-5  A=-1  s=+5  S=+1
  Elbow    (servo 1):  k=-5  K=-1  l=+5  L=+1
  c : MARK capture (records pw + direction; mark the pen dot on paper)
  z : enter (x, y) for all pending marks (in order)
  v : view current pending + finalized samples
  w : write calibration to disk
  u : undo last capture (pending first, then finalized)
  r : reset all samples
  h : help
  q or 0 : save + quit
"""

ANGLE_CONVENTION = "brachiograph_motor_deg"


class TTYInput:
    """Single-key reads in cbreak mode, with a temporary cooked fallback for prompts."""

    def __enter__(self) -> "TTYInput":
        self.is_tty = sys.stdin.isatty()
        if self.is_tty:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read_key(self, timeout: Optional[float] = None) -> str:
        if not self.is_tty:
            line = sys.stdin.readline()
            return line.strip()[:1] if line else ""
        if timeout is not None:
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if not r:
                return ""
        return sys.stdin.read(1)

    def read_line(self, prompt: str) -> str:
        if self.is_tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        try:
            return input(prompt)
        finally:
            if self.is_tty:
                tty.setcbreak(self.fd)

    def __exit__(self, *_):
        if self.is_tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def parse_xy(line: str) -> Optional[tuple[float, float]]:
    """Parse 'x y' or 'x, y' or 'x,y'. Returns None if invalid."""
    cleaned = line.replace(",", " ").split()
    if len(cleaned) != 2:
        return None
    try:
        return float(cleaned[0]), float(cleaned[1])
    except ValueError:
        return None


def compute_motor_angle(servo_idx: int, x: float, y: float, L1: float, L2: float) -> float:
    """Return the BrachioGraph motor angle (degrees) for the moving servo.

    servo_idx == 0 -> shoulder; uses both r and atan2(x, y).
    servo_idx == 1 -> elbow;    uses only r.

    Raises ValueError if (x, y) is unreachable for the given link lengths.
    """
    r = math.hypot(x, y)
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    if r > max_reach + 1e-9 or r < min_reach - 1e-9:
        raise ValueError(
            f"(x, y) = ({x:.3f}, {y:.3f}) -> r={r:.3f}cm is outside reach "
            f"[{min_reach:.3f}, {max_reach:.3f}] for L1={L1}, L2={L2}."
        )

    if servo_idx == 1:
        outer_cos = (L1 * L1 + L2 * L2 - r * r) / (2.0 * L1 * L2)
        outer_cos = max(-1.0, min(1.0, outer_cos))
        outer_angle = math.acos(outer_cos)
        return math.degrees(math.pi - outer_angle)

    # shoulder (servo_idx == 0)
    inner_cos = (r * r + L1 * L1 - L2 * L2) / (2.0 * r * L1)
    inner_cos = max(-1.0, min(1.0, inner_cos))
    inner_angle = math.acos(inner_cos)
    hypotenuse_angle = math.atan2(x, y)
    return math.degrees(hypotenuse_angle - inner_angle)


def load_existing(path: str, L1: float, L2: float) -> dict:
    """Load existing samples in either the new (samples list) or old (bidi dict) format."""
    samples = {"servo_0": [], "servo_1": []}
    if not path or not os.path.exists(path):
        return samples
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: couldn't load {path}: {e}")
        return samples

    for k in ("servo_0", "servo_1"):
        block = data.get(k, {})
        if "samples" in block:
            samples[k] = list(block["samples"])
        elif "angle_pws_bidi" in block:
            for angle_str, pws in block["angle_pws_bidi"].items():
                try:
                    angle = float(angle_str)
                except ValueError:
                    continue
                if pws.get("cw") is not None:
                    samples[k].append({"direction": "cw", "angle": angle, "pw": pws["cw"]})
                if pws.get("acw") is not None:
                    samples[k].append({"direction": "acw", "angle": angle, "pw": pws["acw"]})
    n0, n1 = len(samples["servo_0"]), len(samples["servo_1"])
    print(f"Loaded existing calibration from {path} (servo_0: {n0}, servo_1: {n1} samples)")
    return samples


def save(path: str, samples: dict, L1: float, L2: float) -> None:
    out = {}
    for k in ("servo_0", "servo_1"):
        sorted_samples = sorted(samples[k], key=lambda s: s.get("angle", 0.0))
        out[k] = {
            "angle_convention": ANGLE_CONVENTION,
            "link_lengths_cm": {"L1": L1, "L2": L2},
            "samples": sorted_samples,
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="calibration.json",
                        help="Where to save calibration (default: calibration.json).")
    parser.add_argument("--load", default=None,
                        help="Existing calibration.json to seed samples. "
                             "Defaults to --out if it exists.")
    parser.add_argument("--inner", type=float, default=12.6,
                        help="Inner arm length L1 in cm (shoulder->elbow).")
    parser.add_argument("--outer", type=float, default=14.5,
                        help="Outer arm length L2 in cm (elbow->pen tip).")
    parser.add_argument("--servo0-ch", type=int, default=0)
    parser.add_argument("--servo1-ch", type=int, default=1)

    # Motor-frame mapping (must match move_to_xy.py) so we can also print
    # the linear-default estimated motor angle for each captured pw, alongside
    # the LoC-derived "real" angle from the measured (x, y).
    parser.add_argument("--servo0-home", type=float, default=90.0,
                        help="Shoulder servo angle (0-180) at the physical home pose.")
    parser.add_argument("--servo1-home", type=float, default=90.0,
                        help="Elbow servo angle (0-180) at the physical home pose.")
    parser.add_argument("--motor0-home", type=float, default=-45.0,
                        help="BrachioGraph shoulder motor angle at physical home pose.")
    parser.add_argument("--motor1-home", type=float, default=90.0,
                        help="BrachioGraph elbow motor angle at physical home pose.")
    parser.add_argument("--k0", type=int, choices=[-1, 1], default=-1,
                        help="Shoulder direction (+1 or -1).")
    parser.add_argument("--k1", type=int, choices=[-1, 1], default=-1,
                        help="Elbow direction (+1 or -1).")
    parser.add_argument("--linear-pw-min", type=int, default=150,
                        help="Linear default: pulse-width at servo angle 0°.")
    parser.add_argument("--linear-pw-max", type=int, default=600,
                        help="Linear default: pulse-width at servo angle 180°.")
    parser.add_argument("--start-pw-0", type=int, default=375,
                        help="Initial pulse-width for servo 0 (PCA9685 12-bit count, ~mid).")
    parser.add_argument("--start-pw-1", type=int, default=375)
    parser.add_argument("--pw-min", type=int, default=100, help="Safety min pw.")
    parser.add_argument("--pw-max", type=int, default=650, help="Safety max pw.")
    parser.add_argument("--i2c-address", type=lambda x: int(x, 0), default=0x40)
    parser.add_argument("--coarse-step", type=int, default=5,
                        help="Pulse-width step for a/s/k/l keys.")
    parser.add_argument("--fine-step", type=int, default=1,
                        help="Pulse-width step for A/S/K/L keys.")
    args = parser.parse_args()

    L1, L2 = args.inner, args.outer
    load_path = args.load if args.load is not None else args.out
    samples = load_existing(load_path, L1, L2)

    frame = {
        0: {"servo_home": args.servo0_home, "motor_home": args.motor0_home, "k": args.k0},
        1: {"servo_home": args.servo1_home, "motor_home": args.motor1_home, "k": args.k1},
    }
    pw_lo, pw_hi = args.linear_pw_min, args.linear_pw_max

    def estimate_motor_angle_from_pw(servo_idx: int, pw: int) -> float:
        """Default linear inverse: pw -> servo angle (0-180), then to motor angle."""
        servo_angle = (pw - pw_lo) * 180.0 / (pw_hi - pw_lo)
        f = frame[servo_idx]
        return f["motor_home"] + (servo_angle - f["servo_home"]) / f["k"]

    try:
        from telemetrix import telemetrix
        from telemetrix_pca9685 import telemetrix_pca9685
    except ImportError as e:
        print(f"telemetrix/telemetrix_pca9685 not available ({e}).", file=sys.stderr)
        return 3

    print("Connecting to Arduino...")
    board = telemetrix.Telemetrix()
    board.set_pin_mode_i2c()
    print("Initializing PCA9685...")
    pca = telemetrix_pca9685.TelemetrixPCA9685(board=board, i2c_address=args.i2c_address)
    try:
        pca.set_pwm_freq(50)
    except AttributeError:
        pass

    def clamp(pw: int) -> int:
        return max(args.pw_min, min(args.pw_max, pw))

    state = {
        0: {"ch": args.servo0_ch, "pw": clamp(args.start_pw_0), "dir": None},
        1: {"ch": args.servo1_ch, "pw": clamp(args.start_pw_1), "dir": None},
    }

    def apply(i: int) -> None:
        pca.set_pwm(state[i]["ch"], 0, state[i]["pw"])

    apply(0)
    apply(1)
    time.sleep(0.3)

    def nudge(i: int, delta: int) -> None:
        new_pw = clamp(state[i]["pw"] + delta)
        if new_pw == state[i]["pw"]:
            return
        state[i]["dir"] = "cw" if new_pw > state[i]["pw"] else "acw"
        state[i]["pw"] = new_pw
        apply(i)

    cs, fs = args.coarse_step, args.fine_step
    key_map = {
        "a": (0, -cs), "A": (0, -fs), "s": (0, +cs), "S": (0, +fs),
        "k": (1, -cs), "K": (1, -fs), "l": (1, +cs), "L": (1, +fs),
    }

    def print_status() -> None:
        print(
            f"servo0 pw={state[0]['pw']:4d} (last_dir={state[0]['dir'] or '-'})  "
            f"servo1 pw={state[1]['pw']:4d} (last_dir={state[1]['dir'] or '-'})"
        )

    def view() -> None:
        for k in ("servo_0", "servo_1"):
            print(f"\n{k}: {len(samples[k])} finalized samples")
            for s in sorted(samples[k], key=lambda x: x.get("angle", 0.0)):
                xy = s.get("xy_cm")
                xy_str = f" xy=({xy[0]:.3f}, {xy[1]:.3f})" if xy else ""
                ident = f"#{s.get('id'):>3} " if s.get("id") is not None else "      "
                print(
                    f"  {ident}[{s['direction']:>3}] angle={s['angle']:+8.3f}°  "
                    f"pw={s['pw']:4d}{xy_str}"
                )
        if pending:
            print(f"\nPending captures (need (x, y)): {len(pending)}")
            for p in pending:
                print(
                    f"  #{p['id']:>3} servo_{p['servo_idx']} "
                    f"[{p['direction']:>3}] pw={p['pw']:4d}"
                )

    last_moved_servo: Optional[int] = None
    pending: list[dict] = []
    capture_counter = 0

    # Last action for undo: ("pending", entry) or ("sample", servo_key, sample)
    last_action: Optional[tuple] = None

    print(HELP)
    print(f"Link lengths: L1={L1} cm (inner), L2={L2} cm (outer)")
    print(f"Reach: r ∈ [{abs(L1 - L2):.2f}, {L1 + L2:.2f}] cm")
    print_status()

    try:
        with TTYInput() as kb:
            while True:
                key = kb.read_key()
                if not key:
                    continue

                if key in key_map:
                    i, d = key_map[key]
                    nudge(i, d)
                    last_moved_servo = i
                    print_status()

                elif key == "c":
                    if last_moved_servo is None:
                        print("Move a servo first, then press 'c'.")
                        continue
                    i = last_moved_servo
                    pw = state[i]["pw"]
                    direction = state[i]["dir"] or "cw"
                    capture_counter += 1
                    entry = {
                        "id": capture_counter,
                        "servo_idx": i,
                        "direction": direction,
                        "pw": pw,
                    }
                    pending.append(entry)
                    last_action = ("pending", entry)
                    name = "shoulder" if i == 0 else "elbow"
                    print(
                        f"Marked #{capture_counter}: {name} (servo {i}) "
                        f"{direction} pw={pw}  --  mark this dot on the paper"
                        f"  (pending: {len(pending)})"
                    )

                elif key == "z":
                    if not pending:
                        print("No pending captures. Press 'c' first to mark some.")
                        continue
                    print(
                        f"\nEntering coordinates for {len(pending)} pending capture(s).\n"
                        "  Format: 'x y' or 'x, y' (cm). Blank to skip (keep pending).\n"
                        "  Type 's' to stop early."
                    )
                    remaining: list[dict] = []
                    aborted = False
                    for entry in pending:
                        if aborted:
                            remaining.append(entry)
                            continue
                        name = "shoulder" if entry["servo_idx"] == 0 else "elbow"
                        line = kb.read_line(
                            f"#{entry['id']} {name} (servo {entry['servo_idx']}) "
                            f"{entry['direction']} pw={entry['pw']} -> (x, y) cm: "
                        ).strip()
                        if not line:
                            print("  (skipped, kept pending)")
                            remaining.append(entry)
                            continue
                        if line.lower() == "s":
                            aborted = True
                            remaining.append(entry)
                            continue
                        parsed = parse_xy(line)
                        if parsed is None:
                            print("  Could not parse, kept pending.")
                            remaining.append(entry)
                            continue
                        x, y = parsed
                        try:
                            motor_angle = compute_motor_angle(
                                entry["servo_idx"], x, y, L1, L2
                            )
                        except ValueError as e:
                            print(f"  Geometry error: {e}; kept pending.")
                            remaining.append(entry)
                            continue
                        sample = {
                            "id": entry["id"],
                            "direction": entry["direction"],
                            "angle": round(motor_angle, 4),
                            "pw": entry["pw"],
                            "xy_cm": [x, y],
                        }
                        samples[f"servo_{entry['servo_idx']}"].append(sample)
                        last_action = ("sample", f"servo_{entry['servo_idx']}", sample)
                        estimated = estimate_motor_angle_from_pw(
                            entry["servo_idx"], entry["pw"]
                        )
                        delta = motor_angle - estimated
                        print(
                            f"  Finalized #{entry['id']}: "
                            f"r={math.hypot(x, y):.3f}cm  "
                            f"est={estimated:+7.3f}°  real={motor_angle:+7.3f}°  "
                            f"Δ={delta:+7.3f}°"
                        )
                    pending = remaining
                    print(f"Done. Pending remaining: {len(pending)}")

                elif key == "u":
                    if last_action is None:
                        print("Nothing to undo.")
                        continue
                    if last_action[0] == "pending":
                        entry = last_action[1]
                        if entry in pending:
                            pending.remove(entry)
                            print(
                                f"Undid pending #{entry['id']} "
                                f"(servo_{entry['servo_idx']} pw={entry['pw']})"
                            )
                        else:
                            print("Last pending no longer in list.")
                    else:
                        _, skey, sample = last_action
                        try:
                            samples[skey].remove(sample)
                            print(
                                f"Undid finalized #{sample.get('id', '?')} "
                                f"in {skey} angle={sample['angle']:+.3f}°"
                            )
                        except ValueError:
                            print("Last finalized sample no longer in list.")
                    last_action = None

                elif key == "v":
                    view()

                elif key == "w":
                    if pending:
                        print(
                            f"Note: {len(pending)} pending capture(s) without (x, y) "
                            "won't be saved. Press 'z' first to fill them in."
                        )
                    save(args.out, samples, L1, L2)

                elif key == "r":
                    confirm = kb.read_line("Reset all samples (incl. pending)? (y/N): ").strip().lower()
                    if confirm == "y":
                        samples = {"servo_0": [], "servo_1": []}
                        pending = []
                        capture_counter = 0
                        last_action = None
                        print("Cleared all samples and pending captures.")

                elif key == "h":
                    print(HELP)

                elif key in ("q", "0"):
                    if pending:
                        ans = kb.read_line(
                            f"{len(pending)} pending capture(s) won't be saved. "
                            "Save and quit anyway? (y/N): "
                        ).strip().lower()
                        if ans != "y":
                            continue
                    save(args.out, samples, L1, L2)
                    break

                else:
                    print(f"(unknown key: {key!r}) press 'h' for help")
    except KeyboardInterrupt:
        print("\nInterrupted. Saving what we have.")
        try:
            save(args.out, samples, L1, L2)
        except Exception as e:
            print(f"Save failed: {e}", file=sys.stderr)
    finally:
        print("Shutting down...")
        try:
            board.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
