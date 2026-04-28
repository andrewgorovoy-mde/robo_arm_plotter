"""Interactive servo-calibration capture.

Lets you nudge each servo with a known set of keys, type the physical angle
you measured by hand, and prints a Python-formatted calibration table that
can be pasted back into the plotter script. Optionally fits a cubic
polynomial to the captured pairs and updates a live plotter at runtime.
"""


def capture_calibration(*, pca, read_key, build_cal_poly,
                        servo_ch_0, servo_ch_1, prev_pulse,
                        pulse_min, pulse_max,
                        update_polynomials):
    """Run the interactive calibration capture loop.

    Parameters (all keyword-only so this stays explicit):
      pca             - PCA9685 object with `set_pwm(channel, on, pulse)`.
      read_key        - function returning a single keystroke.
      build_cal_poly  - callable(list_of_(angle, pulse)) -> np.poly1d.
      servo_ch_0      - PCA channel for servo 0.
      servo_ch_1      - PCA channel for servo 1.
      prev_pulse      - 2-element list with the last commanded pulse per servo
                        (used as the starting position; ignored if 0).
      pulse_min/_max  - safety clamp range for raw pulses.
      update_polynomials - callable(poly0, poly1) used to install fresh
                        polynomials in the live plotter when calibration
                        completes with enough points.
    """
    print("\n=== CALIBRATION MODE ===")
    print("Drive each servo to a known angle, then SPACE to record.")
    print("a/s=servo0 ±10  A/S=servo0 ±1  k/l=servo1 ±10  K/L=servo1 ±1")
    print("SPACE=record  v=view so far  0=finish\n")

    pw0 = int(prev_pulse[servo_ch_0]) or 375
    pw1 = int(prev_pulse[servo_ch_1]) or 375

    def _raw(ch, pw):
        pw = int(max(pulse_min, min(pulse_max, pw)))
        pca.set_pwm(ch, 0, pw)

    _raw(servo_ch_0, pw0)
    _raw(servo_ch_1, pw1)

    cal0, cal1 = [], []
    controls = {
        "a": (-10, 0), "s": (+10, 0),
        "A": (-1,  0), "S": (+1,  0),
        "k": (0, -10), "l": (0, +10),
        "K": (0,  -1), "L": (0,  +1),
    }

    while True:
        ch = read_key()
        if ch in controls:
            d0, d1 = controls[ch]
            pw0 = int(max(pulse_min, min(pulse_max, pw0 + d0)))
            pw1 = int(max(pulse_min, min(pulse_max, pw1 + d1)))
            _raw(servo_ch_0, pw0)
            _raw(servo_ch_1, pw1)
            print(f"  pulses → servo0={pw0}  servo1={pw1}")
        elif ch == " ":
            raw = input("\n  Recording. Which servo did you just position? "
                        "(0 / 1 / both): ").strip()
            if raw in ("0", "1", "both"):
                try:
                    angle = float(input("  Physical angle in degrees: ").strip())
                except ValueError:
                    print("  Bad angle — skipping.")
                    continue
                if raw in ("0", "both"):
                    cal0.append((angle, pw0))
                    print(f"  ✓ servo0: {angle}° → pulse {pw0}")
                if raw in ("1", "both"):
                    cal1.append((angle, pw1))
                    print(f"  ✓ servo1: {angle}° → pulse {pw1}")
            else:
                print("  Enter 0, 1, or both.")
        elif ch == "v":
            print(f"\n  servo0 so far: {sorted(cal0)}")
            print(f"  servo1 so far: {sorted(cal1)}\n")
        elif ch == "0":
            break

    print("\n=== CALIBRATION RESULTS — paste into your script ===\n")
    print("SERVO0_CAL = [")
    for a, p in sorted(cal0):
        print(f"    ({a:.1f}, {p}),")
    print("]\n")
    print("SERVO1_CAL = [")
    for a, p in sorted(cal1):
        print(f"    ({a:.1f}, {p}),")
    print("]\n")

    if len(cal0) >= 2 and len(cal1) >= 2:
        print("Fitting polynomials to captured data...")
        update_polynomials(build_cal_poly(cal0), build_cal_poly(cal1))
        print("  ✓ Calibration active for this session.\n")
    else:
        print("  Not enough points to fit (need ≥ 2 per servo) — keeping old cal.\n")
    print("=== END CALIBRATION ===\n")
