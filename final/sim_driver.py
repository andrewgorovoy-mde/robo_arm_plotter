"""
Simulation driver for plotter_v2.py.

Drop-in replacement for the telemetrix board + PCA9685 PWM driver. It
intercepts every `set_pwm(channel, on, pulse)` call, inverts the calibration
polynomial to estimate each servo's *physical* angle, runs forward kinematics
to compute the pen-tip position, and renders a live matplotlib animation of
the linkage and the page being drawn.

Usage:
    PLOTTER_SIM=1 python3 final/plotter_v2.py

Realism dials (set via configure()):
    sim_hysteresis_pulse  : real servo backlash, in pulse counts (the
                            plotter's own HYSTERESIS array tries to cancel it;
                            non-zero here lets you study residual error)
    sim_servo_noise_deg   : uniform random noise added to physical angle each
                            update; 0 disables.
    sim_tau_s             : 1st-order low-pass time constant on the physical
                            angle, modelling servo settling lag (0 = instant).

The recorder is thread-safe, so the production keyboard loop can run in a
worker thread while matplotlib's animation runs on the main thread (required
on macOS).
"""

import math
import os
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox


# ---------------------------------------------------------
# Module-level configuration (populated by configure())
# ---------------------------------------------------------
_cfg = {
    "L1": 10.2,
    "L2": 11.0,
    "K0": -1,
    "K1": -1,
    "home_angles": (90.0, 90.0),
    "servo_ch_0": 0,
    "servo_ch_1": 1,
    "servo_ch_pen": 2,
    "pen_up_pulse": 300,
    "pen_down_pulse": 400,
    "servo_cal": [None, None],         # [SERVO0_CAL, SERVO1_CAL]
    "joint_limits": [(0.0, 90.0), (40.0, 160.0)],
    "draw_box": None,                  # (cx, cy, w, h) in cm, optional
    # Hysteresis the production driver adds to the commanded pulse to cancel
    # real servo backlash. The sim subtracts this so the simulated tip
    # doesn't jitter when commanded angle reverses direction. Set to [0, 0]
    # if you want to *see* the production correction in the sim.
    "prod_hysteresis_pulse": [0, 0],
    # Real-world servo backlash to inject *back into* the sim. Non-zero here
    # lets you study residual error when production HYSTERESIS is mistuned.
    "sim_hysteresis_pulse": [0, 0],
    "sim_servo_noise_deg": 0.0,
    "sim_tau_s": 0.0,
}

_inv_cal = [None, None]      # pulse -> physical-angle polynomial per servo
_recorder = None
_shutdown_event = threading.Event()


def configure(**kwargs):
    """Update sim configuration. Safe to call repeatedly. After this, the
    inverse calibration polynomials are rebuilt from `servo_cal`."""
    _cfg.update(kwargs)
    if "servo_cal" in kwargs:
        for i, cal in enumerate(_cfg["servo_cal"]):
            if cal is None or len(cal) < 2:
                continue
            arr = np.array(cal, dtype=float)
            deg = 3 if len(arr) >= 4 else 1
            # fit pulse -> angle (i.e. swap the columns vs the production fit)
            _inv_cal[i] = np.poly1d(np.polyfit(arr[:, 1], arr[:, 0], deg))


def request_shutdown():
    """Ask the animation main loop to close (called by the keyboard thread
    after the user presses 'q')."""
    _shutdown_event.set()


# ---------------------------------------------------------
# Kinematics
# ---------------------------------------------------------
def forward_kin(s0, s1):
    """Servo angles -> (elbow_x, elbow_y, tip_x, tip_y) in cm.
    Inverse of plotter_v2's xy_to_servo_angles, with K0=K1=-1."""
    L1, L2 = _cfg["L1"], _cfg["L2"]
    theta1 = math.radians(90.0 - s0)
    phi    = math.radians(180.0 - s1)
    ex = L1 * math.cos(theta1)
    ey = L1 * math.sin(theta1)
    tx = ex + L2 * math.cos(theta1 + phi)
    ty = ey + L2 * math.sin(theta1 + phi)
    return ex, ey, tx, ty


def _pulse_to_phys_angle(ch, pulse, prev_phys, last_pulse, last_dir, dt):
    """Estimate the physical angle the servo will end up at, given a
    commanded pulse. Returns `(phys_angle, new_direction)` so the caller can
    persist the inferred motion direction across calls.

    Three error sources are modeled explicitly:
      - production hysteresis correction (subtracted out so the sim shows the
        servo's *intended* angle, not the pre-corrected pulse)
      - simulated real backlash (added back in if requested)
      - 1st-order settling lag (low-pass between commanded and actual)

    `last_dir` is +1 / -1 / 0; we update it whenever the pulse moves more
    than ~0.5 counts so steady-state plateaus don't reset the direction."""
    if _inv_cal[ch] is None:
        return prev_phys, last_dir

    eff_pulse = float(pulse)
    direction = last_dir
    if last_pulse is not None:
        delta = eff_pulse - last_pulse
        if delta > 0.5:
            direction = +1
        elif delta < -0.5:
            direction = -1

    # Undo the production hysteresis (it added +H going up, -H going down,
    # to cancel real servo backlash; in the ideal sim there's no backlash
    # to cancel so the +/-H jumps look like jitter).
    h_prod = _cfg["prod_hysteresis_pulse"][ch]
    if h_prod and direction != 0:
        eff_pulse -= direction * h_prod

    # Inject simulated real-world backlash (off by default).
    h_real = _cfg["sim_hysteresis_pulse"][ch]
    if h_real and direction != 0:
        eff_pulse += direction * h_real

    target = float(_inv_cal[ch](eff_pulse))

    if _cfg["sim_servo_noise_deg"]:
        target += float(np.random.uniform(-1.0, 1.0)) * _cfg["sim_servo_noise_deg"]

    tau = _cfg["sim_tau_s"]
    if tau and dt > 0:
        alpha = 1.0 - math.exp(-dt / tau)
        return prev_phys + alpha * (target - prev_phys), direction
    return target, direction


# ---------------------------------------------------------
# Recorder (thread-safe)
# ---------------------------------------------------------
class SimRecorder:
    """Records every commanded pulse and translates it into a simulated pen
    position. The matplotlib animation reads `snapshot()` each frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self.angles_phys = list(_cfg["home_angles"])
        self.pen_down = False
        self.last_pulse = [None, None]
        self.last_dir = [0, 0]
        self.last_t = [time.monotonic(), time.monotonic()]

        ex, ey, tx, ty = forward_kin(*self.angles_phys)
        self.elbow_xy = (ex, ey)
        self.tip_xy = (tx, ty)

        self.completed_segs = []     # list of (xs, ys) for completed strokes
        self.cur_seg_x = []
        self.cur_seg_y = []
        self.command_count = 0
        self.generation = 0          # bumps on erase() so the animation can
                                     # detect it and drop its Line2D objects

    def notify_pulse(self, channel, pulse):
        ch_s0  = _cfg["servo_ch_0"]
        ch_s1  = _cfg["servo_ch_1"]
        ch_pen = _cfg["servo_ch_pen"]
        with self._lock:
            self.command_count += 1
            if channel == ch_s0 or channel == ch_s1:
                idx = 0 if channel == ch_s0 else 1
                now = time.monotonic()
                dt = now - self.last_t[idx]
                new_angle, new_dir = _pulse_to_phys_angle(
                    idx, pulse, self.angles_phys[idx],
                    self.last_pulse[idx], self.last_dir[idx], dt
                )
                self.angles_phys[idx] = new_angle
                self.last_dir[idx] = new_dir
                self.last_pulse[idx] = float(pulse)
                self.last_t[idx] = now
                ex, ey, tx, ty = forward_kin(*self.angles_phys)
                self.elbow_xy = (ex, ey)
                self.tip_xy = (tx, ty)
                if self.pen_down:
                    self.cur_seg_x.append(tx)
                    self.cur_seg_y.append(ty)
            elif channel == ch_pen:
                want_down = abs(pulse - _cfg["pen_down_pulse"]) < \
                            abs(pulse - _cfg["pen_up_pulse"])
                if want_down and not self.pen_down:
                    self.cur_seg_x = [self.tip_xy[0]]
                    self.cur_seg_y = [self.tip_xy[1]]
                elif (not want_down) and self.pen_down:
                    if len(self.cur_seg_x) >= 2:
                        self.completed_segs.append(
                            (list(self.cur_seg_x), list(self.cur_seg_y))
                        )
                    self.cur_seg_x = []
                    self.cur_seg_y = []
                self.pen_down = want_down

    def erase(self):
        """Wipe all completed strokes and the in-progress segment. The
        physical arm state (angles, pen) is unchanged. The animation watches
        `generation` and removes Line2D objects when it changes."""
        with self._lock:
            self.completed_segs = []
            self.cur_seg_x = []
            self.cur_seg_y = []
            self.generation += 1

    def snapshot(self):
        with self._lock:
            return dict(
                angles=tuple(self.angles_phys),
                elbow=self.elbow_xy,
                tip=self.tip_xy,
                pen=self.pen_down,
                segs=list(self.completed_segs),
                cur=(list(self.cur_seg_x), list(self.cur_seg_y)),
                cmds=self.command_count,
                generation=self.generation,
            )


def get_recorder():
    """Return the singleton recorder, creating it on first call."""
    global _recorder
    if _recorder is None:
        _recorder = SimRecorder()
    return _recorder


# ---------------------------------------------------------
# Hardware stand-ins
# ---------------------------------------------------------
class SimBoard:
    """Stand-in for telemetrix.Telemetrix() - does nothing."""

    def set_pin_mode_i2c(self):
        pass

    def shutdown(self):
        pass


class SimPCA9685:
    """Stand-in for telemetrix_pca9685.TelemetrixPCA9685 - records pulses
    instead of sending I2C commands."""

    def __init__(self, recorder=None):
        self._rec = recorder if recorder is not None else get_recorder()

    def set_pwm(self, channel, on, pulse):
        self._rec.notify_pulse(channel, pulse)

    def set_pwm_freq(self, freq):
        pass


# ---------------------------------------------------------
# Matplotlib default keymap conflicts
# ---------------------------------------------------------
# Matplotlib binds 'g', 's', 'q', 'p', 'o', 'l', 'f', etc. by default. When
# the figure has focus those keystrokes never reach the terminal, so the
# plotter's keyboard shortcuts get hijacked. Disable them here so users
# can rely on the on-figure controls (and still type in the terminal if they
# want).
_KEYMAPS_TO_CLEAR = (
    "save", "fullscreen", "quit", "quit_all",
    "grid", "grid_minor", "yscale", "xscale",
    "pan", "zoom", "back", "forward", "home",
)
for _k in _KEYMAPS_TO_CLEAR:
    try:
        plt.rcParams[f"keymap.{_k}"] = []
    except KeyError:
        pass


# ---------------------------------------------------------
# Live animation + on-figure controls
# ---------------------------------------------------------
def start_animation(title="Plotter Simulation",
                    command_queue=None,
                    available_images=None,
                    default_image=None):
    """Open a matplotlib window with two synchronized panels and (when a
    command queue is provided) an on-figure control panel. Each control
    pushes a command tuple — same format used by the keyboard handler —
    so UI clicks and key presses share the same dispatcher.

    Blocks until the window closes or `request_shutdown()` is called."""
    rec = get_recorder()

    have_controls = command_queue is not None
    fig = plt.figure(figsize=(14, 10 if have_controls else 7))
    fig.suptitle(title, fontsize=12, y=0.97)

    plot_top    = 0.94
    plot_bottom = 0.36 if have_controls else 0.10

    gs = fig.add_gridspec(1, 2,
                          left=0.06, right=0.96,
                          top=plot_top, bottom=plot_bottom,
                          wspace=0.18)
    ax_arm = fig.add_subplot(gs[0])
    ax_paper = fig.add_subplot(gs[1])

    L1, L2 = _cfg["L1"], _cfg["L2"]
    R = L1 + L2

    for ax in (ax_arm, ax_paper):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

    ax_arm.set_xlim(-R - 1, R + 1)
    ax_arm.set_ylim(-2, R + 1)
    ax_arm.set_title("Arm linkage")

    box = _cfg["draw_box"]
    if box:
        cx, cy, w, h = box
        ax_paper.set_xlim(cx - w / 2 - 2, cx + w / 2 + 2)
        ax_paper.set_ylim(cy - h / 2 - 2, cy + h / 2 + 2)
        for ax in (ax_arm, ax_paper):
            ax.add_patch(Rectangle(
                (cx - w / 2, cy - h / 2), w, h,
                fill=False, linestyle="--", edgecolor="gray", linewidth=1,
            ))
    else:
        ax_paper.set_xlim(-R, R)
        ax_paper.set_ylim(-2, R + 1)
    ax_paper.set_title("Paper view")

    # mount point + shoulder origin marker
    ax_arm.plot([0], [0], "ks", ms=8, mfc="0.3")

    arm1_line, = ax_arm.plot([], [], "-", color="#1f77b4", lw=4, solid_capstyle="round")
    arm2_line, = ax_arm.plot([], [], "-", color="#d62728", lw=4, solid_capstyle="round")
    elbow_dot, = ax_arm.plot([], [], "o", color="#444", ms=6)
    pen_dot,   = ax_arm.plot([], [], "o", color="black", ms=8)
    arm_cur,   = ax_arm.plot([], [], "-", color="#2ca02c", lw=1.0, alpha=0.8)
    paper_cur, = ax_paper.plot([], [], "-", color="black", lw=0.9)

    # We append matplotlib lines as new segments complete so they persist.
    paper_seg_lines = []
    arm_seg_lines = []

    status = ax_arm.text(
        0.02, 0.98, "", transform=ax_arm.transAxes,
        fontsize=9, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                  edgecolor="0.7"),
    )

    # ---------------------------------------------------------
    # On-figure controls
    # ---------------------------------------------------------
    widget_refs = []  # keep widgets alive past function exit
    msg_text = None

    if have_controls:
        def _push(cmd):
            command_queue.put(cmd)

        def _flash(text):
            """Set the status message at the bottom of the controls panel."""
            if msg_text is not None:
                msg_text.set_text(text)
                fig.canvas.draw_idle()

        # Row 1 — single-action buttons (y from 0.24 to 0.30)
        def _make_button(rect, label, cmd, color="#dddddd", hover="#bbbbbb"):
            ax = fig.add_axes(rect)
            btn = Button(ax, label, color=color, hovercolor=hover)
            btn.on_clicked(lambda evt: (_push(cmd), _flash(f"queued: {cmd[0]}")))
            widget_refs.append(btn)
            return btn

        _make_button([0.06, 0.255, 0.07, 0.045], "Home",     ("home",))
        _make_button([0.14, 0.255, 0.07, 0.045], "Pen Up",   ("pen_up",))
        _make_button([0.22, 0.255, 0.07, 0.045], "Pen Down", ("pen_down",))
        _make_button([0.30, 0.255, 0.08, 0.045], "Box Edge", ("trace_box",))
        _make_button([0.39, 0.255, 0.07, 0.045], "Square",   ("square",))
        _make_button([0.47, 0.255, 0.07, 0.045], "Star",     ("star",))
        _make_button([0.55, 0.255, 0.08, 0.045], "Erase",    ("erase",),
                     color="#fde0a8", hover="#f7c15c")
        _make_button([0.88, 0.255, 0.08, 0.045], "Quit",     ("quit",),
                     color="#f4cccc", hover="#e69999")

        # Row 2 — Goto x y (y from 0.16 to 0.21)
        ax_xy = fig.add_axes([0.13, 0.165, 0.13, 0.045])
        tb_xy = TextBox(ax_xy, "Goto x y: ", initial="10 11",
                        textalignment="left")
        widget_refs.append(tb_xy)

        ax_goto = fig.add_axes([0.27, 0.165, 0.06, 0.045])
        b_goto  = Button(ax_goto, "Go", color="#cfe2f3", hovercolor="#9fc5e8")
        widget_refs.append(b_goto)

        def _on_goto(_evt):
            raw = tb_xy.text.strip()
            if not raw:
                _flash("Goto: enter 'x y' in cm")
                return
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                _flash(f"Goto: expected 'x y', got {raw!r}")
                return
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                _flash(f"Goto: cannot parse {raw!r}")
                return
            _push(("goto", x, y))
            _flash(f"queued: goto ({x:.2f}, {y:.2f})")
        b_goto.on_clicked(_on_goto)
        tb_xy.on_submit(lambda _t: _on_goto(None))

        # Row 2 — Image trace (y from 0.16 to 0.21)
        ax_img = fig.add_axes([0.43, 0.165, 0.32, 0.045])
        tb_img = TextBox(ax_img, "Image: ",
                         initial=default_image or "",
                         textalignment="left")
        widget_refs.append(tb_img)

        ax_trace = fig.add_axes([0.76, 0.165, 0.07, 0.045])
        b_trace  = Button(ax_trace, "Trace", color="#d9ead3", hovercolor="#a8d18d")
        widget_refs.append(b_trace)

        def _on_trace(_evt):
            path = tb_img.text.strip()
            if not path:
                _flash("Trace: enter an image path")
                return
            _push(("trace_image", path))
            _flash(f"queued: trace {path}")
        b_trace.on_clicked(_on_trace)
        tb_img.on_submit(lambda _t: _on_trace(None))

        # Row 3 — hint about available images and a status line
        if available_images:
            names = [os.path.basename(p) for p in available_images]
            shown = ", ".join(names[:8])
            more = f"  (+{len(names) - 8} more)" if len(names) > 8 else ""
            fig.text(0.43, 0.115, f"Available: {shown}{more}",
                     fontsize=8, color="0.4", family="monospace")

        msg_text = fig.text(0.06, 0.06, "", fontsize=9, color="#1a73e8",
                            family="monospace")
        fig.text(0.06, 0.025,
                 "Click buttons above, type in the boxes (Enter submits), "
                 "or type single-key commands in the terminal.",
                 fontsize=8, color="0.4")

        # Keep a reference so widgets aren't garbage-collected.
        fig._sim_widgets = widget_refs  # type: ignore[attr-defined]

    def init():
        return (arm1_line, arm2_line, elbow_dot, pen_dot,
                arm_cur, paper_cur, status)

    last_generation = [rec.snapshot()["generation"]]

    def update(_):
        if _shutdown_event.is_set():
            plt.close(fig)
            return (arm1_line, arm2_line, elbow_dot, pen_dot,
                    arm_cur, paper_cur, status)

        s = rec.snapshot()

        # Erase: drop every persisted segment line so the page goes blank.
        if s["generation"] != last_generation[0]:
            for ln in paper_seg_lines:
                ln.remove()
            for ln in arm_seg_lines:
                ln.remove()
            paper_seg_lines.clear()
            arm_seg_lines.clear()
            last_generation[0] = s["generation"]

        ex, ey = s["elbow"]
        tx, ty = s["tip"]

        arm1_line.set_data([0, ex], [0, ey])
        arm2_line.set_data([ex, tx], [ey, ty])
        elbow_dot.set_data([ex], [ey])
        pen_dot.set_data([tx], [ty])
        if s["pen"]:
            pen_dot.set_marker("o")
            pen_dot.set_markerfacecolor("black")
        else:
            pen_dot.set_marker("o")
            pen_dot.set_markerfacecolor("none")

        cur_xs, cur_ys = s["cur"]
        arm_cur.set_data(cur_xs, cur_ys)
        paper_cur.set_data(cur_xs, cur_ys)

        # add any newly-completed segments (one matplotlib Line2D each so
        # they stay rendered without redrawing the whole history every frame)
        while len(paper_seg_lines) < len(s["segs"]):
            xs, ys = s["segs"][len(paper_seg_lines)]
            ln_p, = ax_paper.plot(xs, ys, "-", color="black", lw=0.9)
            ln_a, = ax_arm.plot(xs, ys, "-", color="#2ca02c", lw=0.9, alpha=0.5)
            paper_seg_lines.append(ln_p)
            arm_seg_lines.append(ln_a)

        ang0, ang1 = s["angles"]
        s0lim = _cfg["joint_limits"][0]
        s1lim = _cfg["joint_limits"][1]
        warn0 = "!" if (ang0 < s0lim[0] or ang0 > s0lim[1]) else " "
        warn1 = "!" if (ang1 < s1lim[0] or ang1 > s1lim[1]) else " "
        status.set_text(
            f"servo0: {ang0:6.2f}° {warn0}    servo1: {ang1:6.2f}° {warn1}\n"
            f"tip:    ({tx:6.2f}, {ty:6.2f}) cm     pen: "
            f"{'DOWN' if s['pen'] else 'UP  '}\n"
            f"strokes: {len(s['segs'])}{' (drawing...)' if s['pen'] else ''}"
            f"   pwm cmds: {s['cmds']}"
        )
        return (arm1_line, arm2_line, elbow_dot, pen_dot,
                arm_cur, paper_cur, status)

    # cache_frame_data=False so memory doesn't grow unboundedly
    ani = FuncAnimation(
        fig, update, init_func=init, interval=33,
        blit=False, cache_frame_data=False,
    )
    # Keep a reference so the animation isn't garbage-collected.
    fig._sim_animation = ani  # type: ignore[attr-defined]

    plt.show()
    _shutdown_event.set()
    return ani
