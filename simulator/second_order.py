"""Interactive second-order plant and PID experiment. Run this file directly."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_gen import Plant, PIDGains, PlantSimulator


def simulate(zeta=0.2, wn=1.0, gain=1.0, kp=1.0, ki=0.5,
             kd=0.5, setpoint=1.0, duration=30.0, delay=0.2):
    """Output and regular input response for ideal PID and second-order Pade delay.

    The input additionally contains setpoint*kd*delta(t), not a finite spike.
    Uses exactly the same model as the dataset generator.
    """
    simulator = PlantSimulator()
    plant = Plant(zeta, wn, delay, gain)
    simulator.validate_plant(plant)
    if not np.all(np.isfinite([kp, ki, kd, setpoint, duration])) or min(kp, ki, kd) < 0 or duration <= 0:
        raise ValueError('Finite nonnegative gains and positive duration are required.')
    gains = PIDGains(kp, ki, kd)
    t = np.linspace(0, duration, 5000)
    _, open_output = simulator.open_loop(plant, t)
    _, closed_output = simulator.close_loop(plant, gains, t)
    effort, _ = simulator.control_effort(plant, gains, t)
    _, den = simulator.closed_loop_coefficients(plant, gains)
    return t, setpoint * open_output, setpoint * closed_output, setpoint * effort, np.roots(den)


def main():
    fig, (output_ax, effort_ax) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.43, hspace=0.3)
    fig.suptitle("Second-order plant + PID feedback", fontsize=16)
    open_line, = output_ax.plot([], [], label="Without controller: input = setpoint")
    pid_line, = output_ax.plot([], [], label="With PID feedback")
    target_line, = output_ax.plot([], [], "k--", label="Setpoint")
    effort_line, = effort_ax.plot([], [], color="tab:purple")
    output_ax.set_ylabel("Plant output y(t)")
    effort_ax.set_ylabel("Regular PID input u(t)")
    effort_ax.set_xlabel("Time (s)")
    output_ax.legend(loc="upper right", fontsize=8)
    for ax in (output_ax, effort_ax):
        ax.grid(alpha=0.3)
    status = fig.text(0.1, 0.915, "", fontsize=10)
    fig.text(0.1, 0.015,
             "G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)\n"
             "Ideal PID: C(s) = Kp + Ki/s + Kd*s. "
             "Input impulse weight = setpoint*Kd at t=0 (not plotted); delay uses Pade order 2.", fontsize=9)
    settings = [
        ("zeta", "Damping zeta", 0.05, 2.0, 0.2),
        ("wn", "Frequency wn (rad/s)", 0.2, 10.0, 1.0),
        ("gain", "Plant gain K", 0.1, 3.0, 1.0),
        ("delay", "Dead time (s)", 0.0, 1.0, 0.2),
        ("kp", "Kp", 0.0, 20.0, 1.0),
        ("ki", "Ki", 0.0, 20.0, 0.5),
        ("kd", "Kd", 0.0, 10.0, 0.5),
        ("setpoint", "Setpoint", 0.1, 5.0, 1.0),
        ("duration", "Duration (s)", 5.0, 100.0, 30.0),
    ]
    sliders = {}
    for index, (key, label, lower, upper, default) in enumerate(settings):
        slider_ax = fig.add_axes([0.28, 0.355 - index * 0.037, 0.57, 0.021])
        sliders[key] = Slider(slider_ax, label, lower, upper, valinit=default)

    def update(_=None):
        values = {key: slider.val for key, slider in sliders.items()}
        t, y_open, y_pid, effort, poles = simulate(**values)
        stable = np.all(poles.real < 0)
        target_line.set_data(t, np.full_like(t, values["setpoint"]))
        open_line.set_data(t, y_open)
        # Hide divergent traces: huge values otherwise destroy the plot scale.
        if stable and np.all(np.isfinite(y_pid)) and np.all(np.isfinite(effort)):
            pid_line.set_data(t, y_pid)
            effort_line.set_data(t, effort)
            overshoot = max(0.0, (np.max(y_pid) / values["setpoint"] - 1) * 100)
            status.set_text(f"Pade model stable | Overshoot: {overshoot:.1f}% | "
                            f"End error: {values['setpoint'] - y_pid[-1]:.4g} | Input impulse weight: {values['setpoint'] * values['kd']:.3g}")
            status.set_color("darkgreen")
        else:
            pid_line.set_data([], [])
            effort_line.set_data([], [])
            status.set_text("Unstable/marginal or nonfinite response: adjust PID gains. "
                            "PID traces hidden.")
            status.set_color("red")
        for ax in (output_ax, effort_ax):
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    reset_ax = fig.add_axes([0.86, 0.365, 0.09, 0.035])
    reset = Button(reset_ax, "Reset")

    def reset_sliders(_):
        for slider in sliders.values():
            slider.reset()

    reset.on_clicked(reset_sliders)
    update()
    plt.show()


if __name__ == "__main__":
    main()
