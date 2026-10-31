"""Interactive second-order plant and PID experiment. Run this file directly."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy import signal


def simulate(zeta=0.2, wn=1.0, gain=1.0, kp=1.0, ki=0.5,
             kd=0.5, setpoint=1.0, duration=30.0):
    """Return time, open-loop output, PID output, input effort, and poles.

    Zero initial conditions, unity negative feedback, no transport delay.
    The derivative uses a fixed first-order filter with time constant 0.05 s.
    """
    if zeta <= 0 or wn <= 0 or gain <= 0 or duration <= 0:
        raise ValueError("Damping, frequency, plant gain, and duration must be positive.")
    if min(kp, ki, kd) < 0:
        raise ValueError("PID gains must be nonnegative.")
    t = np.linspace(0, duration, 3000)
    plant_num = np.array([gain * wn**2])
    plant_den = np.array([1.0, 2 * zeta * wn, wn**2])
    # C(s) = Kp + Ki/s + Kd*s/(0.05*s + 1).
    # Drop the integrator when Ki=0 to avoid an artificial pole at zero.
    if ki == 0:
        controller_num = np.array([0.05 * kp + kd, kp])
        controller_den = np.array([0.05, 1.0])
    else:
        controller_num = np.array([0.05 * kp + kd, kp + 0.05 * ki, ki])
        controller_den = np.array([0.05, 1.0, 0.0])
    loop_num = np.polymul(plant_num, controller_num)
    loop_den = np.polymul(plant_den, controller_den)
    closed_den = np.polyadd(loop_den, loop_num)
    poles = np.roots(closed_den)
    with warnings.catch_warnings():
        # Zero control and leading zero polynomial coefficients are intentional.
        warnings.simplefilter("ignore", signal.BadCoefficients)
        _, open_output = signal.step((plant_num, plant_den), T=t)
        _, closed_output = signal.step((loop_num, closed_den), T=t)
        # U/R = C/(1 + C*G).
        _, effort = signal.step(
            (np.polymul(controller_num, plant_den), closed_den), T=t
        )
    return t, setpoint * open_output, setpoint * closed_output, setpoint * effort, poles


def main():
    fig, (output_ax, effort_ax) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.43, hspace=0.3)
    fig.suptitle("Second-order plant + PID feedback", fontsize=16)
    open_line, = output_ax.plot([], [], label="Without controller: input = setpoint")
    pid_line, = output_ax.plot([], [], label="With PID feedback")
    target_line, = output_ax.plot([], [], "k--", label="Setpoint")
    effort_line, = effort_ax.plot([], [], color="tab:purple")
    output_ax.set_ylabel("Plant output y(t)")
    effort_ax.set_ylabel("PID input u(t)")
    effort_ax.set_xlabel("Time (s)")
    output_ax.legend(loc="upper right", fontsize=8)
    for ax in (output_ax, effort_ax):
        ax.grid(alpha=0.3)
    status = fig.text(0.1, 0.915, "", fontsize=10)
    fig.text(0.1, 0.015,
             "G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)\n"
             "Filtered PID: C(s) = Kp + Ki/s + Kd*s/(0.05*s + 1). "
             "Ideal actuator; no voltage/force limits or delay.", fontsize=9)
    settings = [
        ("zeta", "Damping zeta", 0.05, 2.0, 0.2),
        ("wn", "Frequency wn (rad/s)", 0.2, 10.0, 1.0),
        ("gain", "Plant gain K", 0.1, 3.0, 1.0),
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
            status.set_text(f"Stable | Overshoot: {overshoot:.1f}% | "
                            f"Error at end: {values['setpoint'] - y_pid[-1]:.4g}")
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
