# Second-order PID simulator

From the project root, run:

```powershell
python -m pip install numpy scipy pandas tqdm matplotlib ipywidgets
python simulator/second_order.py
```

A Matplotlib window opens with sliders for damping ratio, natural frequency,
plant gain, dead time, PID gains, setpoint, and simulation duration. Use a desktop Python
installation with an interactive Matplotlib backend (such as TkAgg or QtAgg).

The upper plot compares the plant driven directly by a constant input equal to
the setpoint against the same plant under unity negative PID feedback. Direct
input approaches K times the setpoint, so it only reaches the target when K=1.
The lower plot shows the input requested by the PID, such as voltage or force.
All simulations start from rest.

Try these experiments:

1. Set Ki and Kd to zero and vary Kp. Observe tracking error and oscillation.
2. Increase Ki gradually to remove persistent tracking error.
3. Increase Kd and observe the change in damping and control effort.
4. Change damping and frequency to see why different plants need different gains.
5. Change the setpoint to see output and input scale together.

The status reports closed-loop pole stability, sampled percentage overshoot,
and error at the end of the simulation (which is not necessarily steady state).
Unstable or marginal PID responses are hidden to keep the plot readable.

Both the notebook and desktop simulator import the ideal-PID plant model from
`src/data_gen.py`. Closed-loop transport delay uses a second-order Pade
approximation; open-loop output uses the analytical true delay. There are no
actuator limits, noise, or automatic tuning. An ideal derivative on a reference
step produces an input impulse of weight `setpoint * Kd` at time zero. The lower
plot shows only the regular part of control effort and explicitly reports the
omitted impulse. The displayed finite-window response is separate from the
optimizer's refined validation. Open the notebook from the project root or the
simulator folder. The plant family does not directly represent ball-on-beam.
