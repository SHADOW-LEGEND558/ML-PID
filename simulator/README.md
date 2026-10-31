# Second-order PID simulator

From the project root, run:

```powershell
python -m pip install numpy scipy matplotlib
python simulator/second_order.py
```

A Matplotlib window opens with sliders for damping ratio, natural frequency,
plant gain, PID gains, setpoint, and simulation duration. Use a desktop Python
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

The model has no transport delay, actuator limits, noise, or automatic tuning.
Derivative action has a 0.05-second first-order filter to keep the response to
a setpoint step finite. This differs from the ideal derivative in src/data_gen.py.
The model represents stable second-order plants such as a series RLC circuit
with capacitor-voltage output; it does not directly represent a ball-on-beam plant.
