"""
Lorenz 63 model from wikipedia
https://en.wikipedia.org/wiki/Lorenz_system
"""
import numpy as np
from scipy.integrate import odeint


def lorenz63():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def f(state, t):
        x, y, z = state  # unpack the state vector
        return sigma * (
            y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 400.0, 0.01)

    y = odeint(f, state0, t)

    return t, y
