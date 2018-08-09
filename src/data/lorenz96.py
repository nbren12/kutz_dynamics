"""
Lorenz 96 model
From https://en.wikipedia.org/wiki/Lorenz_96_model
"""
import numpy as np
from scipy.integrate import odeint


def lorenz96_rhs(x, t, F):

    N = len(x)

    # compute state derivatives
    d = np.zeros(N)
    # first the 3 edge cases: i=1,2,N
    d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
    d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
    d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]
    # then the general case
    for i in range(2, N - 1):
        d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
    # add the forcing term
    d = d + F

    # return the state derivatives
    return d


def lorenz96():
    # these are our constants
    N = 36  # number of variables
    F = 8  # forcing
    x0 = F * np.ones(N)  # initial state (equilibrium)
    x0[19] += 0.01  # add small perturbation to 20th variable
    t = np.arange(0.0, 40.0, 0.01)

    x = odeint(lorenz96_rhs, x0, t, args=(F, ))

    return t, x
