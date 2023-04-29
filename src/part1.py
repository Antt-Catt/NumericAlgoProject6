import numpy as np
import matplotlib.pyplot as plt

def step_euler(y, t, h, f):
    return y + h * f(y, t)

def step_middlepoint(y, t, h, f):
    return y + h * f(y + (h/2) * f(y, t), t + h/2)

def step_Heun(y, t, h, f):
    p1 = f(y, t)
    p2 = f(y + h * p1, t + h)
    return y + h * (p1 + p2) / 2

def step_RK4(y, t, h, f):
    p1 = f(y, t)
    y2 = y + (h * p1) / 2
    p2 = f(y2, t + h / 2)
    y3 = y + (h * p2) / 2
    p3 = f(y3, t + h / 2)
    y4 = y + h * p3
    p4 = f(y4, t + h)
    return y + h * (p1 + 2 * p2 + 2 * p3 + p4) / 6

def meth_n_step(y0, t0, N, h, f, meth):
    y_array = np.full(N, y0)
    tn = t0
    for i in range(1, N):
        tn += h
        y_array[i] = meth(y_array[i - 1], tn, h, f)
    return y_array

def meth_epsilon(y0, t0, tf, eps, f, meth):
    N = 2
    h = (tf - t0) / N
    y1 = meth_n_step(y0, t0, N, h, f, meth)
    while True:
        h /= 2
        N *= 2
        print(N)
        y2 = meth_n_step(y0, t0, N, h, f, meth)
        y_tmp = y2[::2]
        if np.all(abs(y1 - y_tmp) < eps):
            break
        y1 = y2
    return y2