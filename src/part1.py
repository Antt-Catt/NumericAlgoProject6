import numpy as np

def step_euler(y, t, h, f):
    return h * f(y, t)

def step_middlepoint(y, t, h, f):
    return h * f(y + (h/2) * f(y, t), t + h/2)

def step_Heun(y, t, h, f):
    p1 = f(y, t)
    p2 = f(y + h * p1, t + h)
    return h * (p1 + p2) / 2

def step_RK4(y, t, h, f):
    p1 = f(y, t)
    y2 = y + (h * p1) / 2
    p2 = f(y2, t + h / 2)
    y3 = y + (h * p2) / 2
    p3 = f(y3, t + h / 2)
    y4 = y + h * p3
    p4 = f(y4, t + h)

def meth_n_step(y0, t0, N, h, f, meth):
    y_array = np.full(N, y0)
    tn = t0
    for i in range(1, N):
        tn += h
        y_array[i] = meth(y_array[i - 1], tn, h, f)
    return y_array