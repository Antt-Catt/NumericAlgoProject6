import numpy as np
import matplotlib.pyplot as plt

def step_euler(y, t, h, f):
	return y + h * f(y, t)

def step_middle_point(y, t, h, f):
	return y + h * f(y + h * f(y, t) / 2, t + h / 2)

def step_heun(y, t, h, f):
	k1 = f(y, t)
	k2 = f(y + h * k1, t + h)
	return y + h * (k1 + k2) / 2

def step_rk4(y, t, h, f):
	k1 = f(y, t)
	k2 = f(y + h * k1 / 2, t + h / 2)
	k3 = f(y + h * k2 / 2, t + h / 2)
	k4 = f(y + h * k3, t + h)
	return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def meth_n_step(y0, t0, N, h, f, meth):
	res = np.zeros((N, len(y0)))
	res[0] = y0.copy()
	for i in range(1, N):
		res[i] = meth(res[i - 1], t0 + i * h, h, f)
	return res

def meth_epsilon(y0, t0, tf, eps, f, meth):
	N = 2
	h = (tf - t0) / N
	y1 = meth_n_step(y0, t0, N, h, f, meth)
	while True:
		h /= 2
		N *= 2
		y2 = meth_n_step(y0, t0, N, h, f, meth)
		y_tmp = y2[::2]
		if np.all(abs(y1 - y_tmp) < eps):
			break
		y1 = y2
	return y2
