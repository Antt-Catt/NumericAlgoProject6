import numpy as np
import matplotlib.pyplot as plt
from math import *

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
	N = 128
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
		print(N)
	return y2


def tangent_2D(f, t0, tf, y0, yf, N):
	y1 = np.linspace(y0, yf, N)
	y2 = np.linspace(y0, yf, N)
	Y1, Y2 = np.meshgrid(y1, y2)

	U = np.ones_like(Y1)
	V = np.zeros_like(Y2)
	for t in range(t0, tf + 1):
		for i in range(len(y1)):
			for j in range(len(y2)):
				y = np.array([y1[i], y2[j]])
				pente = f(y, t)
				U[j,i] = pente[0]
				V[j,i] = pente[1]

	N = np.sqrt(U**2 + V**2)
	U /= N
	V /= N

	plt.quiver(Y1, Y2, U, V, color='b')
	plt.xlabel('y1')
	plt.ylabel('y2')
	plt.show()

if __name__ == '__main__':
	f = lambda y, t: np.array([y / (1 + t**2)])
	y0 = np.array([1])
	t0 = 0
	tf = 10
	eps = 1e-3
	y = meth_epsilon(y0, t0, tf, eps, f, step_euler)
	t = np.linspace(t0, tf, len(y))

	plt.plot(t, y)
	plt.show()


	f = lambda y, t: np.array((-y[1], y[0]))
	y0 = np.array([1, 0])
	t0 = 0
	tf = 2*pi
	eps = 1e-3
	y = meth_epsilon(y0, t0, tf, eps, f, step_euler)
	t = np.linspace(t0, tf, len(y))

	plt.scatter(y[:, 0], y[:, 1], c=t, marker='x', cmap="jet")
	plt.colorbar()
	plt.show()

	tangent_2D(f, 0, 10, -2, 2, 20)
