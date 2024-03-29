import numpy as np
import matplotlib.pyplot as plt
import math as math


def step_euler(y, t, h, f):
	"""
	Function to compute the next value with euler method.

	Parameters
	----------
	y : array of floats
		Current value(s).
	t : float
		Current time.
	h: float
		Step.
	f: function
		Function representing the differential equation.
	"""
	return y + h * f(y, t)


def step_middle_point(y, t, h, f):
	"""
	Same function as step_euler but with middle point method.
	"""
	return y + h * f(y + h * f(y, t) / 2, t + h / 2)


def step_heun(y, t, h, f):
	"""
	Same function as step_euler but with Heun method.
	"""
	k1 = f(y, t)
	k2 = f(y + h * k1, t + h)
	return y + h * (k1 + k2) / 2


def step_rk4(y, t, h, f):
	"""
	Same function as step_euler but with Runge-Kutta 4 method.
	"""
	k1 = f(y, t)
	k2 = f(y + h * k1 / 2, t + h / 2)
	k3 = f(y + h * k2 / 2, t + h / 2)
	k4 = f(y + h * k3, t + h)
	return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def meth_n_step(y0, t0, N, h, f, meth):
	"""
	Function to compute N steps after y0

	Parameters
	----------
	y : array of floats
		Initial value(s).
	t : float
		Initial time.
	N : integer
		Number of subdivisions.
	h : float
		Step.
	f : function
		Function representing the differential equation.
	meth : function
		Method used to compute next values (step_euler, step_middle_point, step_heun or step_rk4).
	"""
	res = np.zeros((N, len(y0)))
	res[0] = y0.copy()
	for i in range(1, N):
		res[i] = meth(res[i - 1], t0 + i * h, h, f)
	return res


def meth_epsilon(y0, t0, tf, eps, f, meth):
	"""
	Function to compute the approximate solution of the differential equation
	
	Parameters
	----------
	y0 : array of floats
		Initial value(s).
	t0 : float
		Initial time.
	tf : float
		Final time.
	eps : float
		Tolerance for convergence.
	f : callable
		Function representing the differential equation.
	meth : callable
		Method used to compute next values (step_euler, step_middle_point, step_heun or step_rk4)
	"""
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
	return y2


def meth_epsilon_convergence(y0, t0, tf, eps, f, meth):
	"""
	Function to compute the approximate solution of the differential equation, ploting all the previous solutions to see the convergence
	
	Parameters
	----------
	y0 : array of floats
		Initial value(s).
	t0 : float
		Initial time.
	tf : float
		Final time.
	eps : float
		Tolerance for convergence.
	f : callable
		Function representing the differential equation.
	meth : callable
		Method used to compute next values (step_euler, step_middle_point, step_heun or step_rk4)
	"""

	N = 128
	h = (tf - t0) / N
	y1 = meth_n_step(y0, t0, N, h, f, meth)
	while True:
		plt.plot(np.linspace(t0, tf, len(y1)), y1, label=f"N = {N}")
		h /= 2
		N *= 2
		y2 = meth_n_step(y0, t0, N, h, f, meth)
		y_tmp = y2[::2]
		if np.all(abs(y1 - y_tmp) < eps):
			break
		y1 = y2
	plt.plot(np.linspace(t0, tf, len(y1)), y1, label=f"N = {N}")
	return y2


def tangent_2D(f, t0, tf, y0, yf, N):
	"""
	Function to draw the tangent field of a differential equation.

	Parameters
	----------
	f callable:
		Function representing the differential equation.
	t0 float:
		Initial time.
	tf float:
		Final time.
	y0 float:
		Minimum value of y1 and y2.
	yf float:
		Maximum value of y1 and y2.
	N int:
		Number of points to use in each direction to discretize the phase space.
	"""
	y1 = np.linspace(y0, yf, N)
	y2 = np.linspace(y0, yf, N)
	Y1, Y2 = np.meshgrid(y1, y2)

	U = np.ones_like(Y1)
	V = np.zeros_like(Y2)
	for t in np.arange(t0, tf + 1):
		for i in range(len(y1)):
			for j in range(len(y2)):
				y = np.array([y1[i], y2[j]])
				u, v = f(y, t)
				U[j, i] = u
				V[j, i] = v

	N = np.sqrt(U**2 + V**2)
	U /= N
	V /= N

	plt.quiver(Y1, Y2, U, V)
	plt.title("Champ des tangentes")
	plt.xlabel('y[0]')
	plt.ylabel('y[1]')


if __name__ == '__main__':
	f = lambda y, t: np.array([y / (1 + t * t)])
	res = lambda t: np.exp(np.arctan(t))
	y0 = np.array([1])
	t0 = 0
	tf = 10
	eps = 1e-3

	plt.figure()
	plt.title("Courbes obtenues selon le nombre de subdivisions")
	y = meth_epsilon_convergence(y0, t0, tf, eps, f, step_euler)
	t = np.linspace(t0, tf, len(y))
	plt.plot(t, res(t), label="exp(arctan(t))", c="black", linewidth=2)
	plt.xlabel("t")
	plt.ylabel("y")
	plt.legend()
	plt.grid()
	plt.tight_layout()
	plt.show()

	f = lambda y, t: np.array((-y[1], y[0]))
	y0 = np.array([1, 0])
	t0 = 0
	tf = 2 * math.pi
	eps = 1e-3
	y = meth_epsilon(y0, t0, tf, eps, f, step_euler)
	t = np.linspace(t0, tf, len(y))

	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.scatter(y[:, 0], y[:, 1], c=t, marker='x', cmap="jet")
	plt.title("Tracé de la solution en fonction du temps")
	plt.xlabel("y[0]")
	plt.ylabel("y[1]")
	plt.colorbar(label="temps")
	plt.axis('equal')
	plt.grid()

	plt.subplot(1, 2, 2)
	tangent_2D(f, t0, tf, -5, 5, 20)

	plt.tight_layout()
	plt.show()
