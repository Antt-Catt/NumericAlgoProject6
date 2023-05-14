from part1 import step_euler, step_rk4, meth_epsilon, meth_n_step
from math import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

g, l, m1, m2, l1, l2 = 9.8, 1, 1, 1, 1, 1

total_length = l1 + l2


def is_max(res, i):
	"""
	Function to check if the value at index i is a maximum

	Parameters
	----------
	res : array of floats
		Array of values.
	i : integer
		Index of the value to check if it is a maximum.

	Returns
	-------
	boolean
		True if the value at index i is a maximum, False otherwise.
	"""
	if (res[i - 1][0] < res[i][0] and res[i + 1][0] < res[i][0]):
		return True
	return False


def frequencies(theta_0):
	"""
	Function to compute the frequency of the pendulum

	Parameters
	----------
	theta_0 : float
		Initial angle.

	Returns
	-------
	f : float
		Frequency of the oscillations.
	"""
	y0 = np.array([theta_0, 0])
	t0 = 0
	N = 10000
	h = 10 / N
	f = lambda x, t: np.array([x[1], -(g / l) * np.sin(x[0])])
	res = meth_n_step(y0, t0, N, h, f, step_euler)
	i_1 = 1
	while (not (is_max(res, i_1))):
		i_1 += 1
	i_2 = i_1 + 1
	while (not (is_max(res, i_2))):
		i_2 += 1
	T = (i_2 - i_1) * h
	f = 1 / T
	return f


def frequencies_theta_variable_graph():
	"""
	Function to plot the frequencies of the pendulum with theta_0 varying
	"""
	n = 300
	x = np.array([(-np.pi / 2) + np.pi * i / n for i in range(n)])
	x = np.delete(x, np.where(x == 0))
	y = np.array([frequencies(x[i]) for i in range(n - 1)])
	y_const = np.array([
	    (1 / (2 * np.pi)) * math.sqrt(g / l) for _ in range(n - 1)
	])
	plt.plot(x, y)
	plt.plot(x, y_const)
	plt.show()


def w(Y, t):
	theta1, dtheta1, theta2, dtheta2 = Y
	D = (2 * m1 + m2 - m2 * cos(2 * theta1 - 2 * theta2))
	N1 = -g * (2 * m1 + m2) * sin(theta1) - m2 * g * sin(
	    theta1 - 2 * theta2) - 2 * sin(theta1 - theta2) * m2 * (
	        dtheta2 * dtheta2 * l2 +
	        dtheta1 * dtheta1 * l1 * cos(theta1 - theta2))
	N2 = 2 * sin(theta1 - theta2) * (dtheta1 * dtheta1 * l1 * (m1 + m2) + g *
	                                 (m1 + m2) * cos(theta1) + dtheta2 *
	                                 dtheta2 * l2 * m2 * cos(theta1 - theta2))
	return np.array([dtheta1, N1 / (D * l1), dtheta2, N2 / (D * l2)])


def double_pendulum(y, vy, t0, tf):
	"""
	Numerical simulation of the double pendulum

	Parameters
	----------
	y : array
		array of initial conditions
	vy : array
		array of initial conditions for the derivatives
	t0 : float
		initial time
	tf : float
		final time
	
	Returns
	-------
	theta1 : array
		array of values of the first angle
	theta2 : array
		array of values of the second angle
	t : array
		array of time values
	"""
	theta1, theta2 = y
	dtheta1, dtheta2 = vy
	y0 = np.array([theta1, dtheta1, theta2, dtheta2])
	res = meth_epsilon(y0, t0, tf, 1e-2, w, step_rk4)

	theta1 = res[:, 0]
	theta2 = res[:, 2]

	t = np.linspace(t0, tf, len(theta1))

	X1 = l1 * np.sin(theta1)
	X2 = X1 + l2 * np.sin(theta2)

	Y1 = -l1 * np.cos(theta1)
	Y2 = Y1 - l2 * np.cos(theta2)

	plt.title("Probleme du double pendule")
	plt.plot(X1[0], Y1[0], color="blue", marker="x")
	plt.plot(X2[0], Y2[0], color="red", marker="x")
	plt.plot(0, 0, color="black", marker="o", label="Point d'attache")
	plt.plot(X1, Y1, color="blue", label="Masse 1")
	plt.plot(X2, Y2, color="red", label="Masse 2")
	plt.legend()
	plt.grid()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

	return theta1, theta2, t


def animate_double_pendulum(theta1, theta2, t):
	"""
	Animation of the double pendulum, using two initial angles

	Parameters
	----------
	theta1 : float
		first initial angle
	theta2 : float
		second initial angle
	t : array
		array of time values, to be used as the x-axis
	"""
	X1 = l1 * np.sin(theta1)
	X2 = X1 + l2 * np.sin(theta2)

	Y1 = -l1 * np.cos(theta1)
	Y2 = Y1 - l2 * np.cos(theta2)

	fig, ax = plt.subplots()
	ax.set_xlim(-(total_length), (total_length))
	ax.set_ylim(-(total_length), (total_length))
	plt.plot(0, 0, marker="o", color="black", label="Point d'attache")
	line1, = ax.plot([], [], color="black")
	line2, = ax.plot([], [], color="black")
	m1, = ax.plot([], [], color="blue", marker="o", label="Masse 1")
	m2, = ax.plot([], [], color="red", marker="o", label="Masse 2")
	ax.legend()
	ax.grid()
	ax.set_xlabel("X")
	ax.set_ylabel("Y")

	def animate(i):
		m1.set_data([X1[i]], [Y1[i]])
		m2.set_data([X2[i]], [Y2[i]])
		line1.set_data([0, X1[i]], [0, Y1[i]])
		line2.set_data([X1[i], X2[i]], [Y1[i], Y2[i]])
		return line1, line2, m1, m2

	anim = FuncAnimation(fig, animate, frames=len(t), interval=1, blit=True)
	plt.title("Animation du double pendule")
	plt.plot(X1[0], Y1[0], color="blue", marker="x")
	plt.plot(X2[0], Y2[0], color="red", marker="x")
	plt.legend()
	plt.show()


def find_first_loop(theta1, theta2, t):
	"""
	Function to find the time at which the second mass completes its first loop

	Parameters
	----------
	theta1 : array
		array of the first mass' angles
	theta2 : array
		array of the second mass' angles
	t : array
		array of time values
	
	Returns
	-------
	ti : float
		time at which the second mass completes its first loop
	"""
	for i, ti in enumerate(t):
		if abs(theta2[i]) > pi:
			return ti


if __name__ == '__main__':
	frequencies_theta_variable_graph()

	angles = np.array([pi / 2, 0])
	angular_speed = np.array([4, 0])
	theta1, theta2, t = double_pendulum(angles, angular_speed, 0, 10)
	animate_double_pendulum(theta1, theta2, t)

	first_loop_time = find_first_loop(theta1, theta2, t)
	if first_loop_time is None:
		print("Pas de boucle")
	else:
		print("Temps pour le premier tour de la masse 2 : ", first_loop_time)
