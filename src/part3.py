from part1 import step_rk4, meth_epsilon
from math import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

g, m1, m2, l1, l2 = 9.8, 1, 1, 1, 1

total_length = l1 + l2

def w(Y, t):
	theta1, dtheta1, theta2, dtheta2 = Y
	D = (2 * m1 + m2 - m2 * cos(2 * theta1 - 2 * theta2))
	N1 = -g * (2 * m1 + m2) * sin(theta1) - m2 * g * sin(theta1 - 2 * theta2) - 2 * sin(theta1 - theta2) * m2 * (dtheta2 * dtheta2 * l2 + dtheta1 * dtheta1 * l1 * cos(theta1 - theta2))
	N2 = 2 * sin(theta1 - theta2) * (dtheta1 * dtheta1 * l1 * (m1 + m2) + g * (m1 + m2) * cos(theta1) + dtheta2 * dtheta2 * l2 * m2 * cos(theta1 - theta2))
	return np.array([dtheta1, N1 / (D * l1), dtheta2, N2 / (D * l2)])

def double_pendulum(y, t0, tf):
	theta1, theta2 = y
	y0 = np.array([theta1, 0, theta2, 0])
	res = meth_epsilon(y0, t0, tf, 1e-2, w, step_rk4)

	theta1 = res[:, 0]
	theta2 = res[:, 2]

	X1 = l1 * np.sin(theta1)
	X2 = X1 + l2 * np.sin(theta2)

	Y1 = -l1 * np.cos(theta1)
	Y2 = Y1 - l2 * np.cos(theta2)

	t = np.linspace(t0, tf, len(theta1))

	plt.title("Problème du double pendule")
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

	return X1, Y1, X2, Y2, t

def animate_double_pendulum(X1, Y1, X2, Y2, t):
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

	anim = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)
	plt.title("Animation du double pendule")
	plt.plot(X1[0], Y1[0], color="blue", marker="x")
	plt.plot(X2[0], Y2[0], color="red", marker="x")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	X1, Y1, X2, Y2, t = double_pendulum(np.array([pi/2, 0]), 0, 10)
	animate_double_pendulum(X1, Y1, X2, Y2, t)