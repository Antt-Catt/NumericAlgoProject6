from part1 import step_rk4, meth_epsilon
from math import *
import matplotlib.pyplot as plt
import numpy as np

g, m1, m2, l1, l2 = 9.8, 1, 1, 1, 1

def w(Y, t):
	theta1, dtheta1, theta2, dtheta2 = Y
	print(Y)
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

	plt.title("Problème du double pendule")
	plt.plot(0, 0, color="black", marker="o", label="Point d'attache")
	plt.plot(X1, Y1, color="blue", label="Masse 1")
	plt.plot(X2, Y2, color="red", label="Masse 2")
	plt.legend()
	plt.grid()
	plt.xlabel("theta1")
	plt.ylabel("theta2")
	plt.show()

if __name__ == '__main__':
	double_pendulum(np.array([pi/4, 0]), 0, 10)