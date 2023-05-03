from part1 import meth_n_step, step_rk4, meth_epsilon

import matplotlib.pyplot as plt
import numpy as np


def first_models(y0, t0, tf, N, gamma, k):
	"""
	Function to plot a comparison between the first population model
	and the Verhulst model.

	Parameters
	----------
	y0 : float
		Initial value of the population.
	t0 : float
		Initial time.
	tf : float
		Final time.
	N : int
		Number of steps.
	gamma : float
		Growth rate.
	k : float
		Carrying capacity.
	"""
	h = (tf - t0) / N
	t = [t0 + h * i for i in range(N)]

	# First differential equation
	derivN1 = lambda x, t : gamma * x
	res1 = meth_n_step(y0, t0, N, h, derivN1, step_rk4)
	plt.subplot(1, 2, 1)
	plt.title("Premier modèle de population")
	plt.xlabel("Temps")
	plt.ylabel("Nombre d'individus")
	plt.plot(t, res1, marker='x')
	plt.grid()

	# Second differential equation
	derivN2 = lambda x, t : gamma * x * (1 - x / k)
	res2 = meth_n_step(y0, t0, N, h, derivN2, step_rk4)
	plt.subplot(1, 2, 2)
	plt.title("Modèle de population de Verhulst")
	plt.xlabel("Temps")
	plt.ylabel("Nombre d'individus")
	plt.plot(t, res2, marker='x')
	plt.grid()
	plt.tight_layout()
	plt.show()


def second_model(y0, t0, tf, N, a, b, c, d):
	"""
	Function to plot the Lotka-Volterra model.

	Parameters
	----------
	y0 : float
		Initial value of the population.
	t0 : float
		Initial time.
	tf : float
		Final time.
	N : int
		Number of steps.
	a : float
		Growth rate of the predators.
	b : float
		Death rate of the predators.
	c : float
		Growth rate of the prey.
	d : float
		Death rate of the prey.
	"""

	def derivative(Y, t):
		return np.array([Y[0] * (a - b * Y[1]), Y[1] * (c * Y[0] - d)])
	
	res = meth_epsilon(y0, t0, tf, 1e-7, derivative, step_rk4)
	N = len(res)
	h = (tf - t0) / N
	t = [t0 + h * i for i in range(N)]

	plt.subplot(1, 2, 1)
	plt.title("Modèle de Lotka-Volterra")
	plt.xlabel("Temps")
	plt.ylabel("Nombre d'individus")
	plt.plot(t, res[:, 0], marker='x', label='Prédateurs')
	plt.plot(t, res[:, 1], marker='x', label='Proies')
	plt.grid()
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.title("Tracé de (N(t), P(t))")
	plt.xlabel("Proies")
	plt.ylabel("Prédateurs")
	plt.scatter(res[:, 0], res[:, 1], c=t, marker='x', cmap="jet")
	plt.grid()
	plt.axis("equal")
	plt.colorbar(label="Temps")

	plt.tight_layout()
	plt.show()


def find_period(y0, t0, tf, a, b, c, d, eps=5e-3):
	"""
	Function to find the period of the Lotka-Volterra model.

	Parameters
	----------
	y0 : float
		Initial value of the population.
	t0 : float
		Initial time.
	tf : float
		Final time.
	N : int
		Number of steps.
	a : float
		Growth rate of the predators.
	b : float
		Death rate of the predators.
	c : float
		Growth rate of the prey.
	d : float
		Death rate of the prey.
	eps : float (optional)
		Precision of the period.
	"""
	
	def derivative(Y, t):
		return np.array([Y[0] * (a - b * Y[1]), Y[1] * (c * Y[0] - d)])
	
	res = meth_epsilon(y0, t0, tf, 1e-7, derivative, step_rk4)
	N = len(res)
	h = (tf - t0) / N
	x = res[:, 0]
	y = res[:, 1]

	for i in range(1, N):
		if abs(x[0] - x[i]) < eps and abs(y[0] - y[i]) < eps:
			return h * i

def plot_solutions(y0, t0, tf, a, b, c, d):

	def derivative(Y, t):
		return np.array([Y[0] * (a - b * Y[1]), Y[1] * (c * Y[0] - d)])
	
	delta = 0.2
	num = 5
	i = 0
	initial_values = np.zeros((num * num, 2))
	for x in np.linspace(y0[0] - delta, y0[0] + delta, num):
		for y in np.linspace(y0[1] - delta, y0[1] + delta, num):
			initial_values[i] = np.array([x, y])
			i += 1
	for yi in initial_values:
		res = meth_epsilon(yi, t0, tf, 1e-7, derivative, step_rk4)
		plt.plot(res[:, 0], res[:, 1], color="black")
	plt.xlim(y0[0] - delta, y0[0] + delta)
	plt.ylim(y0[1] - delta, y0[1] + delta)
	plt.grid()
	plt.xlabel("Proies")
	plt.ylabel("Prédateurs")
	plt.show()


if __name__ == '__main__':
	gamma, k = 0.7, 6000

	y0 = np.array([5000.])
	N = 40
	t0, tf = 0, 8

	# first_models(y0, t0, tf, N, gamma, k)

	param = [1.2, 1, 0.8, 1]
	y0 = np.array([0.7, 0.3])
	N = 10
	t0, tf = 0, 8.2

	second_model(y0, t0, tf, N, *param)
	period = find_period(y0, t0, tf * 2, *param)
	print(f"La période est de {period:.2f} unités de temps.")

	plot_solutions(y0, t0, tf, *param)
