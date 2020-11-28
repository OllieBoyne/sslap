"""Comparison of performance with Scipy's linear sum assignment, for varying levels of sparsity"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sparse_hungarian.solver import SparseHungarianSolver
from time import perf_counter
from matplotlib import pyplot as plt
from tqdm import tqdm

N = 1000

def count_unique_ints(arr):
	"""Count how many unique integers there are in an array"""
	q = np.zeros(arr.max()+1, dtype=np.bool)
	q[arr.ravel()] = 1
	return q.sum()

def _check_solutions(cost_mat, C1, C2):
	"""Check that both solutions
	-> Use each column exactly once
	-> Produce the same objective function
	(Note: they do not have to be identical! There can be multiple solutions.)
	"""

	assert count_unique_ints(C1) == C1.size, "C1 are not all unique"
	assert count_unique_ints(C2) == C2.size, "C2 are not all unique"

	R = np.arange(cost_mat.shape[0]) # Rows in order
	assert cost_mat[R, C1].sum() == cost_mat[R, C2].sum(), "Objective functions do not match"



def solve_scipy(cost_matrix):
	return linear_sum_assignment(cost_matrix)


def solve_sparse(cost_matrix):
	solver = SparseHungarianSolver(cost_matrix)
	return solver.solve()

def main(dense_factor, seed=1, num_rows=10, num_cols=None):
	"""For each sparse_factor, generate a (num_rows, num_cols) size ndarray of random integers,
	and solve the optimal assignment problem, with sparse_factor amount of the entries being valid"""

	if num_cols is None:
		num_cols = num_rows

	np.random.seed(seed)
	raw_costs = np.random.rand(num_rows, num_cols)

	scipy_times = []
	sparse_times = []

	with tqdm(dense_factor) as tqdm_it:
		for sf in tqdm_it:
			r = f"Density = {100 * sf:.2f}%"
			tqdm_it.set_description(r)

			np.random.seed(seed)
			mask = np.random.rand(num_rows, num_cols) > sf  # values to mask out

			t0 = perf_counter()

			scipy_mat = raw_costs.copy()
			scipy_mat[mask] = 1e8
			scipy_R, scipy_C = solve_scipy(scipy_mat)
			tqdm_it.set_description(r + " Scipy solved.")

			t1 = perf_counter()

			sparse_mat = raw_costs.copy()
			sparse_mat[mask] = -1
			sparse_R, sparse_C = solve_sparse(sparse_mat)
			tqdm_it.set_description(r + " Sparse solved.")

			t2 = perf_counter()

			_check_solutions(raw_costs, scipy_C, sparse_C)

			scipy_times.append(t1-t0)
			sparse_times.append(t2-t1)

	plt.plot(dense_factor, scipy_times, label='scipy')
	plt.plot(dense_factor, sparse_times, label='sparse')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	# CURRENTLY - WORKS FOR FLOATS, NOT FOR INTS....
	dense_factors = np.linspace(0.1, 0.3, 10)
	main(dense_factors, num_rows=100)