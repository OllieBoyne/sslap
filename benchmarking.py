"""Compare the performance of algorithms across different examples"""
import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt

import pyximport
pyximport.install(setup_args=dict(include_dirs=[np.get_include()]), language_level=3)
from sparse_hungarian.auction import from_matrix as auction_from_matrix
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm

seed = 1

def _improve_feasibility(mask):
	"""Ensure every row and column has at least one element labelled False, for valid connections"""
	count_per_row = (~mask).sum(axis=1)
	R, C = mask.shape
	for r in np.nonzero(count_per_row == 0)[0]:
		mask[r, np.random.randint(C)] = False
	count_per_col = (~mask).sum(axis=0)
	for c in np.nonzero(count_per_col == 0)[0]:
		mask[np.random.randint(R), c] = False
	return mask


class Benchmark:
	def __init__(self, size, density=1.):
		np.random.seed(seed)
		mat = np.random.randint(1, 10, (size, size)).astype(np.float)

		np.random.seed(seed)
		mask = np.random.random(mat.shape) > density  # make matrix sparse based on density
		mask = _improve_feasibility(mask)
		mat[mask] = -1

		self.size = size
		self.density = density
		self.mat = mat

	def __call__(self, func):
		"""Given a function which takes the matrix, and returns the columns C corresponding to the rows
		in ascending order. Validates that this assignment is a) legal and b) complete, and returns the time
		elapsed & objective function"""

		t0 = perf_counter()
		sol, *meta = func(self.mat)
		t1 = perf_counter()

		selected_vals = self.mat[np.arange(self.size), sol] # values selected by assignment

		data = dict(
			func_name=func.__name__,
			elapsed=t1 - t0,
			complete_assignment=(np.unique(sol).size==self.size, (sol>=0).all(), (sol<self.size).all()),
			valid_assignment=(selected_vals >= 0).all(),
			objective_func = selected_vals.sum(),
		)

		if isinstance(meta, dict):
			data.update(meta)

		return data


def auction_solve(mat):
	pyxMat = auction_from_matrix(mat, problem='min', max_iter=10000, eps_start = 50)
	s = pyxMat.solve()
	return s, pyxMat.meta


def scipy_solve(mat):
	scipy_mat = (mat).copy()
	scipy_mat[scipy_mat < 0] = np.inf
	R, C = linear_sum_assignment(scipy_mat)
	return C


def evaluate_by_size(*funcs, sizes=100, densities=1.):
	"""Complete speed tests for all functions, for a range of sizes, and fixed density.
	One of sizes and density must be an ndarray"""

	data = {}
	xdata = []

	xaxis = 'sizes' if isinstance(sizes, np.ndarray) else 'density'
	iterable = sizes if xaxis == 'sizes' else densities

	with tqdm(iterable) as tqdm_it:
		for val in tqdm_it:
			r = f"{xaxis.title()} {val:.3f}"
			tqdm_it.set_description(r)

			size = val if xaxis == 'sizes' else sizes
			density = densities if xaxis == 'sizes' else val
			benchmark = Benchmark(size=size, density=density)

			try:
				for f in funcs:
					res = benchmark(f)
					data[res['func_name']] = data.get(res['func_name'], []) + [res['elapsed']]

			except ValueError: # infeasible cost mat
				continue

			xdata.append(val)

		for k, v in data.items():
			plt.semilogy(xdata, 1000*np.array(v), "-o", label=k)

	plt.xlabel(xaxis)
	plt.ylabel("Elapsed, ms")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	# evaluate_by_size(scipy_solve, auction_solve, sizes= 10 ** np.arange(4), densities=0.1)
	# evaluate_by_size(scipy_solve, auction_solve, sizes=1000, densities=1.3**-np.arange(15))
	evaluate_by_size(scipy_solve, auction_solve, sizes=2500, densities= np.arange(50) / 2500 )
# 0.2 * 1.3 ** -np.arange(15))