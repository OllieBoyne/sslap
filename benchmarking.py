"""Compare the performance of algorithms across different examples"""
import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

import pyximport
pyximport.install(setup_args=dict(include_dirs=[np.get_include()]), language_level=3)
from sslap.auction import from_matrix as auction_from_matrix
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm

seed = 1
problem = 'max'

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
	def __init__(self, size, density=1., mode='float'):
		if mode == 'int':
			np.random.seed(seed)
			mat = np.random.randint(1, 100, (size, size)).astype(np.float)
		elif mode == 'float':
			np.random.seed(seed)
			mat = np.random.uniform(0., 100., size=(size, size)).astype(np.float)


		np.random.seed(seed+1)  # alter seed for selecting randomly
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
		sol, meta = func(self.mat)
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
	pyxMat = auction_from_matrix(mat, problem=problem)
	s = pyxMat.solve()
	return s, pyxMat.meta


def scipy_solve(mat):
	scipy_mat = (1/mat).copy() if problem == 'max' else mat.copy()
	scipy_mat[scipy_mat < 0] = np.inf
	R, C = linear_sum_assignment(scipy_mat)
	return C, None


def evaluate(*funcs, sizes=100, densities=1., mode='float', name='example'):
	"""Complete speed tests for all functions, for a range of sizes, and fixed density.
	One of sizes and density must be an ndarray"""

	data = {}
	xdata = []

	xaxis = 'sizes' if isinstance(sizes, np.ndarray) else 'density'
	iterable = sizes if xaxis == 'sizes' else densities

	fig, ax = plt.subplots(figsize=(8, 5))

	with tqdm(iterable) as tqdm_it:
		for val in tqdm_it:
			r = f"{xaxis.title()} {val:.3f}"
			tqdm_it.set_description(r)

			size = val if xaxis == 'sizes' else sizes
			density = densities if xaxis == 'sizes' else val
			benchmark = Benchmark(size=size, density=density, mode=mode)

			try:
				for f in funcs:
					res = benchmark(f)
					data[res['func_name']] = data.get(res['func_name'], []) + [res['elapsed']]

			except ValueError as e: # infeasible cost mat
				continue

			xdata.append(val)

		for k, v in data.items():
			ax.plot(xdata, 1000*np.array(v), "-o", label=k, ms=2.)

	# axis formatting
	ax.set_xlabel(xaxis.title())
	ax.set_ylabel("Algorithm runtime, ms")
	title = f'Assignment solve, matrix size = {sizes}' if xaxis == 'density' else f'Assignment solve, matrix density = {densities*100:.1f}%'
	ax.set_title(title)

	ax.set_xlim(left=0, right=1. if xaxis == 'density' else sizes.max())
	ax.set_ylim(bottom=0)
	ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.))

	plt.legend()
	plt.tight_layout()
	plt.savefig(f'figs/{name}.png', dpi=300)
	# plt.show()


if __name__ == '__main__':
	# evaluate_by_size(scipy_solve, auction_solve, sizes= (10 * 1.3 ** np.arange(20)).astype(np.int), densities=0.05)
	# evaluate_by_size(scipy_solve, auction_solve, sizes=1000*np.arange(10).astype(np.int), densities=0.01)
	# evaluate_by_size(scipy_solve, auction_solve, sizes=5000, densities=0.01*1.2**np.arange(20))
	evaluate(scipy_solve, auction_solve, sizes=1000, densities=np.linspace(0.01, 1, 100),
			 name='density_benchmarking')