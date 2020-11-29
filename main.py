import numpy as np
from sparse_hungarian.solver import SparseHungarianSolver


if __name__ == "__main__":
	np.random.seed(1)
	cost_matrix = np.random.randint(1, 10, (600, 600))
	cost_matrix[np.random.randn(*cost_matrix.shape) > 0] = -1  # make sparse

	solver = SparseHungarianSolver(cost_matrix)
	R, C = solver.solve()