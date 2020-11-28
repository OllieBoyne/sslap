import numpy as np
from sparse_hungarian.solver import SparseHungarianSolver


if __name__ == "__main__":
	np.random.seed(1)
	cost_matrix = np.random.randint(1, 10, (5, 5))
	# cost_matrix[:, :300] = 0
	# cost_matrix[:, 310:] = 0
	cost_matrix[np.random.randn(*cost_matrix.shape) > 0] = -1  # make sparse

	print(cost_matrix)

	solver = SparseHungarianSolver(cost_matrix)
	R, C = solver.solve()
	print(R, C)