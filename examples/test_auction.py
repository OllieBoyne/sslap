"""Perform Auction Algorithm on small matrix"""
import numpy as np
from sslap import auction_solve
from scipy.sparse import coo_matrix


def dense():
	"""Solve of a dense 5x5 matrix"""
	np.random.seed(1)
	mat = np.random.uniform(0, 10, (5, 5)).astype(np.float64)

	sol = auction_solve(mat, problem='min')

	print("---DENSE SOLVE---")
	print(sol['sol'])
	print(sol['meta'])


def sparse():
	"""Solve a sparse 5x5 matrix using dense format"""
	np.random.seed(1)
	mat = np.random.uniform(0, 10, (5, 5)).astype(np.float64)
	np.random.seed(2)
	mat[np.random.rand(5, 5) > 0.5] = -1  # set roughly half values to invalid

	sol = auction_solve(mat=mat, problem='max')

	print("---SPARSE SOLVE---")
	print(sol['sol'])
	print(sol['meta'])


def sparse_coo_mat():
	"""Solve a sparse 5x5 matrix using scipy's sparse coo_matrix format"""
	np.random.seed(1)
	mat = np.random.uniform(0, 10, (5, 5)).astype(np.float64)
	np.random.seed(2)
	mat[np.random.rand(5, 5) > 0.5] = 0  # set roughly half values to invalid (0 for scipy)

	# assign to sparse matrix
	sparse_mat = coo_matrix(mat)
	sol = auction_solve(coo_mat=sparse_mat, problem='max')

	print("---SPARSE COO_MAT SOLVE---")
	print(sol['sol'])
	print(sol['meta'])


if __name__ == '__main__':
	dense()
	sparse()
	sparse_coo_mat()