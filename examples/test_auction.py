"""Perform Auction Algorithm on small matrix"""
import numpy as np
from sslap.auction import auction_solve


def dense():
	np.random.seed(1)
	mat = np.random.uniform(0, 10, (5, 5)).astype(np.float)

	sol = auction_solve(mat, problem='min')

	print("---DENSE SOLVE---")
	print(sol['sol'])
	print(sol['meta'])


def sparse():
	np.random.seed(1)
	mat = np.random.uniform(0, 10, (5, 5)).astype(np.float)
	np.random.seed(2)
	mat[np.random.rand(5, 5) > 0.5] = -1  # set roughly half values to invalid

	sol = auction_solve(mat, problem='max')

	print("---SPARSE SOLVE---")
	print(sol['sol'])
	print(sol['meta'])

if __name__ == '__main__':
	dense()
	sparse()