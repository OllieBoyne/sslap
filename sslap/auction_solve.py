import numpy as np
from sslap.auction_ import _from_matrix, _from_sparse
from scipy.sparse.coo import coo_matrix as CooMatrix


def auction_solve(mat: np.ndarray = None, loc: np.ndarray = None, val: np.ndarray = None, coo_mat:CooMatrix=None,
				  problem: str = 'min', eps_start: float = 0.,
				  max_iter: int = 1000000, fast: bool = False, size=None) -> dict:
	"""Solve an Auction Algorithm problem, finding an optimal assignment i -> j, such that:
	- each i is assigned to a j
	- the total value of mat[i, j] for all matched i and j is minimized/maximized

	Receives data in one of three formats:
	1)
	:param mat: An (N x M) ndarray, in which A_ij gives the cost of assigning i to j. All valid A_ij must be >= 0.
	Use A_ij = -1 to indicate that i cannot be assigned to j

	2)
	:param loc: An (N x 2) ndarray, giving the (i, j)th index for the nth valid value in the matrix
	:param val: An (N,) ndarray, giving the nth value in the matrix

	3)
	:param coo_mat: A scipy COO_matrix

	AS WELL AS:

	:param problem: Whether the solution should minimize (min) or maximize (max) the objective function (default=min)
	:param eps_start: Modify the initial epsilon value used in the algorithm 
	(if not provided, will use half the maximum value in the matrix by default)
	:param max_iter: Maximum number of bidding/assignment stages allowed in the algorithm (default = 1 000 000)
	:param fast: Initialize with the smallest epsilon possible for fast performance.
	WARNING: A SUB-OPTIMAL (BUT STILL GOOD) SOLUTION LIKELY
	:param size: Optional tuple for matrix size - used for loc & val input. If not given, infer shape from loc.
	
	:return res: A dictionary of
	   - sol: An array of size N, where the ith entry gives the jth object assigned to i
	   - meta: A dictionary of meta data, including elapsed time, and number of iterations
	"""

	kw = dict(problem=problem, eps_start=eps_start, max_iter=max_iter, fast=fast)

	if mat is not None:
		solver = _from_matrix(mat=mat, **kw)
	elif loc is not None and val is not None:
		solver = _from_sparse(loc=loc, val=val, size=size, **kw)
	elif coo_mat is not None:
		row, col = coo_mat.row, coo_mat.col
		loc, val, size = np.stack([row, col], axis=-1), coo_mat.data, coo_mat.shape
		solver = _from_sparse(loc=loc, val=val, size=size, **kw)
	else:
		raise ValueError("One of the following formats is expected as input to auction solve: mat OR (loc & val) OR coo_mat.")

	sol = solver.solve()
	return dict(sol=sol, meta=solver.meta)
