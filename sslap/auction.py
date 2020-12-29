import numpy as np
from sslap.auction_ import _from_matrix


def auction_solve(mat:np.ndarray, problem:str = 'min', eps_start:float = 0,
				max_iter:int = 1000000, fast:bool = False) -> dict:
	"""Solve an Auction Algorithm problem, finding an optimal assignment i -> j, such that:
	- each i is assigned to a j
	- the total value of mat[i, j] for all matched i and j is minimized/maximized
	
	:param mat: An (N x M) ndarray, in which A_ij gives the cost of assigning i to j. All valid A_ij must be >= 0.
	Use A_ij = -1 to indicate that i cannot be assigned to j
	:param problem: Whether the solution should minimize (min) or maximize (max) the objective function (default=min)
	:param eps_start: Modify the initial epsilon value used in the algorithm 
	(if not provided, will use half the maximum value in the matrix by default)
	:param max_iter: Maximum number of bidding/assignment stages allowed in the algorithm (default = 1 000 000)
	:param fast: Initialize with the smallest epsilon possible for fast performance. 
	WARNING: A SUB-OPTIMAL (BUT STILL GOOD) SOLUTION LIKELY
	
	:return res: A dictionary of
	   - sol: An array of size N, where the ith entry gives the jth object assigned to i
	   - meta: A dictionary of meta data, including elapsed time, and number of iterations
	"""

	solver = _from_matrix(mat=mat, problem=problem, eps_start=eps_start, max_iter=max_iter, fast=fast)
	sol = solver.solve()
	return dict(sol=sol, meta=solver.meta)