from sslap.feasibility_ import hopcroft_solve as cython_hopcroft_solve
import numpy as np


def hopcroft_solve(loc:np.ndarray = None, mat:np.ndarray = None, lookup:dict = None) -> dict:
	"""Cython/Python callable function to return a maximum matching of a bipartite graph, with sets I and J.
	Can receive one of 3 data types as input:

	loc: A (E x 2) integer ndarray, where each row gives the (i, j)th value of a single edge
	mat: 2D ndarray of floats, where A_ij >= 0 indicates a valid connection between i and j (negative for invalid values)
	lookup: A dictionary where key i gives a list/ndarray of valid connections to J

	returns a dictionary of:
		size: the number of edges included in the maximum matching
		left_pairings: An array of size |I|, where the ith entry gives the vertex in J which connects to I
		right_pairings: An array of size |J|, where the jth entry gives the vertex in I which connects to J

	for left_ and right_pairings, a value of -1 indicates that that vertex is not connected.
	"""
	return cython_hopcroft_solve(loc=loc, mat=mat, lookup=lookup)