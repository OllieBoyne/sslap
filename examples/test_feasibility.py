"""Check the feasibility of a bipartite graph by using SSLAP's feasibility module"""
import numpy as np
from sslap import hopcroft_solve


# All 3 methods will use the same input bipartite graph:
# i = 0 connects to j = 0, 1
# i = 1 connects to j = 1, 2
# i = 2 connects to j = 1, 4
# i = 3 connects to j = 2
# i = 4 connects to j = 3
# which has a maximum matching of 5
# eg i:j of 0:0, 1:1, 2:4, 3:2, 4:3


def dict_usage():
	lookup = {0: [0, 1], 1: [1, 2], 2: [1, 4], 3: [2], 4: [3]}
	res = hopcroft_solve(lookup=lookup)
	print(res)


def mat_usage():
	mat = - np.ones((5, 5))  # all invalid, except
	mat[[0, 0, 1, 1, 2, 2, 3, 4], [0, 1, 1, 2, 1, 4, 2, 3]] = 1  # for valid edges
	res = hopcroft_solve(mat=mat)
	print(res)


def loc_usage():
	loc = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 1], [2, 4], [3, 2], [4, 3]])  # (i, j) for each edge
	res = hopcroft_solve(loc=loc)
	print(res)


if __name__ == "__main__":
	dict_usage()
	mat_usage()
	loc_usage()
