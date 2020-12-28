cimport cython
import numpy as np
cimport numpy as np
from time import perf_counter
from cython.parallel import prange
from libc.stdlib cimport malloc, free

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int* cumulative_idxs(long[:] arr, size_t N):
	"""Given an ordered set of integers 0-N, returns an array of size N+1, where each element gives the index of
	the stop of the number / start of the next
	eg [0, 0, 0, 1, 1, 1, 1] -> [0, 3, 7] 
	"""
	cdef int* out = <int*> malloc((N+1)*cython.sizeof(int))
	cdef size_t arr_size, i, j
	cdef int value
	cdef int diff
	arr_size = arr.size
	value = -1

	for i in range(arr_size):
		if arr[i] > value:
			diff = arr[i] - value
			for j in range(diff): # for each incremental value between (usually just 1)
				value += 1
				out[value] = i # set start of new value to i

	out[value+1] = i+1 # add on last value's stop
	if value < N: # fill up rest of array if stops abruptly
		for j in range(value+2, N+1):
			out[j] = i + 1

	return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int* to_pointer(long[:] mv, size_t N):
	"""Convert 1D memory view (length N, type long) to pointer int"""
	cdef int* out = <int *> malloc(N*cython.sizeof(int))
	cdef int i
	for i in range(N):
		out[i] = mv[i]
	return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[ndim=1, dtype=np.int_t] pointer_to_ndarray(int* arr, size_t N):
	"""Convert 1D cython pointer (length N, type int) to int ndarray"""
	cdef np.ndarray[ndim=1, dtype=np.int_t] out = np.empty(N, dtype=np.int)
	cdef int i
	for i in range(N):
		out[i] = arr[i]
	return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int* fill_int(size_t N, int v):
	"""Return a 1D (N,) array fill with -1 (int)"""
	cdef int* out = <int *> malloc(N*cython.sizeof(int))
	cdef int i
	for i in range(N):
		out[i] = v
	return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double* fill_double(size_t N, double v):
	"""Return a 1D (N,) array fill with -1 (int)"""
	cdef double* out = <double *> malloc(N*cython.sizeof(double))
	cdef int i
	for i in range(N):
		out[i] = v
	return out

cdef float inf = float('inf')

#### Implementation of the Hopcroft-Karp Algorithm
cdef class HopcroftKarpSolverCython:

	cdef int* all_v
	cdef int* start_stops
	cdef int* Pair_U
	cdef int* Pair_V
	cdef double* Dist
	cdef double dist_nil
	cdef int matching
	cdef size_t N, M
	cdef int solved

	def __init__(self,	long[:, :] loc, size_t N, size_t M):
		"""loc: a K x 2 array of (u, v) matchings.
		N = num rows, M = num cols
		"""
		cdef long[:] all_u = loc[:, 0]
		cdef int* all_v = to_pointer(loc[:, 1], loc.shape[0])
		self.all_v = all_v
		self.start_stops = cumulative_idxs(all_u, N) # idxs of start/stop for every u, v

		self.N = N
		self.M = M

		self.Pair_U = fill_int(N, -1)
		self.Pair_V = fill_int(M, -1)
		self.Dist = fill_double(N, 0)

		self.dist_nil = inf
		self.matching = 0
		self.solved = 0


	cdef void breadth_first_search(self,):
		cdef int u, v, nv, pairu
		cdef double Dist_nil
		cdef int start, stop
		cdef int* Q = <int*> malloc((self.N**2)*sizeof(int)) # Unoptimized queue system
		cdef int Q_front = 0
		cdef int Q_back = 0

		for u in range(self.N):
			if self.Pair_U[u] == -1:
				self.Dist[u] = 0
				Q[Q_back] = u
				Q_back += 1

			else:
				self.Dist[u] = inf

		Dist_nil = inf

		while Q_front < Q_back:
			u = Q[Q_front]
			Q_front += 1

			if self.Dist[u] < Dist_nil:
				start = self.start_stops[u]
				stop = self.start_stops[u+1]
				for nv in range(stop-start):
					v = self.all_v[start+nv]
					pairu = self.Pair_V[v]
					if pairu == -1:
						if Dist_nil == inf:
							Dist_nil = self.Dist[u] + 1
							# (no need to add to queue)

					else:
						if self.Dist[pairu] == inf:
							self.Dist[pairu] = self.Dist[u] + 1
							Q[Q_back] = pairu
							Q_back += 1

		self.dist_nil = Dist_nil

	cdef int depth_first_search(self, int u):
		cdef int v, nv, pairu
		cdef int start, stop
		cdef double dist_pairu

		if u != -1:
			start = self.start_stops[u]
			stop = self.start_stops[u+1]
			for nv in range(stop-start):
				v = self.all_v[start+nv]
				pairu = self.Pair_V[v]
				if pairu == -1:
					dist_pairu = self.dist_nil
				else:
					dist_pairu = self.Dist[pairu]

				if dist_pairu == self.Dist[u] + 1:
					if self.depth_first_search(pairu):

						self.Pair_V[v] = u
						self.Pair_U[u] = v
						# print(u, v, self.Pair_U.base)
						return 1

			self.Dist[u] = inf
			return 0

		return 1

	cdef solve(self):
		while True:
			self.breadth_first_search()
			if self.dist_nil == inf:
				break

			for u in range(self.N):
				if self.Pair_U[u] == -1:
					if self.depth_first_search(u):
						self.matching += 1

		self.solved = 1
		return self.matching

	cdef dict result(self):
		"""Return dictionary of data from matching"""
		if not self.solved:
			self.solve()

		cdef dict out = dict(size = self.matching, left_pairings = pointer_to_ndarray(self.Pair_U, self.N),
							 right_pairings = pointer_to_ndarray(self.Pair_V, self.M))

		return out

cdef int c_hopcroft_solve(long[:, :] loc, size_t N, size_t M):
	solver = HopcroftKarpSolverCython(loc, N, M)
	return solver.solve()

cpdef hopcroft_solve(np.ndarray loc=None,
						 np.ndarray mat=None,
						 dict lookup=None):
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

	n_None = (loc is None) + (mat is None) + (lookup is None)
	assert n_None == 2, "Exactly one of the arguments loc, mat, lookup must be provided."

	cdef long[:, :] proc_loc # object to be sent to solver
	cdef size_t N, M, E, max_entries
	cdef int ctr

	# for matrix form
	cdef np.ndarray[np.int_t, ndim=2] loc_padded
	cdef long[:, :] locmv
	cdef double[:, :] matmv

	if loc is not None:
		N = loc[:, 0].max() + 1
		M = loc[:, 1].max() + 1
		proc_loc = loc

	elif mat is not None:
		N = mat.shape[0]
		M = mat.shape[1]
		max_entries = N * M
		loc_padded = np.empty((max_entries,2), dtype=np.int, order='C')
		locmv = loc_padded
		matmv = mat

		ctr = 0
		for r in range(N):
			for c in range(M):
				v = matmv[r, c]
				if v >= 0: # if valid entry
					locmv[ctr, 0] = r
					locmv[ctr, 1] = c
					ctr = ctr + 1

		proc_loc = loc_padded[:ctr] # crop to only valid values

	elif lookup is not None:
		N = max(lookup) + 1 # number of elements in I
		M = max(map(max, lookup.values())) + 1 # number of elements in J
		E = sum(map(len, lookup.values()))  # number of total edges
		proc_loc = np.empty((E, 2), dtype=np.int)

		ctr = 0
		for i in lookup:
			for j in lookup[i]:
				proc_loc[ctr, 0] = i
				proc_loc[ctr, 1] = j
				ctr += 1

	# create solver
	cdef HopcroftKarpSolverCython solver = HopcroftKarpSolverCython(proc_loc, N, M)
	return solver.result()