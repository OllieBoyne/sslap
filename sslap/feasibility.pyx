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
cdef class HopcroftKarpSolver:

	cdef int* all_v
	cdef int* start_stops
	cdef int* Pair_U
	cdef int* Pair_V
	cdef double* Dist
	cdef double dist_nil
	cdef int dfs
	cdef int matching
	cdef size_t N, M

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

		self.solve()


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

		return self.matching

cdef int hopcroft_solve(long[:, :] loc, size_t N, size_t M):
	solver = HopcroftKarpSolver(loc, N, M)
	return solver.solve()