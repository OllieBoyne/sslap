cimport cython
import numpy as np
cimport numpy as np
from time import perf_counter
from cython.parallel import prange
from libc.stdlib cimport malloc, free


np.import_array()

# tolerance to deal with floating point precision for eCE, due to eps being stored as float 32
cdef double tol = 1e-7

# ctypedef np.int_t DTYPE_t
cdef DTYPE = np.float
ctypedef np.float_t DTYPE_t
cdef float inf = float('infinity')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.int_t, ndim=1] cumulative_idxs(np.ndarray[np.int_t, ndim=1] arr, size_t N):
	"""Given an ordered set of integers 0-N, returns an array of size N+1, where each element gives the index of
	the stop of the number / start of the next
	eg [0, 0, 0, 1, 1, 1, 1] -> [0, 3, 7] 
	"""
	cdef np.ndarray[np.int_t, ndim=1] out = np.empty(N+1, dtype=np.int)
	cdef size_t arr_size, i
	cdef int value
	arr_size = arr.size
	value = -1
	for i in range(arr_size):
		if arr[i] > value:
			value += 1
			out[value] = i # set start of new value to i

	out[value+1] = i # add on last value's stop
	return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int* fill_neg1_int(size_t N):
	"""Return a 1D (N,) array fill with -1 (int)"""
	# cdef np.ndarray[ndim=1, dtype=np.int_t] out = np.empty(N, dtype=np.int)
	# cdef array.array[double] out = array.array(-1, N)
	cdef int* out = <int *> malloc(len(N)*cython.sizeof(int))
	# cdef int i = -1
	# cdef int* ptr = <int*> out.data
	# cdef int[::1] ptr = out
	for i in range(N):
		out[i] = -1

	return out

cdef class AuctionSolver:
	#Solver for auction problem
	# Which finds an assignment of N people -> N objects, by having people 'bid' for objects
	cdef size_t num_rows, num_cols
	cdef set unassigned_people

	# common objects to be copies
	cdef set all_people
	cdef np.ndarray empty_N_ints
	cdef np.ndarray empty_N_floats

	cdef np.ndarray p
	cdef dict lookup_by_person
	cdef dict lookup_by_object

	cdef np.ndarray i_starts_stops, j_counts, flat_j
	cdef double[::1] val  # memory view of all values
	cdef np.ndarray person_to_object # index i gives the object, j, owned by person i
	cdef np.ndarray object_to_person # index j gives the person, i, who owns object j

	cdef float eps
	cdef float target_eps
	cdef float delta, theta
	cdef int k

	#meta
	cdef str problem
	cdef int nits, nreductions
	cdef int max_iter
	cdef int optimal_soln_found
	cdef public dict meta
	cdef float extreme
	cdef dict debug_timer


	def __init__(self, np.ndarray[np.int_t, ndim=2] loc, np.ndarray[DTYPE_t, ndim=1] val,
				 size_t  num_rows=0, size_t num_cols=0, str problem='min',
				 size_t max_iter=1000000, float eps_start=0):

		self.debug_timer = {'bidding':0, 'assigning':0, 'setup':0, 'forbids':0}
		t0 = perf_counter()

		cdef size_t N = loc[:, 0].max() + 1
		cdef size_t M = loc[:, 1].max() + 1
		self.num_rows = N
		self.num_cols = M

		self.problem = problem
		self.nits = 0
		self.nreductions = 0
		self.max_iter = max_iter

		self.all_people = {*range(N)} # set of all workers, to be copied to unassigned people at every reset
		self.unassigned_people = self.all_people.copy()

		# set up price vector j
		self.p = np.zeros(M, dtype=DTYPE)

		# indices of starts/stops for every i in sorted list of rows (eg [0,0,1,1,1,2] -> [0, 2, 5, 6])
		self.i_starts_stops = cumulative_idxs(loc[:, 0], N)

		# number of indices per val in list of sorted rows (eg [0,0,1,1,1,2] -> [2, 3, 1])
		cdef np.ndarray[np.int_t, ndim=1] jcounts = np.diff(self.i_starts_stops)
		self.j_counts = jcounts

		# indices of all j values, to be indexed by self.i_starts_stops
		self.flat_j = loc[:, 1].copy(order='C')

		self.person_to_object = np.full(N, dtype=np.int, fill_value=-1)
		self.object_to_person = np.full(M, dtype=np.int, fill_value=-1)

		# to save system memory, make v ndarray an empty, and only populate used elements
		if problem == 'min':
			val *= -1 # auction algorithm always maximises, so flip value signs for min problem

		cdef np.ndarray[DTYPE_t, ndim=1] v = (val * (self.num_rows+1))
		self.val = v

		# Calculate optimum initial eps and target eps
		cdef float C  # = max |aij| for all i, j in A(i)
		cdef int approx_ints # Esimated
		C = np.abs(val).max()

		# choose eps values
		self.eps = C/2
		self.target_eps = 1
		self.k = 0
		self.theta = 0.5 # reduction factor

		# override if given
		if eps_start > 0:
			self.eps = eps_start

		# create common objects
		self.empty_N_ints = np.full(N, dtype=np.int, fill_value=-1)
		self.empty_N_floats = np.full(N, dtype=DTYPE, fill_value=-1)

		self.optimal_soln_found = False
		self.meta = {}
		self.debug_timer['setup'] += perf_counter() - t0

	cpdef np.ndarray solve(self):
		"""Run iterative steps of Auction assignment algorithm"""
		cdef np.ndarray[DTYPE_t, ndim=1] best_bids # highest bid for each object
		cdef np.ndarray[np.int_t, ndim=1] best_bidders # person who made each high bid
		cdef set all_people = self.all_people

		while True:
			t0 = perf_counter()
			best_bids, best_bidders = self.bidding_phase()
			t1 = perf_counter()
			self.assignment_phase(best_bids, best_bidders)
			t2 = perf_counter()
			self.debug_timer['bidding'] += t1-t0
			self.debug_timer['assigning'] += t2-t1

			self.nits += 1

			if self.terminate():
				break

			# full assignment made, but not all people happy, so restart with same prices, but lower eps
			elif len(self.unassigned_people) == 0:
				if self.eps < self.target_eps:  # terminate, shown to be optimal for eps < 1/n
					break

				self.eps = self.eps*self.theta
				self.k += 1

				self.person_to_object[:] = -1
				self.object_to_person[:] = -1
				self.unassigned_people = all_people.copy()
				self.nreductions += 1

		# Finished, validate soln
		self.meta['eCE'] = self.eCE_satisfied(eps=self.target_eps)
		self.meta['its'] = self.nits
		self.meta['nreductions'] = self.nreductions
		self.meta['soln_found'] = self.is_optimal()
		self.meta['n_assigned'] = self.num_rows - len(self.unassigned_people)
		self.meta['obj'] = self.get_obj()
		self.meta['final_eps'] = self.eps
		self.meta['debug_timer'] = {k:f"{1000 * v:.2f}ms" for k, v in self.debug_timer.items()}

		return self.person_to_object

	cdef int terminate(self):
		return (self.nits >= self.max_iter) or ((len(self.unassigned_people) == 0) and self.is_optimal())

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef tuple bidding_phase(self):
		cdef int i, j, jbest, nbidder, idx
		cdef double costbest, vbest, bbest, wi, vi, cost

		# store bids made by each person, stored in sequential order in which bids are made
		cdef size_t N = self.num_cols
		cdef size_t num_bidders = len(self.unassigned_people)  # number of bids to be made

		cdef np.ndarray[np.int_t, ndim=1] bidders = self.empty_N_ints[:num_bidders].copy()
		cdef np.ndarray[np.int_t, ndim=1] objects_bidded = bidders.copy()
		cdef np.ndarray[DTYPE_t, ndim=1] bids = self.empty_N_floats[:num_bidders].copy()

		cdef double* bids_ptr = <double*> bids.data
		cdef int* bidders_ptr = <int*> bidders.data
		cdef int* objects_ptr = <int*> objects_bidded.data
		cdef double* p_ptr = <double*> self.p.data

		# each person now makes a bid:
		cdef np.ndarray[np.int_t, ndim=1] unassigned_people = np.fromiter(self.unassigned_people, dtype=np.int)
		cdef int* unassigned_people_mv = <int*> unassigned_people.data
		cdef size_t num_objects, glob_idx, start

		cdef long* j_counts = <long*> self.j_counts.data
		cdef long* i_starts_stops = <long*> self.i_starts_stops.data
		cdef long* flat_j = <long*> self.flat_j.data
		cdef double[::1] val = self.val

		for nbidder in range(num_bidders):
			i = unassigned_people_mv[nbidder]  # person, i who makes the bid
			num_objects = j_counts[i]  # the number of objects this person is able to bid on
			start = i_starts_stops[i]  # in flattened index format, the starting index of this person's objects/values
			# Start with vbest, costbest etc defined as first available object
			glob_idx = start
			jbest = flat_j[glob_idx]
			cost = val[glob_idx]
			vbest = - cost - p_ptr[jbest]
			wi = - inf  # set second best vi, 'wi', to be -inf by default (if only one object)
			# Go through each object, storing its index & cost if vi is largest, and value if vi is second largest
			for idx in range(num_objects):
				glob_idx = start + idx
				j = flat_j[glob_idx]
				cost = val[glob_idx]
				vi = cost - p_ptr[j]
				if vi >= vbest:
					jbest = j
					wi = vbest # store current vbest as second best, wi
					vbest = vi
					costbest = cost

				elif vi > wi:
					wi = vi

			bbest = costbest - wi + self.eps  # value of new bid

			# store bid & its value
			bidders_ptr[nbidder] = i
			bids_ptr[nbidder] = bbest
			objects_ptr[nbidder] = jbest

		cdef np.ndarray[DTYPE_t, ndim=1] best_bids = self.empty_N_floats.copy()

		# t0 = perf_counter()
		cdef np.ndarray[np.int_t, ndim=1] best_bidders = self.empty_N_ints.copy()
		# cdef int* best_bidders = fill_neg1_int(N)
		# self.debug_timer['forbids'] += perf_counter()-t0

		## memory views for efficient indexing
		cdef double[::1] best_bids_mv = best_bids
		cdef long[::1] best_bidders_mv = best_bidders
		# cdef int* best_bidders_mv = best_bidders
		cdef double bid_val
		cdef size_t jbid, n

		for n in range(num_bidders):  # for each bid made,
			i = bidders_ptr[n]
			bid_val = bids_ptr[n]
			jbid = objects_ptr[n]
			if bid_val > best_bids_mv[jbid]:
				best_bids_mv[jbid] = bid_val
				best_bidders_mv[jbid] = i

		cdef tuple out = (best_bids, best_bidders)
		return out

	cdef void assignment_phase(self, np.ndarray[DTYPE_t, ndim=1] best_bids,
							   np.ndarray[np.int_t, ndim=1] best_bidders):
		cdef tuple tup
		cdef list bidding_i
		cdef np.ndarray[DTYPE_t, ndim=1] bids
		cdef int i = 0

		cdef int prev_i
		cdef int[:] o2p = self.object_to_person
		cdef int[:] p2o = self.person_to_object

		# pointers to 1D arrays for quick accessing
		cdef double* p = <double*> self.p.data
		cdef double* bidsptr = <double*> best_bids.data
		cdef int* bidderptr = <int*> best_bidders.data

		for j in range(self.num_cols):
			i = bidderptr[j]
			if i != -1:
				p[j] = bidsptr[j]

				# unassign previous i (if any)
				prev_i = o2p[j]
				if prev_i != -1:
					self.unassigned_people.update({prev_i})
					p2o[prev_i] = -1

				# make new assignment
				self.unassigned_people.difference_update({i})
				p2o[i] = j
				o2p[j] = i

	cdef int is_optimal(self):
		"""Checks if current solution is a complete solution that satisfies eps-complementary slackness.
		As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
		 it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure"""
		if len(self.unassigned_people) > 0:
			return False
		return self.eCE_satisfied(eps=self.target_eps)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int eCE_satisfied(self, float eps=0.):
		"""Returns True if eps-complementary slackness condition is satisfied"""
		# e-CE: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
		if len(self.unassigned_people) > 0:
			return False

		cdef size_t i, j, k, l, idx, count, num_objects, start
		cdef double choice_cost, v, LHS
		cdef double[:] p = self.p
		cdef long[:] j_counts = self.j_counts
		cdef long[:] i_starts_stops = self.i_starts_stops
		cdef long[:] person_to_object = self.person_to_object
		cdef long[:] flat_j = self.flat_j
		cdef double[:] val = self.val


		for i in range(self.num_rows):
			num_objects = j_counts[i]  # the number of objects this person is able to bid on
			start = i_starts_stops[i]  # in flattened index format, the starting index of this person's objects/values
			j = person_to_object[i]  # chosen object

			# first, get cost of choice j
			for idx in range(num_objects):
				glob_idx = start + idx
				l = flat_j[glob_idx]
				if l == j:
					choice_cost = val[glob_idx]

			# k are all possible biddable objects.
			# Go through each, asserting that (a_ij - p_j) + tol >= max(a_ik - p_k) - eps for all k
			LHS = choice_cost - p[j] + tol  # left hand side of inequality

			for idx in range(num_objects):
				glob_idx = start + idx
				k = flat_j[glob_idx]
				cost = val[glob_idx]
				v = cost - p[k]
				if LHS < v - eps:
					return False  # The eCE condition is not met.

		return True

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef float get_obj(self):
		"""Returns current objective value of assignments"""
		cdef double obj = 0
		cdef size_t i, j, l, counts, idx, glob_idx, num_objects, start
		cdef long[:] js
		cdef double[:] vs

		cdef long[:] j_counts = self.j_counts
		cdef long[:] i_starts_stops = self.i_starts_stops
		cdef long[:] person_to_object = self.person_to_object
		cdef long[:] flat_j = self.flat_j
		cdef double[:] val = self.val


		for i in range(self.num_rows):
			# due to the way data is stored, need to go do some searching to find the corresponding value
			# to assignment i -> j
			j = person_to_object[i] # chosen j
			if j == -1: # skip any unassigned
				continue

			num_objects = j_counts[i]
			start = i_starts_stops[i]

			for idx in range(num_objects):
				glob_idx = start + idx
				l = flat_j[glob_idx]
				if l == j:
					obj += val[glob_idx]

		if self.problem is 'min':
			obj *= -1

		return obj / (self.num_rows+1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef AuctionSolver from_matrix(np.ndarray mat, str problem='min', float eps_start=0,
								size_t max_iter = 1000000):
	# Return an Auction Solver from a dense matrix (M, N), where invalid values are -1
	cdef size_t N = mat.shape[0]
	cdef size_t M = mat.shape[1]
	cdef size_t max_entries = N * M
	cdef size_t ctr = 0
	cdef double v

	cdef np.ndarray[np.int_t, ndim=2] loc_padded = np.empty((max_entries,2), dtype=np.int, order='C')
	cdef np.ndarray[DTYPE_t, ndim=1] val_padded = np.empty((max_entries,), dtype=DTYPE, order='C')

	cdef long[:, :] loc = loc_padded
	cdef double[:] val = val_padded
	cdef double[:, :] matmv = mat

	for r in range(N):
		for c in range(M):
			v = matmv[r, c]
			if v >= 0: # if valid entry
				loc[ctr, 0] = r
				loc[ctr, 1] = c
				val[ctr] = v
				ctr = ctr + 1

	# Now crop to only processed entries
	cdef np.ndarray[np.int_t, ndim=2] cropped_loc = loc_padded[:ctr]
	cdef np.ndarray[DTYPE_t, ndim=1] cropped_val = val_padded[:ctr]

	if ctr < N:
		raise ValueError("Matrix is infeasible")

	return AuctionSolver(cropped_loc, cropped_val, problem=problem, eps_start=eps_start, max_iter=max_iter)