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
cdef double* fill_float(size_t N, double v):
	"""Return a 1D (N,) array fill with -1 (floaT)"""
	cdef double* out = <double *> malloc(N*cython.sizeof(double))
	for i in range(N):
		out[i] = v
	return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int* arange(size_t N):
	"""Return a 1D (N,) array with integers 0 -> N"""
	cdef int* out = <int *> malloc(N*cython.sizeof(int))
	cdef int i
	for i in range(N):
		out[i] = i
	return out

cdef void push_all_left(int* data, int* mapper, int num_ints, size_t size):
	"""Given an array of positive integers (size <size>) and -1s, arrange so that all positive integers are at the start of the array.
	Provided with N (number of positive integers) for speed increase.
	eg [-1, 1, 2, 3, -1, -1] -> [3, 1, 2, -1, -1, -1] (order not important).
	Also updates mapper in tandem, a 1d array in which the ith idx gives the position of integer i in the array data.
	All modifications are inplace."""

	if num_ints == 0:
		return

	cdef int ctr, i
	cdef int left_track = 0 # cursor on left hand side of partition
	cdef int right_track = num_ints # cursor on right hand side of partition
	while left_track < num_ints: # keep going until found all N components
		if data[left_track] == -1: # if empty space
			# move through right track until hit a positive integer (or the end of the array)
			while data[right_track] == -1 and right_track < size:
				right_track += 1

			# swap two elements
			i = data[right_track] # integer taken through
			data[left_track] = i
			data[right_track] = -1
			mapper[i] = left_track

		left_track += 1

cdef class AuctionSolver:
	#Solver for auction problem
	# Which finds an assignment of N people -> N objects, by having people 'bid' for objects
	cdef size_t num_rows, num_cols

	# common objects to be copies
	cdef np.ndarray empty_N_ints
	cdef np.ndarray empty_N_floats

	cdef np.ndarray p
	cdef dict lookup_by_person
	cdef dict lookup_by_object

	cdef np.ndarray i_starts_stops, flat_j, j_counts
	cdef np.ndarray val  # memory view of all values
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

	cdef double* best_bids
	cdef int* best_bidders
	cdef int* best_bidded_objects

	# assignment storage
	cdef int num_unassigned
	cdef int* unassigned_people
	cdef int* person_to_assignment_idx

	def __init__(self, np.ndarray[np.int_t, ndim=2] loc, np.ndarray[DTYPE_t, ndim=1] val,
				 size_t  num_rows=0, size_t num_cols=0, str problem='min',
				 size_t max_iter=1000000, float eps_start=0):

		self.debug_timer = {'bidding_and_assigning':0, 'setup':0, 'forbids':0, 'misc':0}
		t0 = perf_counter()

		cdef size_t N = loc[:, 0].max() + 1
		cdef size_t M = loc[:, 1].max() + 1
		self.num_rows = N
		self.num_cols = M

		self.problem = problem
		self.nits = 0
		self.nreductions = 0
		self.max_iter = max_iter

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
			val = -val # auction algorithm always maximises, so flip value signs for min problem

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

		# store of best bids & bidders
		self.best_bids = fill_float(M, -1)
		self.best_bidders = fill_int(M, -1)
		self.best_bidded_objects = fill_int(M, -1)

		self.num_unassigned = N
		self.unassigned_people = arange(N)
		self.person_to_assignment_idx = arange(N)

		self.optimal_soln_found = False
		self.meta = {}
		self.debug_timer['setup'] += perf_counter() - t0

	cpdef np.ndarray solve(self):
		"""Run iterative steps of Auction assignment algorithm"""

		while True:
			self.bid_and_assign()
			self.nits += 1

			if self.terminate():
				break

			# full assignment made, but not all people happy, so restart with same prices, but lower eps
			elif self.num_unassigned == 0:
				if self.eps < self.target_eps:  # terminate, shown to be optimal for eps < 1/n
					break

				self.eps = self.eps*self.theta
				self.k += 1

				# reset all trackers of people and objects
				self.person_to_object[:] = -1
				self.object_to_person[:] = -1
				self.num_unassigned = self.num_rows
				self.unassigned_people = arange(self.num_rows)
				self.person_to_assignment_idx = arange(self.num_rows)

				self.nreductions += 1

		# Finished, validate soln
		self.meta['eCE'] = self.eCE_satisfied(eps=self.target_eps)
		self.meta['its'] = self.nits
		self.meta['nreductions'] = self.nreductions
		self.meta['soln_found'] = self.is_optimal()
		self.meta['n_assigned'] = self.num_rows - self.num_unassigned
		self.meta['obj'] = self.get_obj()
		self.meta['final_eps'] = self.eps
		self.meta['debug_timer'] = {k:f"{1000 * v:.2f}ms" for k, v in self.debug_timer.items()}

		return self.person_to_object

	cdef int terminate(self):
		return (self.nits >= self.max_iter) or ((self.num_unassigned == 0) and self.is_optimal())

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void bid_and_assign(self):
		cdef int i, j, jbest, nbidder, idx
		cdef double costbest, vbest, bbest, wi, vi, cost

		# store bids made by each person, stored in sequential order in which bids are made
		cdef size_t N = self.num_cols
		cdef size_t num_bidders = self.num_unassigned # len(self.unassigned_people)  # number of bids to be made
		cdef int* unassigned_people = self.unassigned_people
		cdef int* person_to_assignment_idx = self.person_to_assignment_idx

		cdef int* bidders = fill_int(num_bidders, -1)
		cdef int* objects_bidded = fill_int(num_bidders, -1)
		cdef double* bids = fill_float(num_bidders, -1)
		cdef size_t num_objects, glob_idx, start

		cdef double* p = <double*> self.p.data
		cdef long* j_counts = <long*> self.j_counts.data
		cdef long* i_starts_stops = <long*> self.i_starts_stops.data
		cdef long* flat_j = <long*> self.flat_j.data
		cdef double* val = <double*> self.val.data

		cdef int* person_to_object = <int*> self.person_to_object.data
		cdef int* object_to_person = <int*> self.object_to_person.data

		## BIDDING PHASE
		# each person now makes a bid:
		for nbidder in range(num_bidders): # for each person
			i = unassigned_people[nbidder]
			num_objects = j_counts[i]  # the number of objects this person is able to bid on
			start = i_starts_stops[i]  # in flattened index format, the starting index of this person's objects/values
			vbest = - inf
			wi = - inf  # set second best vi, 'wi', to be -inf by default (if only one object)
			# Go through each object, storing its index & cost if vi is largest, and value if vi is second largest
			for idx in range(num_objects):
				glob_idx = start + idx
				j = flat_j[glob_idx]
				cost = val[glob_idx]
				vi = cost - p[j]
				if vi >= vbest or idx == 0: # if best so far (or first entry)
					jbest = j
					wi = vbest # store current vbest as second best, wi
					vbest = vi
					costbest = cost

				elif vi > wi:
					wi = vi

			bbest = costbest - wi + self.eps  # value of new bid

			# store bid & its value
			bidders[nbidder] = i
			bids[nbidder] = bbest
			objects_bidded[nbidder] = jbest

		cdef double* best_bids = self.best_bids
		cdef int* best_bidders = self.best_bidders
		cdef int* best_bidded_objects = self.best_bidded_objects
		cdef double bid_val
		cdef size_t jbid, n, num_successful_bids


		num_successful_bids = 0 # counter of how many succesful bids
		for n in range(num_bidders):  # for each bid made,
			i = bidders[n]
			bid_val = bids[n]
			jbid = objects_bidded[n]
			if bid_val > best_bids[jbid]:
				if best_bidders[jbid] == -1: # if not overwriting existing bid, increment bid counter
					num_successful_bids += 1

				best_bids[jbid] = bid_val
				best_bidders[jbid] = i


		## ASSIGNMENT PHASE
		cdef int prev_i, assignment_idx, bid_ctr
		cdef size_t people_to_unassign_ctr = 0 # counter of how many people have been unassigned
		cdef size_t people_to_assign_ctr = 0 # counter of how many people have been assigned

		bid_ctr = 0
		for j in range(self.num_cols):
			i = best_bidders[j]
			if i != -1:
				p[j] = best_bids[j]
				assignment_idx = person_to_assignment_idx[i]

				# unassign previous i (if any)
				prev_i = object_to_person[j]
				if prev_i != -1:
					people_to_unassign_ctr += 1
					person_to_object[prev_i] = -1

					# let old i take new i's place in unassigned people list for faster reading
					person_to_assignment_idx[i] = -1
					person_to_assignment_idx[prev_i] = assignment_idx
					unassigned_people[assignment_idx] = prev_i

				else:
					unassigned_people[assignment_idx] = -1 # store empty space in assignment list
					person_to_assignment_idx[i] = -1

				# make new assignment
				people_to_assign_ctr += 1
				person_to_object[i] = j
				object_to_person[j] = i

				# bid has been processed, reset best bids store to -1
				best_bidders[j] = -1
				best_bids[j] = -1

				# keep track of number of bids. Stop early if reached all bids
				bid_ctr += 1
				if bid_ctr >= num_successful_bids:
					break

		self.num_unassigned += people_to_unassign_ctr - people_to_assign_ctr
		push_all_left(unassigned_people, person_to_assignment_idx, self.num_unassigned, N)


	cdef int is_optimal(self):
		"""Checks if current solution is a complete solution that satisfies eps-complementary slackness.
		As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
		 it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure"""
		if self.num_unassigned > 0:
			return False
		return self.eCE_satisfied(eps=self.target_eps)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int eCE_satisfied(self, float eps=0.):
		"""Returns True if eps-complementary slackness condition is satisfied"""
		# e-CE: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
		if self.num_unassigned > 0:
			return False

		cdef size_t i, j, k, l, idx, count, num_objects, start
		cdef double choice_cost, v, LHS
		cdef double[:] p = self.p
		cdef long[:] j_counts = self.j_counts
		cdef long[:] i_starts_stops = self.i_starts_stops
		cdef long[:] person_to_object = self.person_to_object
		cdef long[:] flat_j = self.flat_j
		cdef double[:] val = self.val
		cdef double cost


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

		cdef int maximize = self.problem=='max'

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
					if maximize:
						obj += val[glob_idx]
					else:
						obj -= val[glob_idx]

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