cimport cython
import numpy as np
cimport numpy as np
from time import perf_counter
from cython.parallel import prange


np.import_array()

# tolerance to deal with floating point precision for eCE, due to eps being stored as float 32
cdef float tol = 1e-7

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


cdef class AuctionSolver:
	#Solver for auction problem
	# Which finds an assignment of N people -> N objects, by having people 'bid' for objects
	cdef size_t num_rows, num_cols
	cdef set all_people, unassigned_people

	cdef np.ndarray p
	cdef dict lookup_by_person
	cdef dict lookup_by_object

	cdef long[:] i_starts_stops, flat_j, j_counts
	cdef double[:] valmv  # memory view of all values
	cdef np.ndarray person_to_object # index i gives the object, j, owned by person i
	cdef np.ndarray object_to_person # index j gives the person, i, who owns object j

	cdef float eps
	cdef float target_eps
	cdef float delta, theta
	cdef int k

	#meta
	cdef str problem
	cdef int nits
	cdef int max_iter
	cdef int optimal_soln_found
	cdef public dict meta
	cdef float extreme
	cdef dict debug_timer


	def __init__(self, np.ndarray[np.int_t, ndim=2] loc, np.ndarray[DTYPE_t, ndim=1] val,
				 size_t  num_rows=0, size_t num_cols=0, str problem='min',
				 size_t max_iter=200, float eps_start=0):

		self.debug_timer = {'bidding':0, 'assigning':0, 'setup':0, 'forbids':0}
		t0 = perf_counter()

		cdef size_t N = loc[:, 0].max() + 1
		cdef size_t M = loc[:, 1].max() + 1
		self.num_rows = N
		self.num_cols = M

		self.problem = problem
		self.nits = 0
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
		self.flat_j = loc[:, 1]

		self.person_to_object = np.full(N, dtype=np.int, fill_value=-1)
		self.object_to_person = np.full(M, dtype=np.int, fill_value=-1)

		# to save system memory, make v ndarray an empty, and only populate used elements
		if problem == 'min':
			val *= -1 # auction algorithm always maximises, so flip value signs for min problem

		cdef np.ndarray[DTYPE_t, ndim=1] v = (val * (self.num_rows+1))
		self.valmv = v

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

		self.optimal_soln_found = False
		self.meta = {}
		self.debug_timer['setup'] += perf_counter() - t0

	cpdef np.ndarray solve(self):
		# then, reduce eps while rebidding/assigning
		cdef np.ndarray[DTYPE_t, ndim=1] best_bids # highest bid for each object
		cdef np.ndarray[np.int_t, ndim=1] best_bidders # person who made each high bid

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
				self.eps = self.eps*self.theta
				self.k += 1

				if self.eps <= self.target_eps:  # terminate, shown to be optimal for eps < 1/n
					break

				self.person_to_object[:] = -1
				self.object_to_person[:] = -1
				self.unassigned_people = self.all_people.copy()


		# Finished, validate soln
		self.meta['eCE'] = self.eCE_satisfied(eps=self.target_eps)
		self.meta['its'] = self.nits
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
		cdef size_t i, jbestloc, jbest2loc, jbestglob, nbidder, idx
		cdef np.ndarray[DTYPE_t, ndim=1] prices, aij
		cdef float vbest, bbest, wi, vi
		cdef long[:] j

		# store bids made by each person, stored in sequential order in which bids are made
		cdef size_t N = self.num_cols
		cdef size_t B = len(self.unassigned_people)  # number of bids to be made
		cdef np.ndarray[DTYPE_t, ndim=1] bids = np.full(B, dtype=DTYPE, fill_value=-1)
		cdef np.ndarray[np.int_t, ndim=1] bidders = np.full(B, dtype=np.int, fill_value=-1)
		cdef np.ndarray[np.int_t, ndim=1] objects_bidded = np.full(B, dtype=np.int, fill_value=-1)

		cdef double* bids_ptr = <double*> bids.data
		cdef int* bidders_ptr = <int*> bidders.data
		cdef int* objects_ptr = <int*> objects_bidded.data

		cdef double* p_ptr = <double*> self.p.data
		cdef double[:] arow_mv

		# each person now makes a bid:
		cdef np.ndarray[np.int_t, ndim=1] unassigned_people = np.fromiter(self.unassigned_people, dtype=np.int)
		cdef int[:] unassigned_people_mv = unassigned_people
		cdef size_t n_unassigned = unassigned_people.size

		for nbidder in range(n_unassigned):
			i = unassigned_people_mv[nbidder]
			j = self.get_biddable_objects(i)
			count = self.get_num_biddable_objects(i)

			# go through each object, storing its index & value if vi is largest, and value if vi is second largest
			arow_mv = self.get_object_costs(i)
			jbestloc = 0
			vbest = - inf
			wi = - inf

			for idx in range(count):
				vi = arow_mv[idx] - p_ptr[j[idx]]
				if vi > vbest:
					jbestloc = idx
					vbest = vi

				elif vi > wi:
					wi = vi

			jbestglob = j[jbestloc]  # index of best j in global idxing
			bbest = arow_mv[jbestloc] - wi + self.eps

			# store bid & its value
			bidders_ptr[nbidder] = i
			bids_ptr[nbidder] = bbest
			objects_ptr[nbidder] = jbestglob

		cdef np.ndarray[DTYPE_t, ndim=1] best_bids = np.full(N, dtype=DTYPE, fill_value=-1)
		cdef np.ndarray[np.int_t, ndim=1] best_bidders = np.full(N, dtype=np.int, fill_value=-1)

		## memory views for efficient indexing
		cdef double[:] best_bids_mv = best_bids
		cdef long[:] best_bidders_mv = best_bidders
		cdef double val
		cdef size_t jbid, n

		for n in range(B):  # for each bid made,
			i = bidders_ptr[n]
			val = bids_ptr[n]
			jbid = objects_ptr[n]
			if val > best_bids_mv[jbid]:
				best_bids_mv[jbid] = val
				best_bidders_mv[jbid] = i

		cdef tuple out = (best_bids, best_bidders)
		return out

	cdef void assignment_phase(self, np.ndarray[DTYPE_t, ndim=1] best_bids,
							   np.ndarray[np.int_t, ndim=1] best_bidders):
		cdef tuple tup
		cdef list bidding_i
		cdef np.ndarray[DTYPE_t, ndim=1] bids
		cdef size_t i

		# pointers to 1D arrays for quick accessing
		cdef double* p = <double*> self.p.data
		cdef double* bidsptr = <double*> best_bids.data
		cdef int* bidderptr = <int*> best_bidders.data

		for j in range(self.num_cols):
			i = bidderptr[j]
			if i != -1:
				p[j] = bidsptr[j]
				self.assign_pairing(i, j)


	cdef void assign_pairing(self, size_t i, size_t j):
		"""Assign person i to object j"""
		cdef int prev_i
		# unassign previous i (if any)
		prev_i = self.object_to_person[j]
		if prev_i != -1:
			self.unassigned_people.update({prev_i})
			self.person_to_object[prev_i] = -1

		# make new assignment
		self.unassigned_people.difference_update({i})
		self.person_to_object[i] = j
		self.object_to_person[j] = i


	cdef int is_optimal(self):
		"""Checks if current solution is a complete solution that satisfies eps-complementary slackness.
		As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
		 it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure"""
		if len(self.unassigned_people) > 0:
			return False

		return self.eCE_satisfied()

	cdef long[:] get_biddable_objects(self, size_t i):
		"""Return memory view of biddable objects for a given person i"""
		cdef long[:] jbounds, j
		jbounds = self.i_starts_stops[i:i+2] # start and stop of biddable objects, in flattened list of *all* person, object, value
		j = self.flat_j[jbounds[0]:jbounds[1]]
		return j

	cdef double[:] get_object_costs(self, size_t i):
		"""Return memory view of all costs of biddable objects for a given person i"""
		cdef long[:] jbounds
		cdef double[:] vals
		jbounds = self.i_starts_stops[i:i+2] # start and stop of biddable objects
		vals = self.valmv[jbounds[0]:jbounds[1]]
		return vals

	cdef long get_num_biddable_objects(self, size_t i):
		"""Return the number of objects that will be returned. Faster than len(self.get_biddable_objects(i))"""
		cdef long[:] j_counts = self.j_counts
		cdef long N = j_counts[i]
		return N

	cdef int eCE_satisfied(self, float eps=0.):
		"""Returns True if eps-complementary slackness condition is satisfied"""
		# e-CE: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
		if len(self.unassigned_people) > 0:
			return False

		cdef size_t i, j, kidx, count
		cdef double row_max_val, val
		cdef long[:] k
		cdef double[:] p = self.p
		cdef double[:] arow

		for i in range(self.num_rows):
			j = self.person_to_object[i]
			k = self.get_biddable_objects(i)
			count = len(k)
			arow = self.get_object_costs(i)

			row_max_val = - np.inf
			for kidx in range(count):
				val = arow[kidx] - p[k[kidx]]
				if val > row_max_val:
					row_max_val = val

			# a person is unhappy if
			if (arow[j] - p[j]) + tol < row_max_val - eps:
				return False

		return True

	cdef float get_obj(self):
		"""Returns current objective value of assignments"""
		cdef float obj
		cdef size_t i, j, jlocidx, counts, idx
		cdef long[:] js
		cdef double[:] vs
		for i in range(self.num_rows):
			# due to the way data is stored, need to go do some searching to find the corresponding value
			# to assignment i -> j
			j = self.person_to_object[i] # chosen j
			js = self.get_biddable_objects(i)
			vs = self.get_object_costs(i)
			counts = self.get_num_biddable_objects(i)
			for idx in range(counts):
				if js[idx] == j:
					obj += vs[idx]
					break

		if self.problem is 'min':
			obj *= -1

		return obj / (self.num_rows+1)


cpdef AuctionSolver from_matrix(np.ndarray mat, str problem='min', float eps_start=0,
								size_t max_iter = 100):
	# Return an Auction Solver from a dense matrix (M, N), where invalid values are -1
	cdef np.ndarray[np.int_t, ndim=2] loc
	cdef np.ndarray[DTYPE_t, ndim=1] val
	cdef tuple locs

	locs = np.nonzero(mat>=0)
	loc = np.stack(locs, axis=-1).astype(np.int32)
	val = mat[locs]

	return AuctionSolver(loc, val, problem=problem, eps_start=eps_start, max_iter=max_iter)