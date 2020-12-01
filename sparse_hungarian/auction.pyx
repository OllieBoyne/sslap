cimport cython
import numpy as np
cimport numpy as np
from time import perf_counter

np.import_array()

# tolerance to deal with floating point precision for eCE, due to eps being stored as float 32
cdef float tol = 1e-7

# ctypedef np.int_t DTYPE_t
cdef DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef (int, double) eval_choices(np.ndarray[DTYPE_t, ndim=1, mode="c"] v):
	"""Given a set of values v, return the index j, of the maximum, and the value w of the second largest value
	 Special cases: len(v) == 0 -> (0, -inf)
	 """
	cdef size_t j
	cdef double second_largest, largest, vi
	cdef double* arrptr = <double*> v.data
	cdef size_t idx
	j = 0
	largest = v[0]
	second_largest = - np.inf
	for idx in range(1, v.size):
		vi = arrptr[idx]
		if vi > largest:
			j = idx
			largest = vi

		elif vi > second_largest:
			second_largest = vi

	return j, second_largest

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef size_t argmax(np.ndarray[DTYPE_t, ndim=1, mode="c"] v):
	"""Returns index of largest element in v, with preference to earlier idxs if two idxs are equally large"""
	cdef size_t i
	cdef double largest, vi
	cdef double* arrptr = <double*> v.data
	cdef size_t idx
	j = 0
	largest = v[0]
	for idx in range(1, v.size):
		vi = arrptr[idx]
		if vi > largest:
			j = idx
			largest = vi
	return j

cdef class Bid:
	cdef public size_t i, j
	cdef public double value
	def __init__(self, size_t i, size_t j, double value):
		self.i = i
		self.j = j
		self.value = value

cdef class AuctionSolver:
	#Solver for auction problem
	# Which finds an assignment of N people -> N objects, by having people 'bid' for objects
	cdef size_t num_rows, num_cols
	cdef set unassigned_people

	cdef np.ndarray p
	cdef dict lookup_by_person
	cdef dict lookup_by_object

	cdef np.ndarray person_to_object # index i gives the object, j, owned by person i
	cdef np.ndarray object_to_person # index j gives the person, i, who owns object j

	cdef dict empty_bids
	cdef dict bid_by_object # object -> any workers who bid on them

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
	cdef float debug_timer

	cdef np.ndarray a
	cdef np.ndarray b # bids

	def __init__(self, np.ndarray[np.int_t, ndim=2] loc, np.ndarray[DTYPE_t, ndim=1] val,
				 size_t  num_rows=0, size_t num_cols=0, str problem='min',
				 int max_iter=200, float eps_start=0):
		cdef size_t N = num_rows if num_rows != 0 else loc[:, 0].max() + 1
		cdef size_t M = num_cols if num_rows != 0 else loc[:, 1].max() + 1
		self.num_rows = N
		self.num_cols = M

		self.problem = problem
		self.nits = 0
		self.max_iter = max_iter

		self.unassigned_people = set(np.arange(N))

		# set up price vector j
		self.p = np.zeros(M, dtype=DTYPE)

		# person i -> ndarray of all objects they can bid on
		self.lookup_by_person = {i: loc[:, 1][loc[:, 0]==i] for i in range(N)}

		# object j -> ndarray of all people that can bid on them
		self.lookup_by_object = {j: loc[:, 0][loc[:, 1]==j] for j in range(M)}

		# object j -> ndarray of all people that can did on them in last bidding phase
		self.empty_bids = {j: [] for j in range(M)}
		self.bid_by_object = {j: [] for j in range(M)}.copy()

		self.person_to_object = np.full(N, dtype=np.int, fill_value=-1)
		self.object_to_person = np.full(M, dtype=np.int, fill_value=-1)

		# to save system memory, make v ndarray an empty, and only populate used elements
		self.a = np.empty((N, M), dtype=DTYPE)
		cdef int mult = -1 if problem == 'min' else 1 # need to flip the values if min vs max
		self.a[loc[:, 0], loc[:, 1]] = mult * val * (self.num_rows+1)

		self.b = np.empty((N, M), dtype=DTYPE)

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
		self.meta = {'eCE':True, 'its':0, 'soln_found':True, 'n_assigned':0}

		self.debug_timer = 0

	cpdef np.ndarray solve(self):
		# then, reduce eps while rebidding/assigning
		cdef np.ndarray[DTYPE_t, ndim=1] best_bids # highest bid for each object
		cdef np.ndarray[np.int_t, ndim=1] best_bidders # person who made each high bid

		while True:
			best_bids, best_bidders = self.bidding_phase()
			self.assignment_phase(best_bids, best_bidders)
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
				self.unassigned_people = set(np.arange(self.num_rows))

		# Finished, validate soln
		self.meta['eCE'] = self.eCE_satisfied(eps=self.target_eps)
		self.meta['its'] = self.nits
		self.meta['soln_found'] = self.is_optimal()
		self.meta['n_assigned'] = self.num_rows - len(self.unassigned_people)
		self.meta['obj'] = self.get_obj()
		self.meta['final_eps'] = self.eps
		self.meta['debug_timer'] = f"{1000 * self.debug_timer:.2f}ms"

		return self.person_to_object

	cdef int terminate(self):
		return (self.nits >= self.max_iter) or ((len(self.unassigned_people) == 0) and self.is_optimal())

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef tuple bidding_phase(self):
		cdef size_t i, jbestloc, jbest2loc, jbestglob, nbidder
		cdef np.ndarray[DTYPE_t, ndim=1] aij, vi, prices
		cdef np.ndarray[np.int_t, ndim=1] j
		cdef float vbest, bbest, wi
		cdef double[:, :] b = self.b

		# store bids made by each person, stored in sequential order in which bids are made
		cdef size_t N = self.num_cols
		cdef size_t B = len(self.unassigned_people)  # number of bids to be made
		cdef np.ndarray[DTYPE_t, ndim=1] bids = np.full(B, dtype=DTYPE, fill_value=-1)
		cdef np.ndarray[np.int_t, ndim=1] bidders = np.full(B, dtype=np.int, fill_value=-1)
		cdef np.ndarray[np.int_t, ndim=1] objects_bidded = np.full(B, dtype=np.int, fill_value=-1)

		cdef double* bids_ptr = <double*> bids.data
		cdef int* bidders_ptr = <int*> bidders.data
		cdef int* objects_ptr = <int*> objects_bidded.data

		# each person now makes a bid:
		nbidder = 0 # count how many bids made
		for i in self.unassigned_people: # person, i
			j = self.lookup_by_person[i] # objects j that this person can bid on
			aij = self.a[i, j]
			prices = self.p[j]			# get prices for these objects

			vi = aij - prices  # current value (actual val - price) for each object
			jbestloc, wi = eval_choices(vi)  # idx of largest, value of second largest

			vbest = vi[jbestloc]
			bbest = aij[jbestloc] - wi + self.eps
			jbestglob = j[jbestloc]  # index of best j in global idxing

			# store bid & its value
			bidders_ptr[nbidder] = i
			bids_ptr[nbidder] = bbest
			objects_ptr[nbidder] = jbestglob
			nbidder += 1


		cdef np.ndarray[DTYPE_t, ndim=1] best_bids = np.full(N, dtype=DTYPE, fill_value=-1)
		cdef np.ndarray[np.int_t, ndim=1] best_bidders = np.full(N, dtype=np.int, fill_value=-1)

		## memory views for efficient indexing
		cdef double[:] best_bids_mv = best_bids
		cdef long[:] best_bidders_mv = best_bidders
		cdef double val
		cdef size_t jbid

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

	cdef int eCE_satisfied(self, float eps=0.):
		"""Returns True if eps-complementary slackness condition is satisfied"""
		# e-CE: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
		if len(self.unassigned_people) > 0:
			return False

		cdef int i, j
		for i in range(self.num_rows):
			j = self.person_to_object[i]
			k = self.lookup_by_person[i]
			# a person is unhappy if
			if (self.a[i, j] - self.p[j]) + tol < (self.a[i, k] - self.p[k]).max() - eps:
				return False

		return True

	cdef int is_happy(self, int i):
		"""Returns bool for whether person i is happy"""
		j = self.person_to_object[i]
		k = self.lookup_by_person[i]
		# a person is unhappy if
		if (self.a[i, j] - self.p[j]) < (self.a[i, k] - self.p[k]).max():
			return False
		return True

	cdef float get_obj(self):
		"""Returns current objective value of assignments"""
		cdef float obj
		obj = self.a[np.arange(self.num_rows), self.person_to_object].sum()
		if self.problem is 'min':
			obj *= -1

		return obj / (self.num_rows+1)


cpdef AuctionSolver from_matrix(np.ndarray mat, str problem='min', float eps_start=0,
								int max_iter = 100):
	# Return an Auction Solver from a dense matrix (M, N), where invalid values are -1
	cdef np.ndarray loc, val

	rows, cols = np.nonzero(mat>=0)
	loc = np.zeros((rows.size, 2), dtype=np.int)
	val = np.zeros(rows.size, dtype=DTYPE)

	for i in range(rows.size):
		loc[i, 0] = rows[i]
		loc[i, 1] = cols[i]
		val[i] = mat[rows[i], cols[i]]

	return AuctionSolver(loc, val, problem=problem, eps_start=eps_start, max_iter=max_iter)