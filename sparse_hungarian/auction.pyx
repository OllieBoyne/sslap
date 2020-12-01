cimport cython
cimport numpy as np
import numpy as np

np.import_array()

# tolerance to deal with floating point precision for eCE, due to eps being stored as float 32
cdef float tol = 1e-7

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray get_top_two(np.ndarray arr):
	# Return indices of top and second top item
	cdef int topidxs[2], i
	cdef np.ndarray top2

	if arr.size == 1:
		return np.array([0, -1]) # only one index, so top two are [0, -1]

	elif arr.size == 2:
		top2 = np.array([0, 1])
	else:
		top2 = np.argpartition(-arr, 2)[:2] # top 2 indices in unsorted order

	if arr[top2[0]] >= arr[top2[1]]:
		return top2 # in correct order, so return as such

	return top2[::-1] # otherwise flip order

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


	cdef np.ndarray a
	cdef np.ndarray b # bids

	def __init__(self, np.ndarray loc, np.ndarray val, int num_rows=0, int num_cols=0, str problem='min',
				 int max_iter=200):
		N = num_rows if num_rows != 0 else loc[:, 0].max() + 1
		M = num_cols if num_rows != 0 else loc[:, 1].max() + 1
		self.num_rows = N
		self.num_cols = M

		self.problem = problem
		self.nits = 0
		self.max_iter = max_iter

		self.unassigned_people = set(np.arange(N))

		# set up price vector j
		self.p = np.zeros(M, dtype=np.float32)

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
		self.a = np.empty((N, M), dtype=np.float32)
		cdef int mult = -1 if problem == 'min' else 1 # need to flip the values if min vs max
		self.a[loc[:, 0], loc[:, 1]] = mult * val * self.num_rows

		self.b = np.empty((N, M), dtype=np.float32)

		# Calculate optimum initial eps and target eps
		cdef float C  # = max |aij| for all i, j in A(i)
		cdef int approx_ints # Esimated
		C = np.abs(val).max()


		# choose eps values
		self.eps = 10 # C/2
		self.target_eps = 1/(min(N, M))

		self.optimal_soln_found = False
		self.meta = {'eCE':True, 'its':0, 'soln_found':True, 'n_assigned':0}

	cpdef np.ndarray solve(self):
		# then, reduce eps while rebidding/assigning
		while True:
			while len(self.unassigned_people) > 0:
				self.bidding_phase()
				self.assignment_phase()
				self.nits += 1

				# if self.nits > self.max_iter:
				# 	break

			break
			# if self.terminate():
			# 	break

			# else:
			# 	self.eps = max(1, self.delta/(self.theta**self.k))
			# 	self.k += 1

		# Finished, validate soln
		self.meta['eCE'] = self.eCE_satisfied(eps=self.eps)
		self.meta['its'] = self.nits
		self.meta['soln_found'] = self.is_optimal()
		self.meta['n_assigned'] = self.num_rows - len(self.unassigned_people)
		self.meta['obj'] = self.get_obj()
		self.meta['final_eps'] = self.eps

		return self.person_to_object

	cdef int terminate(self):
		return (self.nits >= self.max_iter) or ((len(self.unassigned_people) == 0) and self.is_optimal())

	cdef void bidding_phase(self):
		for i in self.unassigned_people: # person, i
			j = self.lookup_by_person[i] # objects j that this person can bid on
			aij = self.a[i, j]
			vi = aij - self.p[j]  # current value (actual val - price) for each object
			jbestloc, jbest2loc = get_top_two(vi)  # idxs of best two, indexed FROM j, not globally

			vbest = vi[jbestloc]

			if jbest2loc == -1: # if no 'second best' found in set, bid infinite
				bbest = np.inf

			else:
				wi = vi[jbest2loc]
				bbest = aij[jbestloc] - wi + self.eps

			jbestglob = j[jbestloc]  # index of best j in global idxing

			# store bid & its value
			self.b[i, jbestglob] = bbest
			self.bid_by_object[jbestglob].append(i)

	cdef void assignment_phase(self):
		cdef tuple tup
		for j in range(self.num_cols): #self.objects_in_market: #
			bidding_i = self.bid_by_object[j]

			if len(bidding_i) > 0:

				ibest = bidding_i[np.argmax(self.b[bidding_i, j])]  # highest bidding person of all bidders wins the bid
				self.p[j] = self.b[ibest, j]  # update price to reflect new bid

				# assign ibest to j
				self.assign_pairing(ibest, j)

			# clear bids
			self.bid_by_object[j] = []


	cdef void assign_pairing(self, int i, int j):
		"""Assign person i to object j"""
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

		return obj / self.num_rows


cpdef AuctionSolver from_matrix(np.ndarray mat, str problem='min'):
	# Return an Auction Solver from a dense matrix (M, N), where invalid values are -1
	cdef np.ndarray loc, val

	rows, cols = np.nonzero(mat>=0)
	loc = np.zeros((rows.size, 2), dtype=np.int)
	val = np.zeros(rows.size, np.int)

	for i in range(rows.size):
		loc[i, 0] = rows[i]
		loc[i, 1] = cols[i]
		val[i] = mat[rows[i], cols[i]]

	return AuctionSolver(loc, val, problem=problem)