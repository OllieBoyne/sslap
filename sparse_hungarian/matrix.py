import numpy as np

def _fast_append(arr, l):
	"""Fast algorithm to append element l to 1D ndarray, returning ndarray"""
	if arr.size < 95:
		return np.array(arr.tolist() + [l])
	else:
		return np.append(arr, l)

def _fast_remove(arr, l):
	"""Fast implementation of removing an element l to a 1D array, where both contain unique values"""
	return np.setdiff1d(arr, l, assume_unique=True)


class SparseCostMatrix():
	"""Semi-Sparse Cost Matrix initialised for fast Hungarian Solving.
	The matrix is not actually sparse, but all non-zero entries are stored in a separate log for efficient calculations.

	Sparse matrix in which all non-empty entries are stricly positive"""

	def __init__(self, cost_mat: np.ndarray):
		"""
		Parse a cost matrix D into a sparse linear assignment lookup matrix.
		D must be a 2D np.ndarray, with strictly positive costs for valid entries,
		Negative/inf/NaN will all be counted as invalid entries
		:param cost_mat:
		"""

		cost_mat[np.isinf(cost_mat)] = -1  # convert any NaN -> -1
		cost_mat[np.isnan(cost_mat)] = -1  # convert any NaN -> -1

		self.dtype = cost_mat.dtype

		self.costs = cost_mat
		self._r, self._c = cost_mat.shape

		# a valid entry is one which is not np.inf
		valid_mat = (cost_mat >= 0)
		self.valid_by_row = {r: np.nonzero(valid_mat[r])[0] for r in range(self._r)}
		self.valid_by_col = {c: np.nonzero(valid_mat[:, c])[0] for c in range(self._c)}
		self.valid_lookup = dict(row=self.valid_by_row, col=self.valid_by_col)

		# a zero entry is one which is == 0
		is_zero = (cost_mat == 0)
		self.zero_by_row = {r: np.nonzero(is_zero[r])[0] for r in range(self._r)}
		self.zero_by_col = {c: np.nonzero(is_zero[:, c])[0] for c in range(self._c)}
		self.zero_lookup = dict(row=self.valid_by_row, col=self.valid_by_col)

		# A starred entry is a zero marked as a choice
		self.starred_by_row = {r: np.array([]) for r in range(self._r)}
		self.starred_by_col = {c: np.array([]) for c in range(self._c)}

		# A primed entry is another zero marking
		self.prime_by_row = {r: np.array([]) for r in range(self._r)}
		self.prime_by_col = {c: np.array([]) for c in range(self._c)}

		# different 'flags' added to rows/columns during algorithm - starred, covered
		self.starred_rows = np.zeros(self._r, dtype=np.bool)  # boolean store of rows which have successful assignments
		self.starred_columns = np.zeros(self._c, dtype=np.bool)  # boolean store of rows which have successful assignments

		self.covered_rows = np.zeros(self._r, dtype=np.bool)  # boolean store of rows which have been 'covered'
		self.covered_columns = np.zeros(self._c, dtype=np.bool)  # boolean store of rows which have been 'covered'

		self.prime_rows = np.zeros(self._r, dtype=np.bool)  # boolean store of rows which have been 'primed'
		self.prime_columns = np.zeros(self._c, dtype=np.bool)  # boolean store of rows which have been 'primed'

		self._s = {'row': self._r, 'col': self._c}
		self._ax = {'row': 0, 'col': 1}

	def clear(self, t='prime'):
		if t == 'prime':
			self.prime_rows = np.zeros_like(self.prime_rows)
			self.prime_columns = np.zeros_like(self.prime_columns)
			self.prime_by_row = {r: np.array([]) for r in range(self._r)}
			self.prime_by_col = {c: np.array([]) for c in range(self._c)}
		elif t == 'cover':
			self.covered_rows = np.zeros_like(self.covered_rows)
			self.covered_columns = np.zeros_like(self.covered_columns)
		else:
			raise NotImplementedError

	def store_zero(self, r, c):
		"""Store new zero at position (r, c)"""
		self.zero_by_row[r] = _fast_append(self.zero_by_row[r], c)
		self.zero_by_col[c] = _fast_append(self.zero_by_col[c], r)

	def clear_zero(self, r, c):
		"""Remove zero at position (r, c)"""
		self.zero_by_row[r] = _fast_remove(self.zero_by_row[r], c)
		self.zero_by_col[c] = _fast_remove(self.zero_by_col[c], r)

	def star_zero(self, r, c):
		""" 'Star' a zero, marking it as a selected option"""
		self.starred_rows[r] = 1
		self.starred_columns[c] = 1
		self.starred_by_row[r] = _fast_append(self.starred_by_row[r], c)
		self.starred_by_col[c] = _fast_append(self.starred_by_col[c], r)

	def unstar_zero(self, r, c):
		""" Un-'Star' a zero, marking it as a selected option"""
		self.starred_rows[r] = 0
		self.starred_columns[c] = 0
		self.starred_by_row[r] = _fast_remove(self.starred_by_row[r], c)
		self.starred_by_col[c] = _fast_remove(self.starred_by_col[c], r)

	def prime_zero(self, r, c):
		""" 'Prime' a zero, marking it as a primed option"""
		self.prime_rows[r] = 1
		self.prime_columns[c] = 1
		self.prime_by_row[r] = _fast_append(self.prime_by_row[r], c)
		self.prime_by_col[c] = _fast_append(self.prime_by_col[c], r)

	def unprime_zero(self, r, c):
		self.prime_rows[r] = 0
		self.prime_columns[c] = 0
		self.prime_by_row[r] = _fast_remove(self.prime_by_row[r], c)
		self.prime_by_col[c] = _fast_remove(self.prime_by_col[c], r)

	def add_to(self, idx, val, axis='row'):
		"""Add value to all elements in a row.

		Note: if val > row/column minimum, this will result in negative values left in matrix.
		Shouldn't cause issues, as these idxs are stored in all lookups as 'zero'.
		"""
		nz_idxs = self.valid_lookup[axis][idx]
		if nz_idxs.size == 0: return  # end if no non zero in line

		rows = idx if axis == 'row' else nz_idxs
		cols = idx if axis == 'col' else nz_idxs

		# self.min_lookup[axis][idx] += val  # need to track change in minima

		# if adding, need to remove all zeros from tracking
		if val > 0:
			zero_idxs = self.zero_lookup[axis][idx]
			if axis == 'row':
				for c in zero_idxs:
					self.clear_zero(idx, c)
			elif axis == 'col':
				for r in zero_idxs:
					self.clear_zero(r, idx)

		# if subtracting, might need to add some zeros to tracking
		if val < 0:
			will_be_zero = (self.costs[rows, cols] <= -val)&(self.costs[rows, cols] > 0)  # bool for valid set where cost -> zero
			to_zero_idx = self.valid_lookup[axis][idx][will_be_zero]  # idxs in actual matrix which are going to zero
			if axis == 'row':
				for c in to_zero_idx:
					self.store_zero(idx, c)
			elif axis == 'col':
				for r in to_zero_idx:
					self.store_zero(r, idx)

		self.costs[rows, cols] += val

	def min(self, idx, axis='row'):
		"""Return min in idx row/col"""
		return self.get(idx, axis).min()


	def reduce_rows(self):
		"""Reduce every row by its minimum valid value"""
		for r in range(self._r):
			minval = self.costs[r, self.valid_by_row[r]].min()
			self.add_to(r, -minval, 'row')

	def reduce_cols(self):
		"""Reduce every row by its minimum valid value"""
		for c in range(self._c):
			minval = self.costs[self.valid_by_col[c], c].min()
			self.add_to(c, -minval, 'col')

	def get(self, idx, axis='row'):
		if axis == 'col':
			return self.costs[self.valid_by_col[idx], idx]
		else:
			return self.costs[idx, self.valid_by_row[idx]]

	def count_zeros(self, idx, axis='row'):
		return np.count_nonzero(self.get(idx, axis) == 0)

	def get_uncovered_zero(self,f=False):
		"""Returns first uncovered zero found"""
		for c in np.nonzero(~self.covered_columns)[0]:
			zero_idxs = self.zero_by_col[c]
			for r in zero_idxs:
				if not self.covered_rows[r]:
					return r, c
		return False

	def is_zero_alone(self, r, c):
		"""Given a zero at pos (r,c) returns True if this zero is the only one in its row & column"""
		assert self.costs[r, c] == 0, "is_zero_alone called for an element that is not zero."
		return (self.zero_by_row[r].size == 1) and (self.zero_by_col[c].size == 1)

	def get_min_uncovered(self):
		"""Get the smallest valid value NOT in a covered row or column.
		NOTE: Inefficent?"""
		minval = 1e10
		for r in np.nonzero(~self.covered_rows)[0]:  # for each uncovered row
			colidxs = self.valid_by_row[r]  # get all valid columns idxs
			colidxs = colidxs[~self.covered_columns[colidxs]]  # only keep column idxs that ARENT covered
			if colidxs.size>0:
				minval = min(minval, self.costs[r, colidxs].min())
		return minval

	def get_sol(self):
		"""When solveable, return R (_r,), C (_c) assignments"""
		# slow, might need to revisit
		R, C = np.zeros(self._r, dtype=np.int32), np.zeros(self._r, dtype=np.int32)
		for row in range(self._r):  # for each row
			col = self.starred_by_row[row][0]  # col in which zero is
			R[row], C[row] = row, col

		return R, C

	def __repr__(self):
		out = ""
		row_format = "{:>5}" * (self._c + 1) # 5 chars per col
		out += row_format.format("", *[" C"[i] for i in self.covered_columns])
		nfmt = 'int' if 'int' in self.dtype.name else 'float'
		for r in range(self._r):
			line = [" C"[self.covered_rows[r]]]
			for c in range(self._c):
				if c in self.valid_by_row[r]:
					v = f"{self.costs[r,c]:.1f}" if nfmt is float else self.costs[r, c]
					line += [f"{v}{'*'*(c in self.starred_by_row[r])}{'′'*(c in self.prime_by_row[r])}"]
				else:
					line += ["-"]

			out += "\n" + row_format.format(*line)

		out += f"\n{'-'*100}"

		return out
