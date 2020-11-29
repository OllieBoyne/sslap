import numpy as np


def _fast_append(arr, l):
	"""Fast algorithm to append element l to 1D ndarray, returning ndarray"""
	if arr.size < 95:
		o = np.array(arr.tolist() + [l], dtype=arr.dtype)
	else:
		o = np.append(arr, l).astype(arr.dtype)

	return o


def _fast_remove(arr, l):
	"""Fast implementation of removing an element l to a 1D array, where both contain unique values"""
	return np.setdiff1d(arr, l, assume_unique=True).astype(arr.dtype)


def select_dtype(n):
	"""Select dtype to be used for indexing, depending on size of n"""
	if n < 255:
		return np.uint8
	elif n < 65535:
		return np.uint16
	elif n < 4294967295:
		return np.uint32
	return np.uint64


def _empty_lookup(size, dtype=np.uint32):
	"""Return empty lookup dictionary"""
	return {s: np.array([], dtype=dtype) for s in range(size)}


class SparseMatrixFlag:
	"""Metadata stored about a cost matrix in sparse form. Contains 4 elements:

	2 * dict of row/col : all with that flag in col/row
	bool of size num_rows with 1 for flag on in that row
	bool of size num_cols with 1 for flag on in that col"""

	def __init__(self, flag_name, num_rows, num_cols, use_lookup=True, use_bool=True,
				 row_lookup_dtype=np.uint32, col_lookup_dtype=np.uint32):
		self.flag_name = flag_name

		# save computation time by storing empty lookup dict
		self.empty_row_lookup = _empty_lookup(num_rows, dtype=row_lookup_dtype)
		self.empty_col_lookup = _empty_lookup(num_cols, dtype=col_lookup_dtype)

		self.row_lookup, self.col_lookup = self.empty_row_lookup.copy(), self.empty_col_lookup.copy()
		self.row_bool, self.col_bool = np.zeros(num_rows, dtype=np.bool), np.zeros(num_cols, dtype=np.bool)

		self.num_rows, self.num_cols = num_rows, num_cols
		self.use_lookup = use_lookup
		self.use_bool = use_bool

	def set_lookup(self, data, axis='row'):
		if axis == 'row':
			self.row_lookup = data
		elif axis == 'col':
			self.col_lookup = data

	def lookup(self, idx, axis='row', as_flat=False):
		"""Return indices in lookup table.
		as_flat: return in flat idx notation (row * num_rows + col)"""
		if not as_flat:
			if axis == 'row':
				return self.row_lookup[idx]
			elif axis == 'col':
				return self.col_lookup[idx]
		if as_flat:
			if axis == 'row':
				return idx * self.num_rows + self.row_lookup[idx]
			elif axis == 'col':
				return self.col_lookup[idx] * self.num_rows + idx

	def get_rows(self):
		return self.row_bool

	def get_cols(self):
		return self.col_bool

	def set_cols(self, col_bool):
		self.col_bool = col_bool

	def set_rows(self, row_bool):
		self.row_bool = row_bool

	def set_elem(self, r, c):
		"""Store a single element in flags"""
		if self.use_lookup:
			self.row_lookup[r] = _fast_append(self.row_lookup[r], c)
			self.col_lookup[c] = _fast_append(self.col_lookup[c], r)
		if self.use_bool:
			self.row_bool[r] = 1
			self.col_bool[c] = 1

	def clear_elem(self, r, c):
		"""Remove single element in flags.
		This *may* clear bools, as bool flags show if *any* in row"""
		if self.use_lookup:
			self.row_lookup[r] = _fast_remove(self.row_lookup[r], c)
			self.col_lookup[c] = _fast_remove(self.col_lookup[c], r)
		if self.use_bool:
			if self.row_lookup[r].size == 0:
				self.row_bool[r] = 0
			if self.col_lookup[c].size == 0:
				self.col_bool[c] = 0

	def clear(self):
		if self.use_lookup:
			self.set_lookup(self.empty_row_lookup.copy(), 'row')
			self.set_lookup(self.empty_col_lookup.copy(), 'col')
		if self.use_bool:
			self.row_bool = np.zeros_like(self.row_bool)
			self.col_bool = np.zeros_like(self.col_bool)


def flag_manager_wrap(func):
	"""Allows FlagManager to run on function of same name SparseMatrixFlag"""

	def wrapper(self, name, *args, **kwargs):
		f = func.__name__
		# get SparseMatrixFlag instance, run function of same name
		return getattr(self[name], f)(*args, **kwargs)

	return wrapper


class SparseFlagManager():
	def __init__(self, num_rows, num_cols):
		self.flag_lookup = {}  # flag_name : SparseMatrixFlag instance
		self.num_rows, self.num_cols = num_rows, num_cols

	def add_flag(self, name, use_lookup=True, use_bool=True):
		flag = SparseMatrixFlag(name, num_rows=self.num_rows, num_cols=self.num_cols,
								use_lookup=use_lookup, use_bool=use_bool)
		self.flag_lookup[name] = flag

	@flag_manager_wrap
	def set_lookup(self, name, data, axis='row'):
		pass

	@flag_manager_wrap
	def set_elem(self, name, r, c):
		pass

	@flag_manager_wrap
	def clear_elem(self, name, r, c):
		pass

	@flag_manager_wrap
	def get_rows(self, name):
		pass

	@flag_manager_wrap
	def get_cols(self, name):
		pass

	@flag_manager_wrap
	def lookup(self, name, idx, axis='row', as_flat=False):
		pass

	@flag_manager_wrap
	def set_cols(self, name, col_bool):
		pass

	@flag_manager_wrap
	def set_rows(self, name, row_bool):
		pass

	@flag_manager_wrap
	def clear(self, name):
		pass

	def __getitem__(self, name) -> SparseMatrixFlag:
		return self.flag_lookup[name]


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

		# get dtype of lookup tables, based on number of elements in rows/cols
		self.dtype = select_dtype(self._r * self._c)  # for flattened format, highest element = r * c

		# Set up sparsely defined flags
		self.flags = SparseFlagManager(num_rows=self._r, num_cols=self._c)
		self.flags.add_flag("valid", use_bool=False)  # flag to mark if an element can be picked
		self.flags.add_flag("zero", use_bool=False)  # flag to mark if element is zero
		self.flags.add_flag("starred", use_bool=True)  # flag if zero is currently part of solution
		self.flags.add_flag("covered", use_lookup=False, use_bool=True)  # flag for if row/column is covered
		self.flags.add_flag("prime", use_bool=False)  # flag for if zero is 'primed' (next to be made starred)

		# set lookup of valid entries for each row & col
		valid_mat = (cost_mat >= 0)
		self.flags.set_lookup('valid', {r: np.nonzero(valid_mat[r])[0].astype(self.dtype) for r in range(self._r)}, 'row')
		self.flags.set_lookup('valid', {c: np.nonzero(valid_mat[:, c])[0].astype(self.dtype) for c in range(self._c)}, 'col')

		# set lookup of entries = 0 for each row & col
		is_zero = (cost_mat == 0)
		self.flags.set_lookup('zero', {r: np.nonzero(is_zero[r])[0].astype(self.dtype) for r in range(self._r)}, 'row')
		self.flags.set_lookup('zero', {c: np.nonzero(is_zero[:, c])[0].astype(self.dtype) for c in range(self._c)}, 'col')

		# setup set of all uncovered zeros, stored in flat, row major idxs ((1,0) = 1, (1,1) = 2)
		nz_rows, nz_cols = np.nonzero(is_zero)
		self.uncovered_zeros = set(self._c * nz_rows + nz_cols)
		self.covered_zeros = set()

		self._s = {'row': self._r, 'col': self._c}
		self._ax = {'row': 0, 'col': 1}

	def add_to(self, idx, val, axis='row'):
		"""Add value to all elements in a row.

		Note: if val > row/column minimum, this will result in negative values left in matrix.
		Shouldn't cause issues, as these idxs are stored in all lookups as 'zero'.
		"""
		valid_idxs = self.flags.lookup('valid', idx, axis)
		if valid_idxs.size == 0: return  # end if no valid entries in line

		rows = idx if axis == 'row' else valid_idxs
		cols = idx if axis == 'col' else valid_idxs

		# if adding, need to remove all zeros from tracking
		if val > 0:
			zero_idxs = self.flags.lookup('zero', idx, axis)

			for e in zero_idxs:
				r = e if axis=='col' else idx
				c = e if axis=='row' else idx
				# clear zero
				self.flags.clear_elem('zero', r, c)

				# remove from both sets
				self.covered_zeros.difference_update({self._c * r + c})
				self.uncovered_zeros.difference_update({self._c * r + c})

		# if subtracting, might need to add some zeros to tracking
		if val < 0:
			will_be_zero = (self.costs[rows, cols] <= -val) & (
						self.costs[rows, cols] > 0)  # bool for valid set where cost -> zero
			to_zero_idx = valid_idxs[will_be_zero]  # idxs in actual matrix which are going to zero
			for e in to_zero_idx:
				r = e if axis=='col' else idx
				c = e if axis=='row' else idx
				# store zero
				self.flags.set_elem('zero', r, c)

				# add zero to uncovered/covered set. This can all be made much faster
				is_covered = self.flags.get_rows('covered')[r] or self.flags.get_cols('covered')[c]
				set_to_join = self.covered_zeros if is_covered else self.uncovered_zeros
				set_to_join.update({self._c * r + c})

		self.costs[rows, cols] += val

	def min(self, idx, axis='row'):
		"""Return min in idx row/col"""
		return self.get(idx, axis).min()

	def reduce_rows(self):
		"""Reduce every row by its minimum valid value"""
		for r in range(self._r):
			minval = self.costs[r, self.flags.lookup('valid', r, 'row')].min()
			self.add_to(r, -minval, 'row')

	def reduce_cols(self):
		"""Reduce every row by its minimum valid value"""
		for c in range(self._c):
			minval = self.costs[self.flags.lookup('valid', c, 'col'), c].min()
			self.add_to(c, -minval, 'col')

	def get(self, idx, axis='row'):
		if axis == 'col':
			return self.costs[self.flags.lookup('valid', idx, 'col'), idx]
		else:
			return self.costs[idx, self.flags.lookup('valid', idx, 'row')]

	def count_zeros(self, idx, axis='row'):
		return np.count_nonzero(self.get(idx, axis) == 0)

	def flatten_idxs(self, r, c):
		"""No more than one of r and c will be a list. Will return a list of similar size (or single int if r and c are ints)
		Returns indexes in flattened format for uncovered zero set."""
		return self._c * r + c

	def cover(self, idx, axis='row'):
		"""Cover row/col, remove newly covered zeros from uncovered_zeros"""
		if axis == 'row':
			self.flags.get_rows('covered')[idx] = 1
		elif axis == 'col':
			self.flags.get_cols('covered')[idx] = 1

		# remove from uncovered zeros, add to covered zeros
		zero_idxs = self.flags.lookup('zero', idx, axis)
		flat_zero_idxs = self.flatten_idxs(idx, zero_idxs) if axis == 'row' else self.flatten_idxs(zero_idxs, idx)

		self.uncovered_zeros.difference_update(flat_zero_idxs)
		self.covered_zeros.update(flat_zero_idxs)

	def bulk_cover(self, idxs, axis='row'):
		"""Cover all row/cols provided by idxs. Bulk remove newly covered zeros from uncovered_zeros"""

		if idxs.size == 0:
			return None

		if axis == 'row':
			self.flags.get_rows('covered')[idxs] = 1
		elif axis == 'col':
			self.flags.get_cols('covered')[idxs] = 1

		# update covered zeros for each index
		for idx in idxs:
			self.covered_zeros.update(self.flags.lookup('zero', idx, axis, as_flat=True))

		# remove all covered zeros from uncovered zeros
		self.uncovered_zeros.difference_update(self.covered_zeros)

	def uncover(self, idx, axis='row'):
		"""Uncover row/col, add all newly uncovered zeros to uncovered_zeros"""
		if axis == 'row':
			self.flags.get_rows('covered')[idx] = 0
		elif axis == 'col':
			self.flags.get_cols('covered')[idx] = 0

		# add to uncovered zeros, remove from covered zeros
		zero_idxs = self.flags.lookup('zero', idx, axis)  # all zeros in axis

		# of those zeros, ones in which the opposite axis is also uncovered
		opp_axis_covered = self.flags.get_rows('covered') if axis=='col' else self.flags.get_cols('covered')
		uncovered_zeros = zero_idxs[~opp_axis_covered[zero_idxs]]

		# convert to flattened form
		flat_zero_idxs = self.flatten_idxs(idx, uncovered_zeros) if axis == 'row' else self.flatten_idxs(uncovered_zeros, idx)
		self.uncovered_zeros.update(flat_zero_idxs)
		self.covered_zeros.difference_update(flat_zero_idxs)

	def uncover_all(self):
		"""On uncover all, clear flags, and move all covered zeros to uncovered"""
		self.flags.clear('covered')
		self.uncovered_zeros.update(self.covered_zeros)
		self.covered_zeros = set()

	def get_uncovered_zero(self, f=False):
		"""Returns first uncovered zero found"""
		# once an uncovered zero has been used in step 4, it will *always* be covered or starred, so
		# popping here is acceptable
		try:
			flat = self.uncovered_zeros.pop()
			r, c = flat // self._c, flat % self._c
			return r, c
		except KeyError:
			return False

	def is_zero_alone(self, r, c):
		"""Given a zero at pos (r,c) returns True if this zero is the only one in its row & column"""
		assert self.costs[r, c] == 0, "is_zero_alone called for an element that is not zero."
		return (self.flags.lookup('zero', r, 'row').size == 1) and (self.flags.lookup('zero', c, 'col').size == 1)

	def get_min_uncovered(self):
		"""Get the smallest valid value NOT in a covered row or column.
		NOTE: Inefficent?"""
		minval = 1e10
		for r in np.nonzero(~self.flags.get_rows('covered'))[0]:  # for each uncovered row
			colidxs = self.flags.lookup('valid', r, 'row')  # get all valid columns idxs
			colidxs = colidxs[~self.flags.get_cols('covered')[colidxs]]  # only keep column idxs that ARENT covered
			if colidxs.size > 0:
				minval = min(minval, self.costs[r, colidxs].min())
		assert minval != 1e10, "This matrix is infeasible. No solution found."
		return minval

	def get_sol(self):
		"""When solveable, return R (_r,), C (_c) assignments"""
		# slow, might need to revisit
		R, C = np.zeros(self._r, dtype=np.int32), np.zeros(self._r, dtype=np.int32)
		for row in range(self._r):  # for each row
			col = self.flags.lookup('starred', row, 'row')  # col in which zero is
			R[row], C[row] = row, col

		return R, C

	def __repr__(self):
		out = ""
		row_format = "{:>5}" * (self._c + 1)  # 5 chars per col
		out += row_format.format("", *[" C"[i] for i in self.flags.get_cols('covered')])
		nfmt = 'int' if 'int' in self.dtype.name else 'float'
		for r in range(self._r):
			line = [" C"[self.flags.get_rows('covered')[r]]]
			for c in range(self._c):
				if c in self.flags.lookup('valid', r, 'row'):
					v = f"{self.costs[r, c]:.1f}" if nfmt is float else self.costs[r, c]
					if self._c * r + c in self.uncovered_zeros:
						v = f".{v}"
					line += [f"{v}{'*' * (c in self.flags.lookup('starred',r,'row'))}{'′' * (c in self.flags.lookup('prime',r,'row'))}"]
				else:
					line += ["-"]

			out += "\n" + row_format.format(*line)

		out += f"\n{'-' * 100}"
		return out
