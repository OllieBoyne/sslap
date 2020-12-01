cimport cython
cimport numpy as np
import numpy as np

# We need to initialize NumPy.
np.import_array()

cdef _empty_lookup(size, dtype=np.uint32):
	"""Return empty lookup dictionary"""
	cdef dict out
	cdef int s
	for s in range(size):
		out[s] = np.array([])
	return out

cdef class SparseLookupMatrix:
	# Metadata stored about a cost matrix in sparse form. Contains 4 elements:
	# 2 * dict of row/col : all with that flag in col/row
	# bool of size num_rows with 1 for flag on in that row
	# bool of size num_cols with 1 for flag on in that col

	cdef dict row_lookup, col_lookup
	cdef np.ndarray row_bool, col_bool

	cdef size_t num_rows, num_cols
	cdef bool use_lookup, use_bool

	def __init__(self, size_t num_rows, size_t num_cols, bool use_lookup=True, bool use_bool = True):
		# Save computation time by storing empty lookup dict

		self.empty_row_lookup = _empty_lookup(num_rows)
		self.empty_col_lookup = _empty_lookup(num_cols)

		self.row_lookup = self.empty_row_lookup.copy()
		self.col_lookup = self.empty_col_lookup.copy()

		self.row_bool = np.zeros(num_rows, dtype=np.bool)
		self.col_bool = np.zeros(num_cols, dtype=np.bool)

		self.num_rows, self.num_cols = num_rows, num_cols
		self.use_lookup, self.use_bool = use_lookup, use_bool


	def set_elem(self, r, c):
		# add flag at pos r, c
		# Row major representation
		c_idx = self.Row_R[r] + np.searchsorted(self.Row_C[self.Row_R[r]:self.Row_R[r+1]], c)
		self.Row_R[r+1:] += 1 # all subsequent row data shifted one forward for CSR
		self.Row_C.insert(c_idx, c)

		# Col major representation
		r_idx = self.Col_C[c] + np.searchsorted(self.Col_R[self.Col_C[c]:self.Col_C[c+1]], c)
		self.Col_C[c+1:] += 1 # all subsequent row data shifted one forward for CSR
		self.Col_R.insert(r_idx, r)


# CSR & CSC based indexing (but indexing from a 2D numpy array)
cdef class HungarianMatrix:
	cdef np.ndarray V
	cdef np.ndarray Row_C, Row_R # CSR representation
	cdef np.ndarray Col_C, Col_R # CSC representation
	cdef np.ndarray mat

	cdef size_t num_rows, num_cols
	cdef SparseMatrix zeros
	cdef dict flags

	# flags
	cdef np.ndarray covered_rows, covered_cols
	cdef covered_zeros, uncovered_zeros

	def __init__(self, np.ndarray loc, np.ndarray val, int num_rows=0, int num_cols=0):
		# Pass in data, of size (N x 2), where N are the number of valid entires in the matrix
		# & two columns depict row, column
		# + val, an ndarray of size (N,) for the value at each entry
		# If num_rows/num_cols not given, estimate from data

		cdef int r, c
		cdef size_t R, C, S
		cdef np.ndarray row_major_order, col_major_order

		R = num_rows if num_rows != 0 else loc[:, 0].max() + 1
		C = num_cols if num_rows != 0 else loc[:, 1].max() + 1
		S = loc.shape[0]

		self.num_rows = R
		self.num_cols = C

		row_major_order = np.lexsort((loc[:, 1], loc[:, 0]))
		col_major_order = np.lexsort((loc[:, 0], loc[:, 1]))

		self.Row_C = np.zeros(S, dtype=np.int)
		self.Row_R = np.zeros(S, dtype=np.int)
		self.Row_R = get_steps(loc[:, 0][row_major_order], R)
		self.Row_C = loc[:, 1][row_major_order]

		self.Col_R = np.zeros(S, dtype=np.int)
		self.Col_C = np.zeros(S, dtype=np.int)
		self.Col_R = loc[:, 0][col_major_order]
		self.Col_C = get_steps(loc[:, 1][col_major_order], C)

		self.covered_rows = np.zeros(R, dtype=np.bool)
		self.covered_cols = np.zeros(C, dtype=np.bool)

		# initialise matrix
		self.mat = np.empty((R, C), dtype=val.dtype)
		self.mat[loc[:, 0], loc[:, 1]] = val

		# initialise flags
		self.zeros = SparseMatrix(R, C)

		self.flags = dict(zeros=self.zeros)

	cpdef np.ndarray valid_in_row(self, int row):
		return self.Row_C[self.Row_R[row]:self.Row_R[row + 1]]

	cpdef np.ndarray valid_in_col(self, int col):
		return self.Col_R[self.Col_C[col]:self.Col_C[col + 1]]

	cpdef np.ndarray get_row_values(self, int row):
		return self.mat[row, self.valid_in_row(row)]

	cpdef np.ndarray get_col_values(self, int col):
		return self.mat[col, self.valid_in_col(col)]

	cpdef rowmin(self, int row):
		return self.get_row_values(row).min()

	cpdef reducerow(self, int row):
		self.mat[row, self.valid_in_row(row)] -= self.rowmin(row)

	cpdef reducerows(self):
		for r in range(self.num_rows):
			self.reducerow(r)

	cpdef set_elem(self, flag, r, c):
		self.flags[flag].set_elem(r, c)


cdef np.ndarray get_steps(np.ndarray idxs, int size):
	# Given a sorted list of repeating indexes, returns the indices of every increment
	cdef np.ndarray out
	out = np.zeros(size+1, dtype=np.int)
	cdef int i, v
	v = -1

	for i in range(idxs.size):
		if idxs[i] > v:
			out[idxs[i]] = i
			v = idxs[i]

	# set last elem
	out[size] = idxs.size

	return out

cpdef HungarianMatrix from_matrix(np.ndarray mat):
	# Return a hungarian sparse matrix from a dense matrix (M, N), where invalid values are -1
	cdef np.ndarray loc, val

	rows, cols = np.nonzero(mat>=0)
	loc = np.zeros((rows.size, 2), dtype=np.int)
	val = np.zeros(rows.size, np.int)

	for i in range(rows.size):
		loc[i, 0] = rows[i]
		loc[i, 1] = cols[i]
		val[i] = mat[rows[i], cols[i]]

	return HungarianMatrix(loc, val)

