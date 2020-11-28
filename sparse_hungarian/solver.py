from sparse_hungarian.matrix import SparseCostMatrix, _fast_remove
import numpy as np

class SparseHungarianSolver():
	def __init__(self, cost_mat):
		self.mat = SparseCostMatrix(cost_mat)
		self.last_primed = None  # memory for previous primed element

	def solve(self):
		step = self.step_1()
		while step:
			step = step()  # returns either a next function or run, or False

		return self.mat.get_sol()

	def step_1(self):
		"""Subtract smallest element from each row. Go to step 2"""
		self.mat.reduce_rows()
		self.mat.reduce_cols()
		return self.step_2

	def step_2(self):
		"""For each zero, if there is no zero in its row or column, add to starred.
		Go to step 3"""
		for row in range(self.mat._r):  # for each row
			if self.mat.zero_by_row[row].size == 1:  # if only one zero in this row
				col = self.mat.zero_by_row[row][0]  # col in which zero is
				if self.mat.zero_by_col[col].size == 1:  # if also only zero in column
					self.mat.star_zero(row, col)

		return self.step_3

	def step_3(self):
		"""'Cover' each column containing a starred zero.
		If <n_rows> columns are covered, DONE! return False
		else, Go to step 4"""
		self.mat.covered_columns = self.mat.starred_columns.copy()
		if self.mat.covered_columns.sum() == self.mat._r:
			return False
		else:
			return self.step_4

	def step_4(self):
		"""A) Find an uncovered zero and 'prime' it.
		B) If no starred zeros in this row, go to step 5
		C) Else, cover this row and uncover the column containing the starred zero.
		D) If there are still uncovered zeros, return to A
		E) Save the smallest uncovered value, v. Add to each covered row, subtract from each uncovered col. Return to A"""

		while True:
			# A) prime uncovered zero
			next_zero = self.mat.get_uncovered_zero()

			if next_zero:  # if one found
				r, c = next_zero
				self.mat.prime_zero(r, c)
				self.last_primed = (r, c)

				# B)
				starred_in_row = self.mat.starred_rows[r]  #  boolean if row is starred
				if not starred_in_row:  # If no starred zeros in this row
					return self.step_5

				# C) & D)
				star_c = self.mat.starred_by_row[r][0]  # get first starred zero in row
				self.mat.covered_rows[r] = 1  # Cover this row
				self.mat.covered_columns[star_c] = 0  # Uncover the column containing the starred zero

			# E)
			else:
				v = self.mat.get_min_uncovered()
				[self.mat.add_to(r, v, 'row') for r in np.nonzero(self.mat.covered_rows)[0]]  # add to covered rows
				[self.mat.add_to(c, -v, 'col') for c in np.nonzero(~self.mat.covered_columns)[0]]  # subtract from uncovered columns
				break

		return self.step_4

	def step_5(self):
		"""
		seq 1: Starting at last_primed:
		A) Try to get starred zero in column of last_primed. Set this to be last_starred
			i) if none found, go to Sequence 2
			ii) if found, go to B
		B) Get primed zero in row of last_starred. Go to A

		seq 2:
		A) Unstar each starred zero of the series
		B) Star each primed zero of the series
		C) Erase all primes
		D) Uncover all matrix
		E) Go to step 3
		"""
		last_primed = self.last_primed
		# SEQUENCE 1
		starred, primed = [], [last_primed]  # track each starred, primed zeros
		while True:
			# A)
			starred_in_col_rows = np.nonzero(self.mat.starred_columns[last_primed[1]])[0]

			# list of any starred zeros in column of last_primed, except for last_primed
			starred_zeros_in_column = _fast_remove(self.mat.starred_by_col[last_primed[1]], last_primed[0])
			if starred_in_col_rows.size == 0:  # i)
				break
			# ii)
			last_starred = (starred_zeros_in_column[0], last_primed[1])
			starred.append(last_starred)

			primed_in_row = self.mat.prime_by_row[last_starred[0]]
			assert primed_in_row.size == 1, "Unexpected primed_in_row_cols size."
			last_primed = (last_starred[0], primed_in_row[0])
			primed.append(last_primed)

		# SEQUENCE 2
		for r, c in starred:  # A)
			self.mat.unstar_zero(r, c)
		for r, c in primed:   # B)
			self.mat.star_zero(r, c)
		# C)
		self.mat.clear('prime')
		# D
		self.mat.clear('cover')
		return self.step_3

