from sparse_hungarian.matrix import SparseCostMatrix, _fast_remove
import numpy as np
from time import perf_counter

class Timer:
	"""For logging & improving speed of solver"""
	def __init__(self, cats):
		self.log = {c: {} for c in cats}
		self.time = {c: perf_counter() for c in cats}

	def add(self, cat, name):
		assert cat in self.log, f"Timer category {cat} not found."
		new_t = perf_counter()
		self.log[cat][name] = self.log[cat].get(name, []) + [new_t-self.time[cat]]
		self.time[cat] = perf_counter()

	def print(self):
		for c, cat_log in self.log.items():
			print(f"----- {c} -----")
			for k, v in cat_log.items():
				print(f"{k}: {len(v)} @ {np.mean(v) * 1000:.3f}ms = {np.sum(v):.3f}s")


class SparseHungarianSolver():
	def __init__(self, cost_mat, debug=False):
		# TODO: only solves N,M where M >= N. Therefore, pre-transpose if M < N and retranspose after solving

		self.mat = SparseCostMatrix(cost_mat)
		self.flags = self.mat.flags
		self.last_primed = None  # memory for previous primed element

		self.debug = debug
		if debug:
			self.timer = Timer(['steps', 's4', 's3', 's5'])

	def solve(self):
		step = self.step_1()
		if self.debug:
			self.timer.add('steps', 'setup')

		while step:
			fname = step.__name__
			step = step()  # returns either a next function or run, or False

			if self.debug:
				self.timer.add('steps', fname)

		if self.debug:
			self.timer.print()

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
			zeros_in_row = self.flags.lookup('zero', row, 'row')
			if zeros_in_row.size == 1:  # if only one zero in this row
				col = zeros_in_row[0]  # col in which zero is
				if self.flags.lookup('zero', col, 'col').size == 1:  # if also only zero in column
					self.flags.set_elem('starred', row, col)  # star zero

		return self.step_3

	def step_3(self):
		"""'Cover' each column containing a starred zero.
		If <n_rows> columns are covered, DONE! return False
		else, Go to step 4"""

		self.mat.bulk_cover(np.nonzero(self.flags.get_cols('starred'))[0], 'col')

		if self.flags.get_cols('covered').sum() == self.mat._r:
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
				self.flags.set_elem('prime', r, c)
				self.last_primed = (r, c)

				# B)
				starred_in_row = self.flags.get_rows('starred')[r]  #  boolean if row is starred
				if not starred_in_row:  # If no starred zeros in this row
					return self.step_5

				# C) & D)
				star_c = self.flags.lookup('starred', r, 'row')[0]  # get first starred zero in row
				self.mat.cover(r, 'row')  # Cover this row
				self.mat.uncover(star_c, 'col')  # Uncover the column containing the starred zero

			# E)
			else:
				v = self.mat.get_min_uncovered()
				[self.mat.add_to(r, v, 'row') for r in np.nonzero(self.flags.get_rows('covered'))[0]]  # add to covered rows
				[self.mat.add_to(c, -v, 'col') for c in np.nonzero(~self.flags.get_cols('covered'))[0]]  # subtract from uncovered columns
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
			# list of any starred zeros in column of last_primed, except for last_primed
			starred_zeros_in_column = self.flags.lookup('starred', last_primed[1], 'col')

			if starred_zeros_in_column.size == 0:  # i)
				break
			# ii)
			last_starred = (starred_zeros_in_column[0], last_primed[1])
			starred.append(last_starred)
			primed_in_row = self.flags.lookup('prime', last_starred[0], 'row')
			assert primed_in_row.size == 1, "Unexpected primed_in_row_cols size."
			last_primed = (last_starred[0], primed_in_row[0])
			primed.append(last_primed)

		# SEQUENCE 2
		for r, c in starred:  # A)
			self.flags.clear_elem('starred', r, c)  # unstar element
		for r, c in primed:   # B)
			self.flags.set_elem('starred', r, c)  # star element
		# C)
		self.flags.clear('prime')
		# D
		self.mat.uncover_all()
		return self.step_3

