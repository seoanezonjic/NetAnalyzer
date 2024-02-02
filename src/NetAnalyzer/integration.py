import sys
import os
import glob
import numpy as np
import concurrent.futures
import itertools

class Kernels:

	def __init__(self): 
		self.kernels_raw = [] # [Mat1, Mat2, Mat3,...]
		self.local_indexes = [] # [{Node1 => idx1, Node2=> idx2},{Node1 => idx4, Node2 => idx2}]
		self.general_nodes = [] # list of all nodes
		self.kernels_position_index = {} # { Node1 => [idx1, idx2, None], Node2 => ...}
		self.integrated_kernel = [] # [integrated_matrix, list_of_nodes]

	def move2zero_reference(self):
		moved_kernels = []
		for kernel in self.kernels_raw:
			min_kernel = np.min(kernel)
			if min_kernel < 0:
				kernel -= min_kernel
			moved_kernels.append(kernel)
		self.kernels_raw = moved_kernels
		

	def load_kernels_by_bin_matrixes(self, input_matrix, input_nodes, kernels_names):
		for pos, kernel_name in enumerate(kernels_names):
			self.kernels_raw.append(np.load(input_matrix[pos]))
			self.local_indexes.append(self.build_matrix_index(self.lst2arr(input_nodes[pos])))

	def create_general_index(self):
		self.general_nodes = []
		for index in self.local_indexes:
			self.general_nodes += index.keys()
		self.general_nodes = sorted(list(set(self.general_nodes))) # Uniq elements and sorted to remove permutated matrixes.
		
		for node in self.general_nodes:
			self.kernels_position_index[node] = [ind.get(node) for ind in self.local_indexes]
		self.local_indexes = [] # Removing not needed local indexes

	def integrate(self, method, n_workers = 8, symmetry = True, n_partition_axis = None): 
		general_nodes = self.general_nodes.copy()
		nodes_dimension = len(general_nodes)
		general_kernel = np.zeros((nodes_dimension,nodes_dimension))
		n_kernel = len(self.kernels_raw)
		if n_partition_axis == None: n_partition_axis = int(np.trunc(np.sqrt(n_workers))) # Default value to use the corect number of blocks for n_workers

		# Filling the argument section
		splitted_general_nodes = list(self.split(general_nodes, n_partition_axis))
		if symmetry:
			pair_nodes = list(itertools.combinations_with_replacement(splitted_general_nodes, 2))
		else:
			pair_nodes = list(itertools.product(splitted_general_nodes, repeat = 2))
		process_number = len(pair_nodes)

 		# Calling the multiprocessing
		with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:

			results = [executor.submit(self.build_matrix_block, pair_nodes[i][0],
				pair_nodes[i][1], method, n_kernel, general_nodes, splitted_general_nodes, symmetry) for i in range(process_number)]

			for f in concurrent.futures.as_completed(results):
				row_start, row_end, col_start, col_end, block_matrix = f.result()
				general_kernel[row_start:row_end, col_start:col_end] = block_matrix
				if row_start != col_start and symmetry:
					general_kernel[col_start:col_end, row_start:row_end] = block_matrix.transpose()

		self.integrated_kernel = [general_kernel, self.general_nodes]

	
	def build_matrix_block(self, row_nodes, col_nodes, method, n_kernel, general_nodes, splitted_general_nodes, symmetry):
		general_block_matrix = np.zeros((len(row_nodes),len(col_nodes))) # TODO: Add option to considerar matrixes (maybe the min.

		row_start = general_nodes.index(row_nodes[0])
		row_end = row_start + len(row_nodes)
		col_start = general_nodes.index(col_nodes[0])
		col_end = col_start + len(col_nodes)

		if symmetry and row_start == col_start:
			# Filling main diagonal blocks with upper triang.
			nodes_dimension = len(row_nodes)
			i = 0
			while len(row_nodes) > 0:
				node_A = row_nodes[-1]
				ind = len(row_nodes) - 1
				for node_B in reversed(row_nodes):
					j = ind
					values = self.get_values(node_A, node_B)
					if values:
						reversed_i = nodes_dimension -1 - i
						general_block_matrix[reversed_i, j] = method(values, n_kernel)
						general_block_matrix[j, reversed_i] = general_block_matrix[reversed_i, j]
					ind -= 1
				row_nodes.pop()
				i += 1
		else:
			# Filling all vs all
			for i, node_A in enumerate(row_nodes):
				for j, node_B in enumerate(col_nodes):
					values = self.get_values(node_A, node_B)
					if values: 
						general_block_matrix[i, j] = method(values, n_kernel)


		return row_start, row_end, col_start, col_end, general_block_matrix

	def get_values(self, node_A, node_B):
		rows = self.kernels_position_index[node_A]
		cols = self.kernels_position_index[node_B]
		values = []
		for i, r_ind in enumerate(rows): #Load just the pairs in both sides of the kernel matrix 
			if r_ind is not None: 
				c_ind = cols[i] # Maybe a get is needed watch out!!!
				if c_ind is not None:
					values.append(self.kernels_raw[i][r_ind, c_ind])
		return values

	def mean(self, values, n_kernel):
		return sum(values)/n_kernel

	def mean_by_presence(self, values, n_kernel):
		return sum(values)/len(values)

	def median(self, values, n_kernel):
		return np.median(values)

	def max(self, values, n_kernel):
		return max(values)

	def geometric_mean(self, values, n_kernel):
		# TODO: Talk about the possibility of improving execution time with a non-log formula.
		# log to avoid overflows
		return np.exp(np.log(values).mean())

	def integrate_matrix(self, method, n_workers = 8, symmetry = True):
		if method == "mean":
			self.integrate(method = self.mean, n_workers = n_workers, symmetry = symmetry)
		elif method == "integration_mean_by_presence":
			self.integrate(method = self.mean_by_presence, n_workers = n_workers, symmetry = symmetry)
		elif method == "median":
			self.integrate(method = self.median, n_workers = n_workers, symmetry = symmetry)
		elif method == "max":
			self.integrate(method = self.max, n_workers = n_workers, symmetry = symmetry)
		elif method == "geometric_mean":
			self.integrate(method = self.geometric_mean, n_workers = n_workers, symmetry = symmetry)

	## AUXILIAR METHODS
	##############################

	def split(self, a, n): # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
		k, m = divmod(len(a), n)
		return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

	def lst2arr(self,lst_file):
		nodes = []
		with open(lst_file, "r") as file:
			for line in file:
				nodes.append(line.rstrip())
		return nodes

	def build_matrix_index(self, node_list):
		return {node: i for i, node in enumerate(node_list)}