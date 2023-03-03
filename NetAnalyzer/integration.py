import sys
import os
import glob
import numpy as np


class Kernels:

	def __init__(self): 
		self.kernels_raw = [] # [Mat1, Mat2, Mat3,...]
		self.local_indexes = [] # [{Node1 => idx1, Node2=> idx2},{Node1 => idx4, Node2 => idx2}]
		self.general_nodes = [] # list of all nodes
		self.kernels_position_index = {} # { Node1 => [idx1, idx2, None], Node2 => ...}
		self.integrated_kernel = [] # [integrated_matrix, list_of_nodes]

	def load_kernels_by_bin_matrixes(self, input_matrix, input_nodes, kernels_names):
		for pos, kernel_name in enumerate(kernels_names):
			self.kernels_raw.append(np.load(input_matrix[pos]))
			self.local_indexes.append(self.build_matrix_index(self.lst2arr(input_nodes[pos])))

	def create_general_index(self):
		self.general_nodes = []
		for index in self.local_indexes:
			self.general_nodes += index.keys()
		self.general_nodes = list(set(self.general_nodes)) # Uniq elements
		
		for node in self.general_nodes:
			self.kernels_position_index[node] = [ind.get(node) for ind in self.local_indexes]
		self.local_indexes = [] # Removing not needed local indexes

	def integrate(self, method):
		general_nodes = self.general_nodes.copy()
		nodes_dimension = len(general_nodes)
		
		general_kernel = np.zeros((nodes_dimension,nodes_dimension))
		n_kernel = len(self.kernels_raw)
		i = 0
		while len(general_nodes) > 1:
			node_A = general_nodes.pop()
			ind = len(general_nodes) - 1

			for node_B in reversed(general_nodes):
				#x = nodes_dimension - i
				#print x 
				j = ind
				values = self.get_values(node_A, node_B)
				if values:
					result = method(values, n_kernel) 
					reversed_i = nodes_dimension -1 - i
					general_kernel[reversed_i, j] = result
					general_kernel[j, reversed_i] = result
				ind -= 1

			i += 1

		self.integrated_kernel = [general_kernel, self.general_nodes]

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

	def integrate_matrix(self, method):
		if method == "mean":
			self.integrate(method = lambda values, n_kernel: sum(values)/n_kernel)
		elif method == "integration_mean_by_presence":
			self.integrate(method = lambda values, n_kernel: np.mean(values))

	## AUXILIAR METHODS
	##############################

	def lst2arr(self,lst_file):
		nodes = []
		with open(lst_file, "r") as file:
			for line in file:
				nodes.append(line.rstrip())
		return nodes

	def build_matrix_index(self, node_list):
		hash_nodes = {}
		for i, node in enumerate(node_list):
			hash_nodes[node] = i
		return hash_nodes







