import sys
import os
import glob
import numpy as np


class Kernels:

	def __init__(self): # TODO: Watch out with the new initilazers.
		self.Kernels_raw =  []
		self.integrated_kernel = None 
		self.general_nodes = []

		self.kernels_raw = []
		self.local_indexes = []
		self.integrated_kernel = []
		self.general_nodes = []
		self.kernels_position_index = {}

	def load_kernels_by_bin_matrixes(self, input_matrix, input_nodes, kernels_names):
		for pos, kernel_name in enumerate(kernels_names):
			self.kernels_raw.append(np.load(input_matrix))
			self.local_indexes.append(self.build_matrix_index(self.lst2arr(input_nodes[pos])))

	def create_general_index(self):
		self.general_nodes = []

		for index in self.local_indexes:
			self.general_nodes += index.keys
	
		self.general_nodes = list(set(self.general_nodes))
		
		for node in self.general_nodes:
			self.kernels_position_index[node] = [ind[node] for ind in self.local_indexes]

		self.local_indexes = []

	def integrate(self):
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
					result = yield(values, n_kernel) # Mirar para incorporar el yield
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
		integrate do |values, n_kernel|
       if method == "mean" 
			 	values.sum.fdiv(n_kernel)
       elsif method == "integration_mean_by_presence"
       	values.mean

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







