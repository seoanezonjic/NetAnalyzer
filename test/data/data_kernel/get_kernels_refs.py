#!/usr/bin/env python
import argparse
import numpy as np
import scipy.linalg as LA
import networkx as nx
import os

def get_kernel(matrix, node_names, kernel, normalization=False):
		#I = identity matrix
		#D = Diagonal matrix
		#A = adjacency matrix
		#L = laplacian matrix = D − A
		matrix_result = None
		dimension_elements = np.shape(matrix)[1]
		if kernel in ['el', 'ct', 'rf', 'me'] or 'vn' in kernel or 'rl' in kernel:
			diagonal_matrix = np.zeros((dimension_elements,dimension_elements))
			np.fill_diagonal(diagonal_matrix,matrix.sum(axis=1))	# get the total sum for each row, for this reason the sum method takes the 1 value. If sum colums is desired, use 0
													# Make a matrix whose diagonal is row_sum 	
			matrix_L = diagonal_matrix - matrix
			if kernel == 'el':
				beta = 0.02
				beta_product = matrix_L * -beta
				matrix_result = LA.expm(beta_product)
			elif kernel == 'ct': 
				matrix_result = np.linalg.pinv(matrix_L, hermitian=True) 
			elif kernel == 'rf': 
				matrix_result = np.linalg.inv(np.eye(dimension_elements) + matrix_L) #Krf = (I +L ) ^ −1
			elif 'vn' in kernel:
				alpha = float(kernel.replace('vn', '')) * max(np.linalg.eigvals(matrix)) ** -1 
				matrix_result = np.linalg.inv(np.eye(dimension_elements) - matrix * alpha )
			elif 'rl' in kernel: # Regularized Laplacian kernel matrix (active)
				alpha = float(kernel.replace('rl', '')) * max(np.linalg.eigvals(matrix)) ** -1  # alpha = impact_of_penalization (1, 0.5 or 0.1) * spectral radius of A. spectral radius of A = absolute value of max eigenvalue of A
				matrix_result = np.linalg.inv(np.eye(dimension_elements) + matrix_L * alpha ) #  (I + alphaL ) ^ −1
			elif kernel == 'me': # Markov exponential diffusion kernel (active). G Zampieri 2018 | doi.org/10.1186/s12859-018-2025-5 . Taken from compute_kernel script
				beta=0.04
				id_mat = np.eye(dimension_elements)
				m_matrix = (id_mat * dimension_elements - diagonal_matrix + matrix ) * (beta/dimension_elements)
				matrix_result = LA.expm(m_matrix)
		elif kernel == 'ka': # Kernelized adjacency matrix (active). J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
			lambda_value = min(np.linalg.eigvals(matrix)) # TODO implent as power series as shown in ruby equivalent
			matrix_result = matrix + np.eye(dimension_elements) * abs(lambda_value) # Ka = A + lambda*I # lambda = the absolute value of the smallest eigenvalue of A
		elif 'md' in kernel: # Markov diffusion kernel matrix. G Zampieri 2018 | doi.org/10.1186/s12859-018-2025-5 . Taken from compute_kernel script
			t = int(kernel.replace('md', ''))
			#TODO: check implementation
			col_sum = matrix.sum(axis=1)
			p_mat = np.divide(matrix.T,col_sum).T
			p_temp_mat = p_mat.copy()
			zt_mat = p_mat.copy()
			for i in range(0, t-1):
				p_temp_mat = np.dot(p_temp_mat,p_mat)
				zt_mat = zt_mat + p_temp_mat
			zt_mat = zt_mat * (1.0/t)
			matrix_result = np.dot(zt_mat, zt_mat.T)

		if normalization: matrix_result = cosine_normalization(matrix_result) 
		
		return matrix_result

def cosine_normalization(matrix):
		dims = np.shape(matrix)
		normalized_matrix =  np.zeros(dims)
		for i in range(0, dims[0] - 1):
			for j in range(0, dims[1] - 1):
				norm = matrix[i, j]/np.sqrt(matrix[i, i] * matrix[j,j])
				normalized_matrix[i, j] = norm
		return normalized_matrix

def file2nodes(file_name):
	with open(file_name) as file:
		nodes = [line.rstrip() for line in file]
	return nodes


######################## MAIN ########################
######################################################
# Load the numpy matrix
M = np.load("adj_mat.npy")
node_names = file2nodes("./adj_mat.lst")

kernelid2matrix = {}
for kernel_type in ['el', 'ct', 'rf', 'me','vn0.5','rl0.5','md1']:
	kernel = get_kernel(M, node_names, kernel_type)
	kernel_type = kernel_type.replace('.', '_')
	np.save(kernel_type,kernel)

normalized_kernel = get_kernel(M, node_names, "ka", normalization = True)
np.save("ka_normalized",normalized_kernel)



