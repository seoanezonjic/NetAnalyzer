import sys 
import numpy as np
from scipy import linalg
from warnings import warn
import networkx as nx
from node2vec import Node2Vec
from NetAnalyzer.adv_mat_calc import Adv_mat_calc

class Graph2sim:

	allowed_embeddings = ['node2vec', 'deepwalk']
	allowed_kernels = ['el', 'ct', 'rf', 'me', 'vn', 'rl', 'ka', 'md']

	def get_embedding(graph, embedding, dimensions = 64, walk_length=30, num_walks = 200, p = 1, q = 1, workers = 1, window = 10, min_count=1, seed = None, quiet=False, batch_words=4):
		emb_coords = None
		if embedding in ['node2vec', 'deepwalk']: # TODO 'metapath2vec',
			if embedding == 'node2vec' or embedding == "deepwalk":
				if embedding == "deepwalk":
					p = 1
					q = 1 
				node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p = p, q = q, workers = workers, seed = seed, quiet=quiet)
				model = node2vec.fit(window=window, min_count= min_count, batch_words=batch_words) # batch_words=10000
				# min_count: is the minimun number of counts a word must have.
				# batch_words: are Target size (in words) for batches of examples
				list_arrays=[model.wv.get_vector(str(n)) for n in graph.nodes()]
				n_cols=list_arrays[0].shape[0] # Number of col
				n_rows=len(list_arrays)# Number of rows
				emb_coords = np.concatenate(list_arrays).reshape([n_rows,n_cols]) # Concat all the arrays at one.
		else:
			print('Warning: The embedding method was not specified or not exists.')                                                                                                      
			sys.exit(0)

		return emb_coords

	def emb_coords2kernel(emb_coords, normalization = False):
		kernel = emb_coords.dot(emb_coords.T)
		if normalization: kernel = Adv_mat_calc.cosine_normalization(kernel)
		return kernel

	def get_kernel(matrix, kernel, normalization=False):
		#I = identity matrix
		#D = Diagonal matrix
		#A = adjacency matrix
		#L = laplacian matrix = D − A
		matrix_result = None
		dimension_elements = np.shape(matrix)[1]
		# In scuba code, the diagonal values of A is set to 0. In weighted matrix the kernel result is the same with or without this operation. Maybe increases the computing performance?
		# In the md kernel this operation affects the values of the final kernel
		#dimension_elements.times do |n|
		#	matrix[n,n] = 0.0
		#end
		if kernel in ['el', 'ct', 'rf', 'me'] or 'vn' in kernel or 'rl' in kernel:
			diagonal_matrix = np.zeros((dimension_elements,dimension_elements))
			np.fill_diagonal(diagonal_matrix,matrix.sum(axis=1))	# get the total sum for each row, for this reason the sum method takes the 1 value. If sum colums is desired, use 0
													# Make a matrix whose diagonal is row_sum
			matrix_L = diagonal_matrix - matrix
			if kernel == 'el': #Exponential Laplacian diffusion kernel(active). F Fouss 2012 | doi: 10.1016/j.neunet.2012.03.001
				beta = 0.02
				beta_product = matrix_L * -beta
				#matrix_result = beta_product.expm
				matrix_result = linalg.expm(beta_product)
			elif kernel == 'ct': # Commute time kernel (active). J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
				matrix_result = np.linalg.pinv(matrix_L, hermitian=True) # Hermitian parameter added to ensure convergence, just for real symmetric matrixes.
				# Anibal saids that this kernel was normalized. Why?. Paper do not seem to describe this operation for ct, it describes for Kvn or for all kernels, it is not clear.
			elif kernel == 'rf': # Random forest kernel. J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
				matrix_result = np.linalg.inv(np.eye(dimension_elements) + matrix_L) #Krf = (I +L ) ^ −1
			elif 'vn' in kernel: # von Neumann diffusion kernel. J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
				alpha = float(kernel.replace('vn', '')) * max(np.linalg.eigvals(matrix)) ** -1  # alpha = impact_of_penalization (1, 0.5 or 0.1) * spectral radius of A. spectral radius of A = absolute value of max eigenvalue of A 
				# TODO: The expresion max(np.linalg.eigvals(matrix)) obtain the max eigen computing all but in ruby was used a power series to compute directly this value. Implement this
				matrix_result = np.linalg.inv(np.eye(dimension_elements) - matrix * alpha ) #  (I -alphaA ) ^ −1
			elif 'rl' in kernel: # Regularized Laplacian kernel matrix (active)
				alpha = float(kernel.replace('rl', '')) * max(np.linalg.eigvals(matrix)) ** -1  # alpha = impact_of_penalization (1, 0.5 or 0.1) * spectral radius of A. spectral radius of A = absolute value of max eigenvalue of A
				matrix_result = np.linalg.inv(np.eye(dimension_elements) + matrix_L * alpha ) #  (I + alphaL ) ^ −1
			elif kernel == 'me': # Markov exponential diffusion kernel (active). G Zampieri 2018 | doi.org/10.1186/s12859-018-2025-5 . Taken from compute_kernel script
				beta=0.04
				#(beta/N)*(N*I - D + A)
				id_mat = np.eye(dimension_elements)
				m_matrix = (id_mat * dimension_elements - diagonal_matrix + matrix ) * (beta/dimension_elements)
				#matrix_result = m_matrix.expm
				matrix_result = linalg.expm(m_matrix)
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
		else:
			matrix_result = matrix
			warn('Warning: The kernel method was not specified or not exists. The adjacency matrix will be given as result')
			# This allows process a previous kernel and perform the normalization in a separated step.
		if normalization: matrix_result = Adv_mat_calc.cosine_normalization(matrix_result)  #TODO: check implementation with Numo::array
		return matrix_result