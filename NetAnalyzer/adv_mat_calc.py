import sys 
import numpy as np
from scipy import linalg
import scipy.stats as stats
from scipy.spatial import distance_matrix
import umap
from warnings import warn
class Adv_mat_calc:

	@staticmethod
	def get_kernel(matrix, node_names, kernel, normalization=False):
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

	@staticmethod
	def cosine_normalization(matrix):
		dims = np.shape(matrix)
		normalized_matrix =  np.zeros(dims)
		for i in range(0, dims[0]):
			for j in range(0, dims[1]):
				norm = matrix[i, j]/np.sqrt(matrix[i, i] * matrix[j,j])
				normalized_matrix[i, j] = norm
		return normalized_matrix

	@staticmethod
	def coords2sim(coords, sim = "dotProduct"):
	    if sim == "dotProduct":
	        sim_mat = coords.dot(coords.T)
	    else:
	        sim_mat = Adv_mat_calc.coords2dis(coords, dist = sim)
	        sim_mat = 1/(sim_mat+1)
	    return sim_mat

	@staticmethod
	def coords2dis(coords, dist = "euclidean"):
	    if "euclidean": dist = 2
	    dist = distance_matrix(coords, coords, p = dist)
	    return dist

	@staticmethod
	def data2umap(data, n_neighbors = 15, min_dist = 0.1, n_components = 2, metric = 'euclidean', random_seed = None):
	    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state= random_seed)
	    umap_coords = reducer.fit_transform(data)
	    return umap_coords

	# Alaimo 2014, doi: 10.3389/fbioe.2014.00071
	@staticmethod
	def tranference_resources(matrix1, matrix2, lambda_value1 = 0.5, lambda_value2 = 0.5):
		# TODO (Fede,19/12/22) An extension to n layers would be possible with an iterative process.
		m1rowNumber, m1colNumber = matrix1.shape
		m2rowNumber, m2colNumber = matrix2.shape
		matrix1Weight = Adv_mat_calc.graphWeights(m1colNumber, m1rowNumber, matrix1.T, lambda_value1)
		matrix2Weight = Adv_mat_calc.graphWeights(m2colNumber, m2rowNumber, matrix2.T, lambda_value2)
		matrixWeightProduct = np.dot(matrix1Weight, np.dot(matrix2, matrix2Weight))
		finalMatrix = np.dot(matrix1, matrixWeightProduct)
		return finalMatrix

	@staticmethod
	def graphWeights(rowsNumber, colsNumber, inputMatrix, lambdaValue = 0.5):
	 	ky = np.diag((1.0 / inputMatrix.sum(0))) #sum cols
	 	weigth = np.dot(inputMatrix, ky).T
	 	weigth[np.isnan(weigth)] = 0 # if there is no neighbors, there is no weight
	 	ky = None #free memory
	 	weigth = np.dot(inputMatrix, weigth)

	 	kx = inputMatrix.sum(1) #sum rows
	 	
	 	kx_lamb = kx ** lambdaValue
	 	kx_lamb_mat = np.zeros((rowsNumber, rowsNumber))
	 	for j in range(0,rowsNumber):
	 		for i in range(0,rowsNumber):
	 			kx_lamb_mat[j,i] = kx_lamb[i]
	 	kx_lamb = None #free memory

	 	kx_inv_lamb = kx ** (1 - lambdaValue)
	 	kx_inv_lamb_mat = np.zeros((rowsNumber, rowsNumber))
	 	for j in range(0,rowsNumber):
	 		for i in range(0,rowsNumber):
	 			kx_inv_lamb_mat[j,i] = kx_inv_lamb[i]
	 	kx_inv_lamb = None #free memory

	 	nx = 1.0/(kx_lamb_mat * kx_inv_lamb_mat) # inplace marks a matrix to be used by reference, not for value
	 	nx[nx == np.inf] = 0 # if there is no neighbors, there is no weight
	 	kx_lamb_mat = None #free memory
	 	kx_inv_lamb_mat = None #free memory
	 	weigth = weigth * nx
	 	return weigth


	@staticmethod
	def zscore_normalize(data):
	    data = np.array(data)
	    data = (data - np.mean(data)) / np.std(data)
	    return data

	@staticmethod
	def pearsonr(x=None,y=None, alternative='two-sided', method=None):
	    if y is not None:
	        corr_obj = stats.pearsonr(x, y, alternative=alternative, method= method)
	        rs = corr_obj.correlation
	        prob = corr_obj.pvalue
	    else:
	        if alternative == 'two-sided':
	            p = lambda t,n: 2 * (1 - stats.t.cdf(abs(t), n - 2))
	        elif alternative == 'greater':
	            p = lambda t,n: 1 - stats.t.cdf(t, n - 2)
	        elif alternative == 'less':
	            p = lambda t,n: stats.t.cdf(t, n - 2)
	        else:
	            raise ValueError("Invalid alternative argument. Valid options are 'two-sided', 'greater', or 'less'.")
    
	        N = np.ones((x.shape[0], x.shape[0]))*x.shape[1] # This should be adopted to na cases.
	        R = np.corrcoef(x)
	        T = (R*np.sqrt(N-2))/(np.sqrt(1-R**2))
	        P = 2*(1-stats.t.cdf(abs(T), N - 2))
	    return R, P

	@staticmethod
	def get_stats_from_list(data): #TODO: think inj a dry version
		data = np.array(data).reshape((1,len(data)))
		primary_stats = Adv_mat_calc.get_primary_stats(data)
		stats = []
		stats.append(['Elements', primary_stats["count"]])
		stats.append(['Elements Non Zero', primary_stats["countNonZero"]])
		stats.append(['Non Zero Density', primary_stats["countNonZero"]/primary_stats["count"]])
		stats.append(['Max', primary_stats["max"]])
		stats.append(['Min', primary_stats["min"]])
		stats.append(['Average', primary_stats["average"]])
		stats.append(['Variance', primary_stats["variance"]])
		stats.append(['Standard Deviation', primary_stats["standardDeviation"]])
		stats.append(['Q1', primary_stats["q1"]])
		stats.append(['Median', primary_stats["median"]])
		stats.append(['Q3', primary_stats["q3"]])
		stats.append(['Min Non Zero', primary_stats["minNonZero"]])
		stats.append(['Average Non Zero', primary_stats["averageNonZero"]])
		stats.append(['Variance Non Zero', primary_stats["varianceNonZero"]])
		stats.append(['Standard Deviation Non Zero', primary_stats["standardDeviationNonZero"]])
		stats.append(['Q1 Non Zero', primary_stats["q1NonZero"]])
		stats.append(['Median Non Zero', primary_stats["medianNonZero"]])
		stats.append(['Q3 Non Zero', primary_stats["q3NonZero"]])
		stats = map(lambda x: [x[0],str(x[1])],stats)
		return stats


	@staticmethod
	def get_stats_from_matrix(matrix): 
		stats = []
		primary_stats = Adv_mat_calc.get_primary_stats(matrix)
		#stats << ['Matrix - Symmetric?', matrix.symmetric?]
		stats.append(['Matrix - Dimensions', 'x'.join(map(str, matrix.shape))])
		stats.append(['Matrix - Elements', primary_stats["count"]])
		stats.append(['Matrix - Elements Non Zero', primary_stats["countNonZero"]])
		stats.append(['Matrix - Non Zero Density', primary_stats["countNonZero"]/primary_stats["count"]])
		stats.append(['Weigth - Max', primary_stats["max"]])
		stats.append(['Weigth - Min', primary_stats["min"]])
		stats.append(['Weigth - Average', primary_stats["average"]])
		stats.append(['Weigth - Variance', primary_stats["variance"]])
		stats.append(['Weigth - Standard Deviation', primary_stats["standardDeviation"]])
		stats.append(['Weigth - Q1', primary_stats["q1"]])
		stats.append(['Weigth - Median', primary_stats["median"]])
		stats.append(['Weigth - Q3', primary_stats["q3"]])
		stats.append(['Weigth - Min Non Zero', primary_stats["minNonZero"]])
		stats.append(['Weigth - Average Non Zero', primary_stats["averageNonZero"]])
		stats.append(['Weigth - Variance Non Zero', primary_stats["varianceNonZero"]])
		stats.append(['Weigth - Standard Deviation Non Zero', primary_stats["standardDeviationNonZero"]])
		stats.append(['Weigth - Q1 Non Zero', primary_stats["q1NonZero"]])
		stats.append(['Weigth - Median Non Zero', primary_stats["medianNonZero"]])
		stats.append(['Weigth - Q3 Non Zero', primary_stats["q3NonZero"]])
		connections = Adv_mat_calc.get_connection_number(matrix)
		connection_stats = Adv_mat_calc.get_primary_stats(connections)
		stats.append(['Node - Elements', connection_stats["count"]])
		stats.append(['Node - Elements Non Zero', connection_stats["countNonZero"]])
		stats.append(['Node - Non Zero Density', connection_stats["countNonZero"]/connection_stats["count"]])
		stats.append(['Edges - Max', connection_stats["max"]])
		stats.append(['Edges - Min', connection_stats["min"]])
		stats.append(['Edges - Average', connection_stats["average"]])
		stats.append(['Edges - Variance', connection_stats["variance"]])
		stats.append(['Edges - Standard Deviation', connection_stats["standardDeviation"]])
		stats.append(['Edges - Q1', connection_stats["q1"]])
		stats.append(['Edges - Median', connection_stats["median"]])
		stats.append(['Edges - Q3', connection_stats["q3"]])
		stats.append(['Edges - Min Non Zero', connection_stats["minNonZero"]])
		stats.append(['Edges - Average Non Zero', connection_stats["averageNonZero"]])
		stats.append(['Edges - Variance Non Zero', connection_stats["varianceNonZero"]])
		stats.append(['Edges - Standard Deviation Non Zero', connection_stats["standardDeviationNonZero"]])
		stats.append(['Edges - Q1 Non Zero', connection_stats["q1NonZero"]])
		stats.append(['Edges - Median Non Zero', connection_stats["medianNonZero"]])
		stats.append(['Edges - Q3 Non Zero', connection_stats["q3NonZero"]])
    
		stats = map(lambda x: [x[0],str(x[1])],stats)
    
		return stats

	@staticmethod
	def get_primary_stats(matrix):
	    stats = {}
	    max = matrix[0, 0] # Initialize max value
	    min = matrix[0, 0] # Initialize min value
	    min_non_zero = matrix[0, 0] # Initialize min value
	    values = matrix.flatten()
	
	    stats["count"] = 0
	    stats["countNonZero"] = 0 
	    stats["sum"] = 0
	    for value in values:
	        stats["count"] += 1
	        stats["countNonZero"] += 1 if value != 0 else 0
	        stats["sum"] += value
	        max = value if value > max else max
	        min = value if value < min else min
	        if value != 0 and value < min:
	            min_non_zero = value
	    
	    stats["max"] = max
	    stats["min"] = min
	    stats["minNonZero"] = min_non_zero
	
	    quartile_stats = Adv_mat_calc.get_quartiles(values, stats["count"])
	    stats.update(quartile_stats)
	    non_zero_values = [v for v in values if v != 0]
	    quartile_stats_non_zero = Adv_mat_calc.get_quartiles(non_zero_values, stats["countNonZero"])
	    stats.update(Adv_mat_calc.transform_keys(quartile_stats_non_zero, lambda x: x + "NonZero"))
	    Adv_mat_calc.get_composed_stats(stats, values)
	    return stats

	@staticmethod
	def get_connection_number(matrix):
	    rows, cols = matrix.shape
	    connections = np.zeros((1, cols))
	    for i in range(cols):
	        column = matrix[:, i] 
	        count = 0
	        for value in column:
	            count += 1 if value != 0 else 0
	
	        connections[0, i] = count - 1 # the connection with self is removed
	
	    return connections

	@staticmethod
	def get_quartiles(values, n_items):
	    stats = {}
	    stats['q1'] = np.percentile(values,25)
	    stats['median'] = np.percentile(values,50)
	    stats['q3'] = np.percentile(values,75)
	    return stats

	@staticmethod
	def get_composed_stats(stats, values):
	    average = stats["sum"]/stats["count"]
	    average_non_zero = stats["sum"]/stats["countNonZero"]
	    stats["average"] = average
	    stats["averageNonZero"] = average_non_zero
	
	    stats["sumDevs"] = 0
	    stats["sumDevsNonZero"] = 0
	    for value in values:
	        stats["sumDevs"] += (value - average) ** 2
	        stats["sumDevsNonZero"] += (value - average_non_zero) ** 2 if value != 0 else 0
	
	    stats["variance"] = stats["sumDevs"]/stats["count"]
	    stats["varianceNonZero"] = stats["sumDevsNonZero"]/stats["countNonZero"]
	    stats["standardDeviation"] = stats["variance"] ** 0.5
	    stats["standardDeviationNonZero"] = stats["varianceNonZero"] ** 0.5

	@staticmethod
	def binarize_mat(matrix):
	    matrix = matrix > 0
	    matrix = matrix.astype(float)
	    return matrix

	@staticmethod
	def filter_cutoff_mat(matrix, cutoff):
	    filtered_mat = matrix >= cutoff
	    mat_result = matrix * filtered_mat
	    return mat_result

	@staticmethod
	def percentile_filter(matrix, percentile_threshold=90):
	    # TODO: Is necessary to obtain percentiles just form non zero data
	    percentiles = np.percentile(matrix, percentile_threshold, axis=1)
	    result_mat = matrix >= percentiles[:, np.newaxis]
	    new_adj = result_mat.transpose()*result_mat
	    mat1[~new_adj] = 0
	    return mat1


	@staticmethod
	def disparity_filter_mat(matrix, rowIds, colIds, pval_threshold = 0.05):
	    pval_mat = Adv_mat_calc.get_disparity_backbone_pval(matrix)
	    # Create edge list from that p value matrix
	    result_mat = pval_mat < pval_threshold
	    # adjacency matrix, obtained when both p[i,j] and p[j,i] match the criteria
	    new_adj = result_mat.transpose()*result_mat

	    # remove genes with no significance
	    k = np.sum(new_adj, axis=0)

	    final_adj_mat = matrix[:,k>0]
	    final_adj_mat = final_adj_mat[k>0,:]
	    final_rowIds = [node_id for node_id, is_good in zip(rowIds, list(k>0)) if is_good]
	    final_colIds = [node_id for node_id, is_good in zip(colIds, list(k>0)) if is_good]

	    return final_adj_mat, final_rowIds, final_colIds

	@staticmethod
	def remove_zero_lines(matrix, rowIds, colIds, symmetric=False):
		# TODO: Need test.
		# TODO: semgent thsi in two functions for rows and columns.
	    rows2delete = np.all(matrix == 0, axis=1)
	    if symmetric:
	        cols2delete = rows2delete
	    else:
	        cols2delete = np.all(matrix == 0, axis=0)

	    matrix = matrix[~rows2delete]
	    matrix = matrix[:,~cols2delete]
	    if rowIds is not None: rowIds = [rowIds[i] for i in range(0,len(rowIds)) if not rows2delete[i]]
	    if colIds is not None: colIds = [colIds[i] for i in range(0,len(colIds)) if not cols2delete[i]]

	    return matrix, rowIds, colIds

	@staticmethod
	def get_disparity_backbone_pval(matrix):
	    # by the moment, implementetion square (?)
	    # TODO: Add a warning when not square matrix.
	    pval_mat = matrix.copy()
	    W = np.sum(pval_mat, axis=0)
	    k = (pval_mat > 0).sum(0)
	    # operacion vectorizada.
	    for i in range(0,pval_mat.shape[1]):
	        pval_mat[:,i] = (1-(matrix[:,i]/W[i]))**(k[i]-1)
	    return pval_mat

	@staticmethod
	def transform_keys(hash, function):
	    new_hash = {}
	    for key, val in hash.items():
		    new_key = function(key)
		    new_hash[new_key] = val
	    return new_hash