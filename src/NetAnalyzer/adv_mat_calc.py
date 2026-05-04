import warnings

import numpy as np
class Adv_mat_calc:

	# Alaimo 2014, doi: 10.3389/fbioe.2014.00071
	@staticmethod
	def tranference_resources(matrix1, matrix2, lambda_value1 = 0.5, lambda_value2 = 0.5): #2exp?
		# TODO (Fede,19/12/22) An extension to n layers would be possible with an iterative process.
		m1rowNumber, m1colNumber = matrix1.shape
		m2rowNumber, m2colNumber = matrix2.shape
		matrix1Weight = Adv_mat_calc.graphWeights(m1colNumber, m1rowNumber, matrix1.T, lambda_value1)
		matrix2Weight = Adv_mat_calc.graphWeights(m2colNumber, m2rowNumber, matrix2.T, lambda_value2)
		matrixWeightProduct = np.dot(matrix1Weight, np.dot(matrix2, matrix2Weight))
		finalMatrix = np.dot(matrix1, matrixWeightProduct)
		return finalMatrix

	@staticmethod
	def graphWeights(rowsNumber, colsNumber, inputMatrix, lambdaValue = 0.5): #2exp?
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
	def disparity_filter_mat(matrix, rowIds, colIds, pval_threshold = 0.05): #2exp?
		pval_mat = Adv_mat_calc.get_disparity_backbone_pval(matrix)
		result_mat = pval_mat < pval_threshold
		new_adj = result_mat.transpose() + result_mat # adjacency matrix, obtained when p[i,j] OR p[j,i] match the criteria
		matrix[~new_adj] = 0

		# remove nodes with no significance
		k = np.sum(result_mat, axis=1)
		final_adj_mat = matrix[:,k>0]
		final_adj_mat = final_adj_mat[k>0,:]
		final_rowIds = [node_id for node_id, is_good in zip(rowIds, list(k>0)) if is_good]
		final_colIds = [node_id for node_id, is_good in zip(colIds, list(k>0)) if is_good]

		return final_adj_mat, final_rowIds, final_colIds

	@staticmethod
	def filter_rowcols_by_whitelist(matrix, rowIds, colIds, whitelist, symmetric = False): #2exp?
		row_index = [ i for i, rowId in enumerate(rowIds) if rowId in whitelist ]
		if symmetric:
			col_index = row_index
		else:
			col_index = [ i for i, colId in enumerate(colIds) if colId in whitelist ]
		matrix = matrix[row_index]
		matrix = matrix[:,col_index]
		rowIds = [rowIds[i] for i in row_index]
		colIds = [colIds[i] for i in col_index]
		return matrix, rowIds, colIds


	@staticmethod
	def get_disparity_backbone_pval(matrix): #2exp?
		# by the moment, implementetion square (?)
		# TODO: Add a warning when not square matrix.
		if np.any(matrix < 0):
			warnings.warn("Negative values detected in matrix. Passing to positive values by subtracting the minimum.")
			matrix = matrix - np.min(matrix)
		
		# row_maxs = np.max(matrix, axis=1, keepdims=True)
    	# # Avoid division by zero if a row is all zeros
		# row_maxs[row_maxs == 0] = 1.0 
		# matrix = matrix / row_maxs
		W = np.sum(matrix, axis=0)
		k = (matrix > 0).sum(0)
		# operacion vectorizada.
		#
		pval_mat = np.ones(matrix.shape)
		for i in range(0,pval_mat.shape[1]):
			pval_mat[i, :] = (1 - matrix[i, :] / W[i])**(k[i] - 1)
		return pval_mat
