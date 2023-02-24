#! /usr/bin/env python

import argparse
import sys
import os
import glob
import numpy as np

#############################################################################
## METHODS
#############################################################################

def load_matrix_file(source, splitChar = "\t"):
	matrix = None
	counter = 0
	for line in source:
		line = line.strip()
		
		row = [float(c) for c in line.split(splitChar)]
		if matrix is None:
			matrix = np.zeros((len(row), len(row)))
		for i, val in enumerate(row):
			matrix[counter, i] = val 	
		counter += 1

	return matrix

def load_pair_file(source, byte_format = "float32"): 
	# Not used byte_forma parameter
	connections = {}
	for line in source:
		node_a, node_b, weight = line.strip().split("\t")
		weight = float(weight) if weight is not None else 1.0 
		add_pair(node_a, node_b, weight, connections)
		add_pair(node_b, node_a, weight, connections)

	matrix, names = dicti2wmatrix_squared(connections)
	return matrix, names


def add_pair(node_a, node_b, weight, connections):
	query = connections.get(node_a)
	if query is not None:
		query[node_b] = weight
	else:
		subhash = {}
		subhash[node_b] = weight
		connections[node_a] = subhash

def dicti2wmatrix_squared(dicti,symm= True):
	element_names = dicti.keys()
	matrix = np.zeros((len(element_names), len(element_names)))
	i = 0
	for  elementA, relations in dicti.items():
		for j, elementB in enumerate(element_names):
			if elementA != elementB:
				query = relations.get(elementB)
				if query is not None:
					matrix[i, j] = query
					if symm:
						matrix[j, i] = query 
		i += 1
	return matrix, element_names

def get_stats(matrix):
	stats = []
	#TODO: trnasform to Numo::Array operations
	primary_stats = get_primary_stats(matrix)
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
	connections = get_connection_number(matrix)
	connection_stats = get_primary_stats(connections)
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

def transform_keys(hash, function):
	new_hash = {}
	for key, val in hash.items():
		new_key = function(key)
		new_hash[new_key] = val

	return new_hash

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

	quartile_stats = get_quartiles(values, stats["count"])
	stats.update(quartile_stats)
	non_zero_values = [v for v in values if v != 0]
	quartile_stats_non_zero = get_quartiles(non_zero_values, stats["countNonZero"])
	stats.update(transform_keys(quartile_stats_non_zero, lambda x: x + "NonZero"))
	get_composed_stats(stats, values)
	return stats

def get_quartiles(values, n_items):
	stats = {}
	stats['q1'] = np.percentile(values,25)
	stats['median'] = np.percentile(values,50)
	stats['q3'] = np.percentile(values,75)
	return stats

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

#############################################################################
## OPTPARSE
#############################################################################

parser = argparse.ArgumentParser(description="Transforming matrix format and obtaining statistics")

parser.add_argument('-i', '--input_file', dest="input_file", default=None,
	help="input file")
parser.add_argument('-o', '--output_matrix_file', dest="output_matrix_file", default=None,
	help="Output matrix file")
parser.add_argument('-b', '--byte_format', dest="byte_format", default="float64",
	help='Format of the numeric values stored in matrix. Default: float64, warning set this to less precission can modify computation results using this matrix.')
parser.add_argument('-t', '--input_type', dest="input_type", default='pair',
	help='Set input format file. "pair", "matrix" or "bin"')
parser.add_argument('-d', '--set_diagonal', dest="set_diagonal", default=False, action='store_true',
	help='Set to 1.0 the main diagonal')
parser.add_argument('-B', '--binarize', dest="binarize", default=None, type = lambda x: float(x),
	help='Binarize matrix changin x >= thr to one and any other to zero into matrix given')
parser.add_argument('-c', '--cutoff', dest="cutoff", default=None, type = lambda x: float(x),
	help='Cutoff matrix values keeping just x >= and setting any other to zero into matrix given')
parser.add_argument('-s', '--get_stats', dest="stats", default=False, action= "store_true",
	help='Get stats from the processed matrix')
parser.add_argument('-O', '--output_type', dest="output_type", default='bin',
	help='Set output format file. "bin" for binary (default) or "mat" for tabulated text file matrix')
options = parser.parse_args()

################################################################################
## MAIN
###############################################################################
if options.input_file == '-':
	source = sys.stdin 
else:
	source = open(options.input_file) 

if options.input_type == 'bin':
	matrix = np.load(options.input_file)
elif options.input_type == 'matrix':
	matrix = load_matrix_file(source)
elif options.input_type == 'pair':
	matrix, names = load_pair_file(source, options.byte_format)
	with open(options.output_matrix_file + ".lst", 'w') as f:
		f.write("\n".join(names))

source.close()

if options.set_diagonal:
	elements = matrix.shape[-1]
	for n in range(elements):
		matrix[n, n] = 1.0


if options.binarize is not None and options.cutoff is None:
	elements = matrix.shape[-1]
	for i in range(elements):
		for j in range(elements):
			matrix[i,j] = 1 if matrix[i,j] >= options.binarize else 0 


if options.cutoff is not None and options.binarize is None:
	elements = matrix.shape[-1]
	for i in range(elements):
		for j in range(elements):
			matrix[i,j] = matrix[i,j] if matrix[i,j] >= options.cutoff else 0 


if options.stats:
	stats = get_stats(matrix)

	for stat in stats:
		print("\t".join(stat))


if options.output_type == 'bin':	
	np.save(options.output_matrix_file, matrix)
elif options.output_type == 'mat':
	np.savetxt(options.output_matrix_file, matrix, delimiter='\t')