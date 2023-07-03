#! /usr/bin/env python

import argparse
import sys
import os
import glob
import numpy as np
ROOT_PATH=os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_PATH, '..'))
from NetAnalyzer import Adv_mat_calc

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
parser.add_argument('-s', '--get_stats', dest="stats", default=None,
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
	matrix = Adv_mat_calc.filter_cutoff_mat(matrix, options.binarize)
	matrix = Adv_mat_calc.binarize_mat(matrix)


if options.cutoff is not None and options.binarize is None:
	matrix = Adv_mat_calc.filter_cutoff_mat(matrix, options.cutoff)


if options.stats is not None:
	stats = Adv_mat_calc.get_stats_from_matrix(matrix)
	with open(options.stats, 'w') as f:
		for row in stats: f.write("\t".join([str(item) for item in row]) + "\n")


if options.output_type == 'bin':	
	np.save(options.output_matrix_file, matrix)
elif options.output_type == 'mat':
	np.savetxt(options.output_matrix_file, matrix, delimiter='\t')