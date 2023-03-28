#! /usr/bin/env python

import argparse
import sys
import os
import glob
import numpy as np
import timeit

ROOT_PATH=os.path.dirname(__file__)
sys.path.insert(0,os.path.join(ROOT_PATH, '..'))
from NetAnalyzer import Kernels

########################### OPTPARSE ########################
#############################################################

parser = argparse.ArgumentParser(description='Integrate kernels or embedding in matrix format')
parser.add_argument("-t", "--input_kernels", dest="kernel_files", default= None, type= lambda x : x.strip().split(";"),
					help="The roots from each kernel to integrate")
parser.add_argument("-n", "--input_nodes", dest="node_files", default= None, type = lambda x : x.strip().split(";"),
					help="The list of node for each kernel in lst format")
parser.add_argument("-I", "--kernel_ids", dest="kernel_ids", default= None, type = lambda x : x.strip().split(";"),
					help="The names of each kernel")
parser.add_argument("-f","--format_kernel",dest= "input_format", default="bin", 
					help= "The format of the kernels to integrate")
parser.add_argument("-i","--integration_type",dest= "integration_type", default=None, 
					help= "It specifies how to integrate the kernels")
parser.add_argument("--cpu",dest= "n_workers", default=8,  type = lambda x: int(x),
					help= "It specifies the number of cpus available for the process parallelization")
parser.add_argument("-o","--output_matrix",dest= "output_matrix_file", default="general_matrix", 
					help= "The name of the matrix output")
options = parser.parse_args()

######################### MAIN #############################
############################################################
kernels = Kernels()

if not options.kernel_ids:
	options.kernel_ids = list(range(0,len(options.kernel_files)))
	options.kernel_ids = [str(k) for k in options.kernel_ids]

# TODO: Consider adding more options to integrate to accept other formats
if options.input_format == "bin":
	kernels.load_kernels_by_bin_matrixes(options.kernel_files, options.node_files, options.kernel_ids)
	kernels.create_general_index()

if options.integration_type is not None:
	kernels.integrate_matrix(options.integration_type, options.n_workers)

if options.output_matrix_file is not None:
	kernel, names = kernels.integrated_kernel
	np.save(options.output_matrix_file, kernel)

	with open(options.output_matrix_file +'.lst', 'w') as f:
		for name in names:
			f.write(name + "\n")  

