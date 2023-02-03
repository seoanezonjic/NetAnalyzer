#!usr/bin/env python

import argparse
import sys
import os
import glob
ROOT_PATH=os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_PATH, '..'))
from NetAnalyzer import Net_parser, NetAnalyzer

########################### METHODS ########################
############################################################

def load_file(path):
	data = []
	with open(path, 'r') as file:
		for line in file:
			data.append(line.rstrip.split("\t"))

########################### OPTPARSE ########################
#############################################################

#def threads_based_0(trheads): return int(threads) - 1
#
#parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')
#
#parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
#					help="Input file to create networks for further analysis")
#parser.add_argument("-n","--node_names_file", dest="node_file", default=None,
#					help="File with node names corresponding to the input matrix, only use when -i is set to bin or matrix")
#parser.add_argument("-f","--input_format", dest="input_format", default='pair',
#					help="Input file format: pair (default), bin, matrix")
#parser.add_argument("-s","--split_char", dest="split_char", default='\t',
#					help="Character for splitting input file. Default: tab")
#parser.add_argument("-P","--use_pairs", dest="use_pairs", default='conn',
#					help="Which pairs must be computed. 'all' means all posible pair node combinations and 'conn' means the pair are truly connected in the network. Default 'conn' ")
#parser.add_argument("-o","--output_file", dest="output_file", default='network2plot',
#					help="Output file name")
#parser.add_argument("-a","--assoc_file", dest="assoc_file", default='assoc_values.txt',
#					help="Output file name for association values")
#parser.add_argument("-K","--kernel_file", dest="kernel_file", default='kernel_file',
#					help="Output file name for kernel values")
#parser.add_argument("-p","--performance_file", dest="performance_file", default='perf_values.txt',
#					help="Output file name for performance values")

# reference nodes
# threads
# group_nodes
# use_pairs
# no_autorrelations
# delete nodes
# meth
# use_layers
# assoc_file
# control_file
# performance_file
# kernel
# normalize_kernel
# kernel_file

########## MAIN ##########
##########################
puts "Loading network data"
fullNet = Net_parser.load(vars(parser)) # FRED: Remove this part of vars and modify the loads methods (Tlk wth PSZ)
fullNet.reference_nodes = parser.reference_nodes
fullNet.threads = parser.threads
fullNet.group_nodes = parser.group_nodes
fullNet.set_compute_pairs(parser.use_pairs, not parser.no_autorelations)

if not parser.delete_nodes:
  node_list = load_file(parser.delete_nodes[0])
  node_list = [item for sublist in node_list for item in sublist] # flatten method not available on python
  mode = parser.delete_nodes[1] if len(parser.delete_nodes) > 1 else 'd'
  fullNet.delete_nodes(node_list, mode)

if parser.meth is not None:
	print(f"Performing association method {parser.meth} on network \n")
	if parser.meth == "transference":
		fullNet.generate_adjacency_matrix(parser.use_layers[0][0], parser.use_layers[0][1])
		fullNet.generate_adjacency_matrix(parser.use_layers[1][0], parser.use_layers[1][1])
		fullNet.get_association_values(
			[parser.use_layers[0][0], parser.use_layers[0][1]], 
			[parser.use_layers[1][0], parser.use_layers[1][1]],
			"transference")
	else:
		fullNet.get_association_values(
			parser.use_layers[0],
			parser.use_layers[1][0], 
			parser.meth)
	with open(parser.assoc_file, 'w') as f:
		for val in fullNet.association_values[parser.meth]:
			f.write("\t".join(map(str,val)) + "\n")

  if parser.control_file is not None:
  	print(f"Doing validation on association values obtained from method {parser.meth}")
  	control = []
  	with open(parser.control_file) as file:
  		for line in file:
  			control.append(line.strip.split("\t"))

  	Performancer.load_control(control)
    predictions = fullNet.association_values[parser.meth]
  	performance = Performancer.get_pred_rec(predictions)
  	with open(parser.performance_file, 'w') as file:
  		file.write(%w[cut prec rec meth].join("\t"))
  		for item in performance:
  			item.append(str(parser.meth))
  			f.write("\t".join(map(str,item)) + "\n")
  		
  print(f"End of analysis: {parser.meth}")

if parser.kernel is not None:
  layer2kernel = parser.use_layers[0] # we use only a layer to perform the kernel, so only one item it is selected.
  fullNet.get_kernel(layer2kernel, parser.kernel, parser.normalize_kernel)
  fullNet.write_kernel(layer2kernel, parser.kernel_file)

