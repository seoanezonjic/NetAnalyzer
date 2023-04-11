#! /usr/bin/env python

import argparse
import sys
import os
import glob
import random
ROOT_PATH=os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_PATH, '..'))
from NetAnalyzer import Net_parser, NetAnalyzer

########################### METHODS ########################
############################################################

def load_file(path):
	data = []
	with open(path, 'r') as file:
		for line in file:
			data.append(line.rstrip.split("\t"))
	return data

########################### OPTPARSE ########################
#############################################################

def based_0(string): return int(string) - 1
def layer_parse(string): return [sublst.split(",") for sublst in string.split(";")]
def group_nodes_parse(string):
	group_nodes = {}
	if os.path.isfile(string):
		with open(string) as file:
			for line in file:
				groupID, nodeID = line.strip().split("\t")
				query = group_nodes.get(groupID)
				if query is None:
					group_nodes[groupID] = [nodeID]
				else:
					query.append(nodeID)
	else:
		for i, group in enumerate(string.split(";")):
			group_nodes[i] = group.split(',')
	
	return group_nodes

def graph_options_parse(string):
	graph_options = {}
	for pair in string.split(','):
	  fields = pair.split('=')
	  graph_options[fields[0]] = fields[1]
	return graph_options

parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')

parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
					help="Input file to create networks for further analysis")
parser.add_argument("-n","--node_names_file", dest="node_file", default=None,
					help="File with node names corresponding to the input matrix, only use when -i is set to bin or matrix")
parser.add_argument("-s","--split_char", dest="split_char", default='\t',
					help = "Character for splitting input file. Default: tab")
parser.add_argument("-f","--input_format", dest="input_format", default='pair',
					help="Input file format: pair (default), bin, matrix")
parser.add_argument("--both_repre_formats", dest="load_both", default=False, action='store_true',
					help="If we need to load the adjacency matrixes and the graph object")
parser.add_argument("-o","--output_file", dest="output_file", default='output_file',
					help="Output file name")
parser.add_argument("-P","--use_pairs", dest="use_pairs", default='conn',
					help="Which pairs must be computed. 'all' means all posible pair node combinations and 'conn' means the pair are truly connected in the network. Default 'conn' ")
parser.add_argument("-a","--assoc_file", dest="assoc_file", default='assoc_values.txt',
					help="Output file name for association values")
parser.add_argument("-K","--kernel_file", dest="kernel_file", default='kernel_file',
					help="Output file name for kernel values")
parser.add_argument("-p","--performance_file", dest="performance_file", default='perf_values.txt',
					help="Output file name for performance values")
parser.add_argument("-l","--layers", dest="layers", default=[['layer', '-']], type= layer_parse,
					help="Layer definition on network: layer1name,regexp1;layer2name,regexp2...")
parser.add_argument("-u","--use_layers", dest="use_layers", default=[], type= layer_parse,
					help="Set which layers must be used on association methods: layer1,layer2;layerA,layerB")
parser.add_argument("-c","--control_file", dest="control_file", default=None,
					help="Control file name")
parser.add_argument("-m","--association_method", dest="meth", default=None,
					help="Control file name")
parser.add_argument("-k","--kernel_method", dest="kernel", default=None,
					help="Kernel operation to perform with the adjacency matrix")
parser.add_argument("--embedding_add_options", dest="embedding_add_options", default="",
					help="Additional options for embedding kernel methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...' ")
parser.add_argument("-N","--no_autorelations", dest="no_autorelations", default=False, action='store_true',
					help="Kernel operation to perform with the adjacency matrix")
parser.add_argument("-z","--normalize_kernel_values", dest="normalize_kernel", default=False, action='store_true',
					help="Apply cosine normalization to the obtained kernel")
parser.add_argument("-g", "--graph_file", dest="graph_file", default=None,
					help="Build a graphic representation of the network")
parser.add_argument("--graph_options", dest="graph_options", default={'method': 'elgrapho', 'layout': 'forcedir', 'steps': '30'}, type= graph_options_parse,
					help="Set graph parameters as 'NAME1=value1,NAME2=value2,...")
parser.add_argument("-T","--threads", dest="threads", default=0, type= based_0,
					help="Number of threads to use in computation, one thread will be reserved as manager.")
parser.add_argument("-r","--reference_nodes", dest="reference_nodes", default=[], type= lambda x: x.split(","),
					help="Node ids comma separared")
parser.add_argument("-R","--compare_clusters_reference", dest="compare_clusters_reference", default=None, type= group_nodes_parse,
					help="File path or groups separated by ';' and group node ids comma separared")
parser.add_argument("-G","--group_nodes", dest="group_nodes", default={}, type= group_nodes_parse,
					help="File path or groups separated by ';' and group node ids comma separared")
parser.add_argument("-b", "--build_clusters_alg", dest="build_cluster_alg", default=None,
					help="Type of cluster algorithm")
parser.add_argument("-B", "--build_clusters_add_options", dest="build_clusters_add_options", default="",
					help="Additional options for clustering methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...'")
parser.add_argument("-d","--delete", dest="delete_nodes", default=[], type= lambda x: x.split(";"),
					help="Remove nodes from file. If PATH;r then nodes not included in file are removed")
parser.add_argument("-x","--expand_clusters", dest="expand_clusters", default=None,
					help="Method to expand clusters Available methods: sht_path")
parser.add_argument("-M", "--group_metrics", dest="group_metrics", default=None, type= lambda x: x.split(";"),
					help="Perform group group_metrics")
parser.add_argument("-S", "--summarize_metrics", dest="summarize_metrics", default=False, action='store_true',
					help="Summarize metrics from groups")
parser.add_argument("--seed", dest="seed", default=None, type = lambda x: x,
					help="sepecify seed for clusterin processes")

options = parser.parse_args()
########## MAIN ##########
##########################
print("Loading network data")
fullNet = Net_parser.load(vars(options)) # FRED: Remove this part of vars and modify the loads methods (Tlk wth PSZ)
fullNet.reference_nodes = options.reference_nodes
#fullNet.threads = options.threads
fullNet.group_nodes = options.group_nodes
fullNet.set_compute_pairs(options.use_pairs, not options.no_autorelations)

if options.delete_nodes:
  node_list = load_file(options.delete_nodes[0])
  node_list = [item for sublist in node_list for item in sublist] 
  mode = options.delete_nodes[1] if len(options.delete_nodes) > 1 else 'd'
  fullNet.delete_nodes(node_list, mode)

if options.meth is not None:
	print(f"Performing association method {options.meth} on network \n")
	if options.meth == "transference":
		if not (options.use_layers[0][0], options.use_layers[1][0]) in fullNet.adjacency_matrices:
			fullNet.generate_adjacency_matrix(options.use_layers[0][0], options.use_layers[1][0])
		if not (options.use_layers[1][0], options.use_layers[0][1]) in fullNet.adjacency_matrices:
			fullNet.generate_adjacency_matrix(options.use_layers[1][0], options.use_layers[0][1])

		fullNet.get_association_values(
			(options.use_layers[0][0], options.use_layers[1][0]), 
			(options.use_layers[1][0],options.use_layers[0][1]),
			"transference")
	else:
		fullNet.get_association_values(
			options.use_layers[0],
			options.use_layers[1][0], 
			options.meth)

	with open(options.assoc_file, 'w') as f:
		for val in fullNet.association_values[options.meth][:-1]:
			f.write("\t".join(map(str,val)) + "\n")
		f.write("\t".join(map(str,fullNet.association_values[options.meth][-1])))
	print(f"End of analysis: {options.meth}")

	if options.control_file != None:
		with open(options.control_file, "r") as f:
			control = [ control.append(line.rstrip().split("\t")) for line in f ]
		Performancer.load_control(control)
		predictions = fullNet.association_values[options.meth]
		performance = Performancer.get_pred_rec(predictions)
		with open(options.performance_file, 'r') as f:
			f.write("\t".join(['cut', 'prec', 'rec', 'meth']) + "\n")
			for item in performance:
				item.append(options['meth'])
				f.write("\t".join(item) + "\n")

if options.kernel is not None:
	exec('embedding_kwargs = {' + options.embedding_add_options +'}') # This allows inject custom arguments for each embedding method
	if len(options.use_layers) == 1:
		layers2kernel = (options.use_layers[0][0], options.use_layers[0][0]) # we use only a layer to perform the kernel, so only one item it is selected.
	else:
		layers2kernel = tuple(options.use_layers[0])

	fullNet.get_kernel(layers2kernel, options.kernel, options.normalize_kernel, embedding_kwargs)
	fullNet.write_kernel(layers2kernel, options.kernel_file)

if options.graph_file is not None:
  options.graph_options['output_file'] = options.graph_file
  fullNet.plot_network(options.graph_options)

# Group creation
if options.build_cluster_alg is not None:
	exec('clust_kwargs = {' + options.build_clusters_add_options +'}') # This allows inject custom arguments for each clustering method
	fullNet.discover_clusters(options.build_cluster_alg, clust_kwargs, **{'seed': options.seed})
  
	with open(os.path.join(os.path.dirname(options.output_file), options.build_cluster_alg + '_' + 'discovered_clusters.txt'), 'w') as out_file:
		for cl_id, nodes in fullNet.group_nodes.items():
			for node in nodes: out_file.write(f"{cl_id}\t{node}\n")

# Group metrics 
if options.group_metrics:
	if options.summarize_metrics:
		fullNet.compute_summarized_group_metrics(output_filename=os.path.join(os.path.dirname(options.output_file), 'group_metrics_summarized.txt'), metrics = options.group_metrics)
	else:
		fullNet.compute_group_metrics(output_filename= os.path.join(os.path.dirname(options.output_file), 'group_metrics.txt'), metrics = options.group_metrics)

# Comparing Group Families (Two by now)
if options.compare_clusters_reference is not None:
	res = fullNet.compare_partitions(options.compare_clusters_reference)
	print(str(res.score))

# Group Expansion
if options.expand_clusters is not None:
  expanded_clusters = fullNet.expand_clusters(options.expand_clusters)
  with open(os.path.join(os.path.dirname(options.output_file), 'expand_clusters.txt'), 'w') as out_file:
    for cl_id, nodes in expanded_clusters.items():
      for node in nodes: out_file.write(f"{cl_id}\t{node}\n")