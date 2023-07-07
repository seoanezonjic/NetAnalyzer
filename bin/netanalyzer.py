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
			data.append(line.rstrip().split("\t"))
	return data

def get_args(dsl_args):
	"""return args, kwargs"""
	args = []
	kwargs = {}
	for dsl_arg in dsl_args:
		if '=' in dsl_arg:
			k, v = dsl_arg.split('=', 1)
			kwargs[k] = eval(v)
		else:
			args.append(eval(dsl_arg))
	return args, kwargs

def execute_dsl_script(net_obj, dsl_path):
	with open(dsl_path, 'r') as file:
		for line in file:
			line = line.strip()
			if not line or line[0] == '#': continue
			command = line.split()
			func = getattr(net_obj, command.pop(0))
			args, kwargs = get_args(command)
			func(*args, **kwargs)

########################### OPTPARSE ########################
#############################################################

def based_0(string): return int(string) - 1
def string_list(string): return string.split(",")
def layer_parse(string): return [sublst.split(",") for sublst in string.split(";")]
def list_parse(string): return [sublst.split(":") for sublst in string.split(";")]
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
parser.add_argument("-n","--node_names_file", dest="node_files", default=None, type = lambda x: x.split(","),
					help="Files with node names corresponding to the input matrix, only use when -i is set to bin or matrix, could be two paths, indicating rows and cols, respectively. If just one path added, it is assumed to be for rows and cols")
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
parser.add_argument("-T","--processes", dest="processes", default=2, type= int,
					help="Number of processes to use in computation")
parser.add_argument("-r","--reference_nodes", dest="reference_nodes", default=[], type= lambda x: x.split(","),
					help="Node ids comma separared")
parser.add_argument("-R","--compare_clusters_reference", dest="compare_clusters_reference", default=None, type= group_nodes_parse,
					help="File path or groups separated by ';' and group node ids comma separared")
parser.add_argument("-G","--group_nodes", dest="group_nodes", default={}, type= group_nodes_parse,
					help="File path or groups separated by ';' and group node ids comma separared")
parser.add_argument("-b", "--build_clusters_alg", dest="build_cluster_alg", default=None,
					help="Type of cluster algorithm")
parser.add_argument("--output_build_clusters", dest="output_build_clusters", default=None, help= "output name for discovered clusters")
parser.add_argument("-B", "--build_clusters_add_options", dest="build_clusters_add_options", default="",
					help="Additional options for clustering methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...'")
parser.add_argument("-d","--delete", dest="delete_nodes", default=[], type= lambda x: x.split(";"),
					help="Remove nodes from file. If PATH;r then nodes not included in file are removed")
parser.add_argument("-x","--expand_clusters", dest="expand_clusters", default=None,
					help="Method to expand clusters Available methods: sht_path")
parser.add_argument("--output_expand_clusters", dest= "output_expand_clusters", default="expand_clusters.txt", help="outputname fopr expand clusters file")
parser.add_argument("--one_sht_pairs", dest="one_sht_pairs", default=False, action='store_true',
					help="add this flag if expand cluster needed with just one of the shortest paths")
parser.add_argument("-M", "--group_metrics", dest="group_metrics", default=None, type= lambda x: x.split(";"),
					help="Perform group group_metrics")
parser.add_argument("--output_metrics_by_cluster", dest="output_metrics_by_cluster", default='group_metrics.txt', help= "output name for metrics by cluster file")
parser.add_argument("-S", "--summarize_metrics", dest="summarize_metrics", default=None, type= lambda x: x.split(";"),
					help="Summarize metrics from groups")
parser.add_argument("--output_summarized_metrics", dest="output_summarized_metrics", default='group_metrics_summarized.txt', help= "output name for summarized metrics file")
parser.add_argument("--seed", dest="seed", default=None, type = lambda x: x,
					help="sepecify seed for clusterin processes")
parser.add_argument("-A", "--attributes", dest="get_attributes", default=[], type =string_list,
					help="String separated by commas with the name of network attribute")
parser.add_argument("--attributes_summarize", dest="attributes_summarize", default= False, action = "store_true", help="If the attribtes needs to be obtained summarized") 
parser.add_argument("--dsl_script", dest="dsl_script", default=None,
					help="Path to dsl script to perform complex analysis")
parser.add_argument("-O", "--ontology", dest="ontologies", default=[], type=list_parse,
					help="String that define which ontologies must be used with each layer. String definition:'layer_name1:path_to_obo_file1;layer_name2:path_to_obo_file2'")
options = parser.parse_args()

########## MAIN ##########
##########################
print("Loading network data")
opts = vars(options)
fullNet = Net_parser.load(opts) # FRED: Remove this part of vars and modify the loads methods (Tlk wth PSZ)
fullNet.set_compute_pairs(options.use_pairs, not options.no_autorelations)
fullNet.threads = options.processes

fullNet.reference_nodes = options.reference_nodes
for ont_data in opts['ontologies']:
	layer_name, ontology_file_path = ont_data
	fullNet.link_ontology(ontology_file_path, layer_name)
	fullNet.ontology

if options.group_nodes:
	fullNet.set_groups(options.group_nodes)

if options.delete_nodes:
  node_list = load_file(options.delete_nodes[0])
  node_list = [item for sublist in node_list for item in sublist] 
  mode = options.delete_nodes[1] if len(options.delete_nodes) > 1 else 'd'
  fullNet.delete_nodes(node_list, mode)

if options.dsl_script is not None:
	execute_dsl_script(fullNet, options.dsl_script)
	sys.exit()

if options.meth is not None:
	print(f"Performing association method {options.meth} on network \n")
	if options.meth == "transference":
		if not (options.use_layers[0][0], options.use_layers[1][0]) in fullNet.matrices["adjacency_matrices"]:
			fullNet.generate_adjacency_matrix(options.use_layers[0][0], options.use_layers[1][0])
		if not (options.use_layers[1][0], options.use_layers[0][1]) in fullNet.matrices["adjacency_matrices"]:
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

	fullNet.get_kernel(layers2kernel, options.kernel, options.normalize_kernel, embedding_kwargs, add_to_object= True)
	fullNet.write_kernel(layers2kernel, options.kernel, options.kernel_file)

if options.graph_file is not None:
  options.graph_options['output_file'] = options.graph_file
  fullNet.plot_network(options.graph_options)

# Group creation
if options.build_cluster_alg is not None:
	exec('clust_kwargs = {' + options.build_clusters_add_options +'}') # This allows inject custom arguments for each clustering method
	fullNet.discover_clusters(options.build_cluster_alg, clust_kwargs, **{'seed': options.seed})

	if options.output_build_clusters is None:
		options.output_build_clusters = options.build_cluster_alg + '_' + 'discovered_clusters.txt'
  
	with open(options.output_build_clusters, 'w') as out_file:
		for cl_id, nodes in fullNet.group_nodes.items():
			for node in nodes: out_file.write(f"{cl_id}\t{node}\n")

# Group metrics by cluster.
if options.group_metrics:
	fullNet.compute_group_metrics(output_filename= options.output_metrics_by_cluster, metrics = options.group_metrics)
	
# Group metrics summarized.
if options.summarize_metrics:
	fullNet.compute_summarized_group_metrics(output_filename=options.output_summarized_metrics, metrics = options.summarize_metrics)

# Comparing Group Families (Two by now)
if options.compare_clusters_reference is not None:
	res = fullNet.compare_partitions(options.compare_clusters_reference)
	print(str(res.score))

# Group Expansion
if options.expand_clusters is not None:
  expanded_clusters = fullNet.expand_clusters(options.expand_clusters, options.one_sht_pairs)
  with open(options.output_expand_clusters, 'w') as out_file:
    for cl_id, nodes in expanded_clusters.items():
      for node in nodes: out_file.write(f"{cl_id}\t{node}\n")

if len(options.get_attributes) > 0:
  fullNet.get_node_attributes(options.get_attributes, summary=options.attributes_summarize, output_filename= "node_attributes.txt")

