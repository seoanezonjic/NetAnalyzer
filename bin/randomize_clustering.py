#!/usr/bin/env python
import sys
import argparse
import random

##############################
#FUNCTIONS
##############################

def load_clusters(options):
	clusters = {}
	with open(options['input_file'], "r") as f:
		for line in f:
			line = line.rstrip().split(options['column_sep'])
			cluster = line[options['cluster_index']]
			node = line[options['node_index']]
			if options.get('node_sep') != None: 
				node = node.split(options['node_sep'])
				clusters[cluster] = node
			else:
				query = clusters.get(cluster)
				if query == None: 
					clusters[cluster] = [node] 
				else:
					query.append(node)
	return clusters

def random_sample(nodes, replacement, all_sizes, seed):
	random_clusters = {}
	uniq_node_list = set(nodes)
	random.seed(seed)
	for counter, cluster_size in enumerate(all_sizes):
		if cluster_size > len(uniq_node_list) and not replacement: sys.exit("Not enough nodes to generate clusters. Please activate replacement or change random mode") 
		random_nodes = random.sample(uniq_node_list, cluster_size)
		if not replacement: uniq_node_list = [n for n in uniq_node_list if n not in random_nodes]
		random_clusters[f"{counter}_random"] = random_nodes
	return random_clusters

def write_clusters(clusters, output_file, sep):
	with open(output_file, "w") as outfile:
		for cluster, nodes in clusters.items():
			if sep != None: nodes = [sep.join(nodes)]
			for node in nodes:
				outfile.write(f"{cluster}\t{node}\n")

##############################
#OPTPARSE
##############################

def based_0(string): return int(string) - 1
def array(string): return string.split(':')

parser = argparse.ArgumentParser(description='Perform clusters randomization')

parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
					help="Input file to create networks for further analysis")
parser.add_argument("-N", "--node_column", dest="node_index", default= 1, type=based_0, 
					help="Number of the nodes column")
parser.add_argument("-C", "--cluster_column", dest="cluster_index", default= 0, type=based_0, 
					help="Number of the clusters column")
parser.add_argument("-S", "--split_char", dest="column_sep", default = "\t", 
					help="Character for splitting input file. Default: tab")
parser.add_argument("-s", "--node_sep", dest="node_sep", default = None, 
					help="Node split character. This option must to be used when input file is aggregated")
parser.add_argument("-r", "--random_type", dest="random_type", default = ["size"], type = array, 
					help="Indicate random mode. 'size' for radomize clusters with the same size as input clusters. 'full_size' same as 'size' but all nodes are repaeted as same as input. 'fixed:n:s' for generate 'n' clusters of 's' nodes. Default = 'size'")
parser.add_argument("-R", "--replacement", dest="replacement", default=False, action='store_true',
					help="Boolean. Activates ramdom sampling with replacement. Sampling witout replacement will be executed instead.")  
parser.add_argument("-o", "--output_file", dest="output_file", default= 'random_clusters.txt', 
					help="Output file")
parser.add_argument("-a", "--aggregate_sep", dest="aggregate_sep", default = None, 
					help="This option activates aggregation in output. Separator character must be provided")

options = parser.parse_args()

##########################
#MAIN
##########################

options = vars(options)
clusters = load_clusters(options)

nodes = [ n for cl in clusters.values() for n in cl ] # Get list of nodes
if options['random_type'][0] != "full_size": nodes = set(nodes)

if "size" in options['random_type'][0] and len(options['random_type']) == 1:
	all_sizes = [ len(nodes) for cluster_id, nodes in clusters.items() ]
elif options['random_type'][0] == "fixed" and len(options['random_type']) == 3:
	all_sizes = [int(options['random_type'][2])] * int(options['random_type'][2]) # cluster_size * n clusters

random_clusters = random_sample(nodes, options['replacement'], all_sizes, 123)
write_clusters(random_clusters, options['output_file'], options['aggregate_sep'])
