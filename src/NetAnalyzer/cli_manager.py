import argparse
import os
from py_cmdtabs.cmdtabs import CmdTabs
from NetAnalyzer.main_modules import *

## TYPES 
def based_0(string): return int(string) - 1

def list_based_0(string): return CmdTabs.parse_column_indices(",", string)

def single_split(string, sep = ","):
    return string.strip().split(sep)

def double_split(string, sep1=";", sep2=","):
    return [sublst.split(sep2) for sublst in string.strip().split(sep1)]

def loading_dic(string, sep1=";", sep2=","):
    return {key: value for key, value in double_split(string, sep1=";", sep2=",")}

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

## Common options
##############################################

def add_kernel_flags(parser, multiple = False):
    if multiple:
        type_parse = lambda x: single_split(x, sep=";")
    else:
        type_parse = lambda x: str(x)

    parser.add_argument("-k", "--input_kernels", dest="kernel_files", default= None, type= lambda x: type_parse(x),
                        help="The roots from each kernel to integrate")
    parser.add_argument("-n", "--input_nodes", dest="node_files", default= None, type = lambda x: type_parse(x),
                        help="The list of node for each kernel in lst format")

def add_seed_flags(parser):
    parser.add_argument("--seed_nodes", dest="seed_nodes", default=None,
        help="The name of the nodes to use as seeds")
    parser.add_argument("--seed_sep", dest="seed_sep", default=",",
        help="Separator of seed genes. Only use when -s point to a file")

def add_random_seed(parser, default_seed=None):
    parser.add_argument("--seed", dest="seed", default= default_seed, type=int, 
        help="Allows to set a seed for the randomization process. Set to a number. Otherwise results are not reproducible.")

def add_output_flags(parser, default_opt={"output_file": "output_file"}):
    parser.add_argument("-o","--output_file", dest="output_file", default=default_opt['output_file'],
                        help="Output file name")

def add_input_graph_flags(parser, multinet = False):
    if multinet:
        parser.add_argument("-i", "--input_file", dest="input_file", default= None, type = lambda string: loading_dic(string, sep1=";", sep2=","), 
                    help="Input file to create networks for further analysis")
        parser.add_argument("-n","--node_names_file", dest="node_files", default=None, type = lambda string: loading_dic(string, sep1=";", sep2=","),
                    help="Files with node names corresponding to the input matrix, only use when -i is set to bin or matrix, could be two paths, indicating rows and cols, respectively. If just one path added, it is assumed to be for rows and cols")
        # parser.add_argument("-l","--layers", dest="layers", default=[['layer', '-']], type= lambda x: double_split(x, sep1=";",sep2=","),
        #                     help="Layer definition on network: layer1name,regexp1;layer2name,regexp2...")
    else:
        parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
                            help="Input file to create networks for further analysis")
        parser.add_argument("-n","--node_names_file", dest="node_files", default=None, type = lambda x: single_split(x, sep=","),
                            help="Files with node names corresponding to the input matrix, only use when -i is set to bin or matrix, could be two paths, indicating rows and cols, respectively. If just one path added, it is assumed to be for rows and cols")
        parser.add_argument("-l","--layers", dest="layers", default=[['layer', '-']], type= lambda x: double_split(x, sep1=";",sep2=","),
                            help="Layer definition on network: layer1name,regexp1;layer2name,regexp2...")
    parser.add_argument("-s","--split_char", dest="split_char", default='\t',
                        help = "Character for splitting input file. Default: tab")
    parser.add_argument("-f","--input_format", dest="input_format", default='pair',
                        help="Input file format: pair (default), bin, matrix")
    parser.add_argument("--both_repre_formats", dest="load_both", default=False, action='store_true',
                        help="If we need to load the adjacency matrixes and the graph object")

def add_resources_flags(parser, default_opt={"threads": 1}):
    parser.add_argument("-T", "--threads", dest="threads", default=default_opt["threads"], type=int,
        help="Number of threads to use in computation.")


def add_common_relations_process(parser):
    parser.add_argument("-N","--no_autorelations", dest="no_autorelations", default=False, action='store_true',
                        help="No processing autorelations")

##############################################

def integrate_kernels(args=None):
    parser = argparse.ArgumentParser(description='Integrate kernels or embedding in matrix format')
    add_kernel_flags(parser, multiple = True)
    add_output_flags(parser, default_opt={"output_file": "general_matrix"})
    parser.add_argument("-f","--format_kernel",dest= "input_format", default="bin", 
                        help= "The format of the kernels to integrate")
    parser.add_argument("-I", "--kernel_ids", dest="kernel_ids", default= None, type = lambda x: single_split(x, sep=";"),
                        help="The names of each kernel")
    # Integration conf
    parser.add_argument("-i","--integration_type",dest= "integration_type", default=None, 
                        help= "It specifies how to integrate the kernels")
    parser.add_argument("--raw_values", dest="raw_values", default=False, action='store_true',help="Select this option to use the negatives and positives values, without translation")
    parser.add_argument("--asym",dest= "symmetric", default=True, action='store_false',
                        help= "It specifies if the kernel matrixes are or not symmetric")

    # Resources
    add_resources_flags(parser=parser, default_opt={"threads": 8})  
    opts = parser.parse_args(args)
    main_integrate_kernels(opts)

def netanalyzer(args=None):
    parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')
    add_common_relations_process(parser)
    add_input_graph_flags(parser)
    add_output_flags(parser, default_opt={"output_file": "output_file"})
    add_random_seed(parser, default_seed=None)
    parser.add_argument("-O", "--ontology", dest="ontologies", default=[], type=lambda x: double_split(x, sep1=";",sep2=":"),
                        help="String that define which ontologies must be used with each layer. String definition:'layer_name1:path_to_obo_file1;layer_name2:path_to_obo_file2'")
    # Assoc
    parser.add_argument("-P","--use_pairs", dest="use_pairs", default='conn',
    help="Which pairs must be computed. 'all' means all posible pair node combinations and 'conn' means the pair are truly connected in the network. Default 'conn' ")
    parser.add_argument("-m","--association_method", dest="meth", default=None,
    help="select association method to perform the projections: counts, jaccard, simpson, geometric, cosine, pcc, hypergeometric, hypergeometric_bf, hypergeometric_bh, csi, transference, correlation, umap, pca, bicm")
    parser.add_argument("-a","--assoc_file", dest="assoc_file", default='assoc_values.txt',
    help="Output file name for association values")
    parser.add_argument("-p","--performance_file", dest="performance_file", default='perf_values.txt',
    help="Output file name for performance values")
    parser.add_argument("-u","--use_layers", dest="use_layers", default=[], type= lambda x: double_split(x, sep1=";",sep2=","),
    help="Set which layers must be used on association methods: layer1,layer2;layerA,layerB")
    parser.add_argument("-c","--control_file", dest="control_file", default=None,
    help="Control file name")
    # Kernel 
    parser.add_argument("-k","--kernel_method", dest="kernel", default=None,
    help="Kernel operation to perform with the adjacency matrix")
    parser.add_argument("--embedding_add_options", dest="embedding_add_options", default="",
    help="Additional options for embedding kernel methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...' ")
    parser.add_argument("-z","--normalize_kernel_values", dest="normalize_kernel", default=False, action='store_true',
    help="Apply cosine normalization to the obtained kernel")
    parser.add_argument("--coords2sim_type", dest="coords2sim_type", default="dotProduct", help= "Select the type of transformation from coords to similarity: dotProduct, normalizedScaling, infinity and int or float numbers")
    parser.add_argument("-K","--kernel_file", dest="kernel_file", default='kernel_file',
    help="Output file name for kernel values")
    # Plotting
    parser.add_argument("-g", "--graph_file", dest="graph_file", default=None,
    help="Build a graphic representation of the network")
    parser.add_argument("--graph_options", dest="graph_options", default={'method': 'elgrapho', 'layout': 'forcedir', 'steps': '30'}, type= graph_options_parse,
    help="Set graph parameters as 'NAME1=value1,NAME2=value2,...")
    # Nodes states
    parser.add_argument("-r","--reference_nodes", dest="reference_nodes", default=[], type= lambda x: single_split(x, sep=","),
    help="Node ids comma separared")
    parser.add_argument("-G","--group_nodes", dest="group_nodes", default={}, type= group_nodes_parse,
    help="File path or groups separated by ';' and group node ids comma separared")
    parser.add_argument("-d","--delete", dest="delete_nodes", default=[], type= lambda x: single_split(x, sep=";"),
    help="Remove nodes from file. If PATH;r then nodes not included in file are removed")
    # Compare cluster
    parser.add_argument("-R","--compare_clusters_reference", dest="compare_clusters_reference", default=None, type= group_nodes_parse,
    help="File path or groups separated by ';' and group node ids comma separared")
    # Build cluster
    parser.add_argument("-b", "--build_clusters_alg", dest="build_cluster_alg", default=None,
    help="Type of cluster algorithm")
    parser.add_argument("-B", "--build_clusters_add_options", dest="build_clusters_add_options", default="",
    help="Additional options for clustering methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...'")
    parser.add_argument("--output_build_clusters", dest="output_build_clusters", default=None, help= "output name for discovered clusters")
    # Expand cluster
    parser.add_argument("-x","--expand_clusters", dest="expand_clusters", default=None,
    help="Method to expand clusters Available methods: sht_path")
    parser.add_argument("--one_sht_pairs", dest="one_sht_pairs", default=False, action='store_true',
    help="add this flag if expand cluster needed with just one of the shortest paths")
    parser.add_argument("--output_expand_clusters", dest= "output_expand_clusters", default="expand_clusters.txt", help="outputname fopr expand clusters file")
    # Cluster metrics
    parser.add_argument("-M", "--group_metrics", dest="group_metrics", default=None, type= lambda x: single_split(x, sep=";"),
    help="Perform group group_metrics")
    parser.add_argument("--output_metrics_by_cluster", dest="output_metrics_by_cluster", default='group_metrics.txt', help= "output name for metrics by cluster file")
    parser.add_argument("-S", "--summarize_metrics", dest="summarize_metrics", default=None, type= lambda x: single_split(x, sep=";"),
    help="Summarize metrics from groups")
    parser.add_argument("--output_summarized_metrics", dest="output_summarized_metrics", default='group_metrics_summarized.txt', help= "output name for summarized metrics file")
    # Graph metrics by net or node
    parser.add_argument("-A", "--attributes", dest="get_attributes", default=[], type =lambda x: single_split(x, sep=","),
    help="String separated by commas with the name of network attribute")
    parser.add_argument("--graph_attributes", dest="get_graph_attributes", default=[], type =lambda x: single_split(x, sep=","),
    help="String separated by commas with the name of network attribute")
    parser.add_argument("--attributes_summarize", dest="attributes_summarize", default= False, action = "store_true", help="If the attribtes needs to be obtained summarized") 
    # DSL section
    parser.add_argument("--dsl_script", dest="dsl_script", default=None,
    help="Path to dsl script to perform complex analysis")
    # Resources
    add_resources_flags(parser=parser, default_opt={"threads": 2})

    opts = parser.parse_args(args)
    main_netanalyzer(opts)
	
def randomize_clustering(args=None):
    parser = argparse.ArgumentParser(description='Perform clusters randomization')
    add_output_flags(parser, default_opt={"output_file": "random_clusters.txt"})
    parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
    help="Input file to create networks for further analysis")
    parser.add_argument("-S", "--split_char", dest="column_sep", default = "\t", 
    help="Character for splitting input file. Default: tab")
    parser.add_argument("-a", "--aggregate_sep", dest="aggregate_sep", default = None, 
    help="This option activates aggregation in output. Separator character must be provided")
    parser.add_argument("-N", "--node_column", dest="node_index", default= 1, type=based_0, 
    help="Number of the nodes column")
    parser.add_argument("-C", "--cluster_column", dest="cluster_index", default= 0, type=based_0, 
    help="Number of the clusters column")
    parser.add_argument("-s", "--node_sep", dest="node_sep", default = None, 
    help="Node split character. This option must to be used when input file is aggregated")
    # random conf
    parser.add_argument("-r", "--random_type", dest="random_type", default = ["size"], type = lambda x: single_split(x,sep=":"), 
    help="""Indicate random mode. 'size' for radomize clusters with the same size as input clusters.
     'full_size' same as 'size' but all nodes are repaeted as same as input. 'fixed:n:s' for generate 'n' clusters of 's' nodes. Default = 'size'""")
    parser.add_argument("-R", "--replacement", dest="replacement", default=False, action='store_true',
    help="Boolean. Activates ramdom sampling with replacement. Sampling witout replacement will be executed instead.")  
    add_random_seed(parser,  default_seed=123)

    opts = parser.parse_args(args)
    main_randomize_clustering(opts)

def randomize_network(args=None):
    parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')
    add_input_graph_flags(parser)
    add_output_flags(parser, default_opt={"output_file": None})
    add_random_seed(parser, default_seed=None)
    # random conf
    parser.add_argument("-r", "--type_random", dest="type_random", default= None, 
            help="Randomized basis. 'nodes' Node-baseds randomize or 'links' Links-baseds randomize")
    opts = parser.parse_args(args)
    main_randomize_network(opts)

def ranker(args=None):
    parser = argparse.ArgumentParser(description='Get the ranks from a matrix similarity score and a list of seeds')
    add_seed_flags(parser)
    add_output_flags(parser, default_opt={"output_file": "ranked_genes"})
    add_kernel_flags(parser, multiple = False)

    # Output ranking
    parser.add_argument("--seed_presence", dest="seed_presence", default=None,
    help="Seed presence on list: 'remove', when seed is not added on the ranker calculation; 'annotate', when just tag on output is needed, and calculation is obtained inclusing the seeds: new, seed.")
    parser.add_argument("--header", dest="header", default=False, 
    action='store_true', help="Select this option if header needed")
    # filtering
    parser.add_argument("-f", "--filter", dest="filter", default=None,
    help="PATH to file with seed_name and genes to keep in output")
    parser.add_argument("--whitelist", dest="whitelist", default=None, type = open_whitelist, help= "File Path with the whitelist of nodes to take into account in the ranker process")
    parser.add_argument("--minimum_size", dest="minimum_size", default=1, type=int)
    # Options in ranker alg
    parser.add_argument("-N","--normalize_matrix", dest="normalize_matrix", default= None,
    help="Select the type of normalization, options are: None, by_column, by_row, by_row_col")
    parser.add_argument("-p", "--propagate", dest="propagate", default = False, action = "store_true")
    parser.add_argument("--propagate_options", dest="propagate_options", default =  '"tolerance": 1e-5, "iteration_limit": 100, "with_restart": 0',
    help="Additional options for propagation methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...'")
    # Benchmarking mode
    parser.add_argument("-l", "--cross_validation", dest="cross_validation", default=False, 
    action='store_true', help="To activate cross validation")
    parser.add_argument("-K", "--k_fold", dest="k_fold", default=None, type=int, 
    help="Indicate the number of itrations needed, not used for leave one out cross validation (loocv)")
    # Select tops
    parser.add_argument("-t", "--top_n", dest="top_n", default=None, type=int,
    help="Top N genes to print in output")
    parser.add_argument("--output_top", dest="output_top", default=None,
    help="File to save Top N genes")
    parser.add_argument("--representation_seed_metric", dest = "representation_seed_metric", default = "mean", 
        help = "select the type of representation on seed, default mean, options: mean and max")
    # Resources
    add_resources_flags(parser=parser, default_opt={"threads": 1})
    opts = parser.parse_args(args)
    main_ranker(opts)

def text2binary_matrix(args=None):
    parser = argparse.ArgumentParser(description="Transforming matrix format and obtaining statistics")
    add_output_flags(parser, default_opt={"output_file": None})
    parser.add_argument('-i', '--input_file', dest="input_file", default=None,
    help="input file")
    parser.add_argument('-b', '--byte_format', dest="byte_format", default="float64",
    help='Format of the numeric values stored in matrix. Default: float64, warning set this to less precission can modify computation results using this matrix.')
    parser.add_argument('-t', '--input_type', dest="input_type", default='pair',
    help='Set input format file. "pair", "matrix" or "bin"')
    parser.add_argument('-O', '--output_type', dest="output_type", default='bin',
    help='Set output format file. "bin" for binary (default) or "mat" for tabulated text file matrix')
    # Process matrix
    parser.add_argument('-d', '--set_diagonal', dest="set_diagonal", default=False, action='store_true',
    help='Set to 1.0 the main diagonal')
    parser.add_argument('-B', '--binarize', dest="binarize", default=None, type = float,
    help='Binarize matrix changin x >= thr to one and any other to zero into matrix given')
    parser.add_argument('-c', '--cutoff', dest="cutoff", default=None, type = float,
    help='Cutoff matrix values keeping just x >= and setting any other to zero into matrix given')
    # Get stats
    parser.add_argument('-s', '--get_stats', dest="stats", default=None,
    help='Get stats from the processed matrix')
    
    opts = parser.parse_args(args)
    main_text2binary_matrix(opts)

def net_explorer(args=None, test=False):
    parser = argparse.ArgumentParser(description="Transforming matrix format and obtaining statistics")
    add_common_relations_process(parser) # Common relations options
    add_input_graph_flags(parser, multinet = True) # Input graph
    add_seed_flags(parser) # Adding seeds
    add_output_flags(parser, default_opt={"output_file": "output_file"})
    # layer processing
    parser.add_argument('-c', '--layer_cutoff', dest="layer_cutoff", default={}, type = lambda string: loading_dic(string, sep1=";", sep2=","),
    help='Cutoff to apply to every layer in the multiplexed one')
    # Analysis options
    parser.add_argument("-l", "--neigh_level", dest="neigh_level", default={}, type = lambda string: loading_dic(string, sep1=";", sep2=","),
    help="Defining the level of neighbourhood on the initial set of nodes")
    parser.add_argument("--plot_network_method", dest="plot_network_method", default="pyvis",
    help="Defining the plot method used on report")
    parser.add_argument("--embedding_proj", dest="embedding_proj", default=None,
    help="Select different projections methods: umap")
    parser.add_argument("--compare_nets", dest="compare_nets", default=False, action="store_true")
    opts = parser.parse_args(args)
    to_test = main_net_explorer(opts, test)
    if test: return to_test
