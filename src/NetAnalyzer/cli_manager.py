import argparse
import sys
import os
import glob
import numpy as np
from py_cmdtabs.cmdtabs import CmdTabs
import random
import py_exp_calc.exp_calc as pxc

""" ROOT_PATH=os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_PATH, '..')) """
from NetAnalyzer import Net_parser, NetAnalyzer
from NetAnalyzer import Kernels
from NetAnalyzer import Net_parser, NetAnalyzer
from NetAnalyzer import Ranker
from NetAnalyzer.performancer import Performancer


## TYPES 
def based_0(string): return int(string) - 1

def list_based_0(string): return CmdTabs.parse_column_indices(",", string)

def single_split(string, sep = ","):
    return string.strip().split(sep)

def double_split(string, sep1=";", sep2=","):
    return [sublst.split(sep2) for sublst in string.strip().split(sep1)]

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

##############################################

def integrate_kernels(args=None):
    parser = argparse.ArgumentParser(description='Integrate kernels or embedding in matrix format')
    parser.add_argument("-t", "--input_kernels", dest="kernel_files", default= None, type= lambda x: single_split(x, sep=";"),
                        help="The roots from each kernel to integrate")
    parser.add_argument("-n", "--input_nodes", dest="node_files", default= None, type = lambda x: single_split(x, sep=";"),
                        help="The list of node for each kernel in lst format")
    parser.add_argument("-I", "--kernel_ids", dest="kernel_ids", default= None, type = lambda x: single_split(x, sep=";"),
                        help="The names of each kernel")
    parser.add_argument("-f","--format_kernel",dest= "input_format", default="bin", 
                        help= "The format of the kernels to integrate")
    parser.add_argument("-i","--integration_type",dest= "integration_type", default=None, 
                        help= "It specifies how to integrate the kernels")
    parser.add_argument("--cpu",dest= "n_workers", default=8,  type = int,
                        help= "It specifies the number of cpus available for the process parallelization")
    parser.add_argument("--asym",dest= "symmetric", default=True, action='store_false',
                        help= "It specifies if the kernel matrixes are or not symmetric")
    parser.add_argument("-o","--output_matrix",dest= "output_matrix_file", default="general_matrix", 
                        help= "The name of the matrix output")
    
    opts = parser.parse_args(args)
    main_integrate_kernels(opts)

def netanalyzer(args=None):
    parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')
    parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
                        help="Input file to create networks for further analysis")
    parser.add_argument("-n","--node_names_file", dest="node_files", default=None, type = lambda x: single_split(x, sep=","),
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
    parser.add_argument("-l","--layers", dest="layers", default=[['layer', '-']], type= lambda x: double_split(x, sep1=";",sep2=","),
                        help="Layer definition on network: layer1name,regexp1;layer2name,regexp2...")
    parser.add_argument("-u","--use_layers", dest="use_layers", default=[], type= lambda x: double_split(x, sep1=";",sep2=","),
                        help="Set which layers must be used on association methods: layer1,layer2;layerA,layerB")
    parser.add_argument("-c","--control_file", dest="control_file", default=None,
                        help="Control file name")
    parser.add_argument("-m","--association_method", dest="meth", default=None,
                        help="select association method to perform the projections")
    parser.add_argument("-k","--kernel_method", dest="kernel", default=None,
                        help="Kernel operation to perform with the adjacency matrix")
    parser.add_argument("--embedding_add_options", dest="embedding_add_options", default="",
                        help="Additional options for embedding kernel methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...' ")
    parser.add_argument("-N","--no_autorelations", dest="no_autorelations", default=False, action='store_true',
                        help="Kernel operation to perform with the adjacency matrix")
    parser.add_argument("-z","--normalize_kernel_values", dest="normalize_kernel", default=False, action='store_true',
                        help="Apply cosine normalization to the obtained kernel")
    parser.add_argument("--coords2sim_type", dest="coords2sim_type", default="dotProduct", help= "Select the type of transformation from coords to similarity: dotProduct, normalizedScaling, infinity and int or float numbers")
    parser.add_argument("-g", "--graph_file", dest="graph_file", default=None,
                        help="Build a graphic representation of the network")
    parser.add_argument("--graph_options", dest="graph_options", default={'method': 'elgrapho', 'layout': 'forcedir', 'steps': '30'}, type= graph_options_parse,
                        help="Set graph parameters as 'NAME1=value1,NAME2=value2,...")
    parser.add_argument("-T","--processes", dest="processes", default=2, type= int,
                        help="Number of processes to use in computation")
    parser.add_argument("-r","--reference_nodes", dest="reference_nodes", default=[], type= lambda x: single_split(x, sep=","),
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
    parser.add_argument("-d","--delete", dest="delete_nodes", default=[], type= lambda x: single_split(x, sep=";"),
                        help="Remove nodes from file. If PATH;r then nodes not included in file are removed")
    parser.add_argument("-x","--expand_clusters", dest="expand_clusters", default=None,
                        help="Method to expand clusters Available methods: sht_path")
    parser.add_argument("--output_expand_clusters", dest= "output_expand_clusters", default="expand_clusters.txt", help="outputname fopr expand clusters file")
    parser.add_argument("--one_sht_pairs", dest="one_sht_pairs", default=False, action='store_true',
                        help="add this flag if expand cluster needed with just one of the shortest paths")
    parser.add_argument("-M", "--group_metrics", dest="group_metrics", default=None, type= lambda x: single_split(x, sep=";"),
                        help="Perform group group_metrics")
    parser.add_argument("--output_metrics_by_cluster", dest="output_metrics_by_cluster", default='group_metrics.txt', help= "output name for metrics by cluster file")
    parser.add_argument("-S", "--summarize_metrics", dest="summarize_metrics", default=None, type= lambda x: single_split(x, sep=";"),
                        help="Summarize metrics from groups")
    parser.add_argument("--output_summarized_metrics", dest="output_summarized_metrics", default='group_metrics_summarized.txt', help= "output name for summarized metrics file")
    parser.add_argument("--seed", dest="seed", default=None, 
                        help="sepecify seed for clusterin processes")
    parser.add_argument("-A", "--attributes", dest="get_attributes", default=[], type =lambda x: single_split(x, sep=","),
                        help="String separated by commas with the name of network attribute")
    parser.add_argument("--graph_attributes", dest="get_graph_attributes", default=[], type =lambda x: single_split(x, sep=","),
                        help="String separated by commas with the name of network attribute")
    parser.add_argument("--attributes_summarize", dest="attributes_summarize", default= False, action = "store_true", help="If the attribtes needs to be obtained summarized") 
    parser.add_argument("--dsl_script", dest="dsl_script", default=None,
                        help="Path to dsl script to perform complex analysis")
    parser.add_argument("-O", "--ontology", dest="ontologies", default=[], type=lambda x: double_split(x, sep1=";",sep2=":"),
                        help="String that define which ontologies must be used with each layer. String definition:'layer_name1:path_to_obo_file1;layer_name2:path_to_obo_file2'")

    opts = parser.parse_args(args)
    main_netanalyzer(opts)
	
def randomize_clustering(args=None):
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
    parser.add_argument("-r", "--random_type", dest="random_type", default = ["size"], type = lambda x: single_split(x,sep=":"), 
                        help="Indicate random mode. 'size' for radomize clusters with the same size as input clusters. 'full_size' same as 'size' but all nodes are repaeted as same as input. 'fixed:n:s' for generate 'n' clusters of 's' nodes. Default = 'size'")
    parser.add_argument("-R", "--replacement", dest="replacement", default=False, action='store_true',
                        help="Boolean. Activates ramdom sampling with replacement. Sampling witout replacement will be executed instead.")  
    parser.add_argument("-o", "--output_file", dest="output_file", default= 'random_clusters.txt', 
                        help="Output file")
    parser.add_argument("-a", "--aggregate_sep", dest="aggregate_sep", default = None, 
                        help="This option activates aggregation in output. Separator character must be provided")
    parser.add_argument("-d", "--seed", dest="seed", default= 123, type=int, 
            help="Allows to set a seed for the randomization process. Set to a number. Otherwise results are not reproducible.")

    opts = parser.parse_args(args)
    main_randomize_clustering(opts)

def randomize_network(args=None):
    parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')
    parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
            help="Input file to create networks for further analysis")
    parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
            help="Output file to save random network")
    parser.add_argument("-n", "--node_names_file", dest="node_file", default= None, 
            help="File with node names corresponding to the input matrix, only use when -f is set to bin or matrix.")
    parser.add_argument("-f", "--input_format", dest="input_format", default= 'pair', 
            help="Input file format: pair (default), bin, matrix")
    parser.add_argument("-s", "--split_char", dest="split_char", default= "\t", 
            help="Character for splitting input file. Default: tab")
    parser.add_argument("-l","--layers", dest="layers", default=['layer', '-'], type= lambda x: double_split(x, sep1=";",sep2=","),
            help="Layer definition on network: layer1name,regexp1;layer2name,regexp2...")
    parser.add_argument("-r", "--type_random", dest="type_random", default= None, 
            help="Randomized basis. 'nodes' Node-baseds randomize or 'links' Links-baseds randomize")
    parser.add_argument("-d", "--seed", dest="seed", default= None, 
            help="Allows to set a seed for the randomization process. Set to a number. Otherwise results are not reproducible.")
    
    opts = parser.parse_args(args)
    main_randomize_network(opts)

def ranker(args=None):
    parser = argparse.ArgumentParser(description='Get the ranks from a matrix similarity score and a list of seeds')
    parser.add_argument("-k", "--input_kernels", dest="kernel_file", default=None, 
    help="The roots from each kernel to integrate")
    parser.add_argument("-n", "--input_nodes", dest="input_nodes", default=None,
    help="The list of node for each kernel in lst format")
    parser.add_argument("-s", "--genes_seed", dest="genes_seed", default=None,
    help="The name of the gene to look for backups")
    parser.add_argument("-N","--normalize_matrix", dest="normalize_matrix", default= None,
    help="Select the type of normalization, options are: None, by_column, by_row, by_row_col")
    parser.add_argument("-p", "--propagate", dest="propagate", default = False, action = "store_true")
    parser.add_argument("--propagate_options", dest="propagate_options", default =  '"tolerance": 1e-5, "iteration_limit": 100, "with_restart": 0',
    help="Additional options for propagation methods. It must be defines as '\"opt_name1\" : value1, \"opt_name2\" : value2,...'")
    parser.add_argument("-S", "--seed_sep", dest="seed_sep", default=",",
    help="Separator of seed genes. Only use when -s point to a file")
    parser.add_argument("-f", "--filter", dest="filter", default=None,
    help="PATH to file with seed_name and genes to keep in output")
    parser.add_argument("-l", "--cross_validation", dest="cross_validation", default=False, 
    action='store_true', help="To activate cross validation")
    parser.add_argument("-K", "--k_fold", dest="k_fold", default=None, type=int, 
    help="Indicate the number of itrations needed, not used for leave one out cross validation (loocv)")
    parser.add_argument("-t", "--top_n", dest="top_n", default=None, type=int,
    help="Top N genes to print in output")
    parser.add_argument("--output_top", dest="output_top", default=None,
    help="File to save Top N genes")
    parser.add_argument("-o", "--output_name", dest="output_name", default="ranked_genes",
    help="PATH to file with seed_name and genes to keep in output")
    parser.add_argument("--type_of_candidates", dest="type_of_candidates", default=False, action="store_true",
    help="type of candidates to output in ranking list: all, new, seed.")
    parser.add_argument("--whitelist", dest="whitelist", default=None, type = open_whitelist, help= "File Path with the whitelist of nodes to take into account in the ranker process")

    # TODO: Add Threat section
    #parser.add_argument("-T", "--threads", dest="threads", default=0, type=based_0,
    # help="Number of threads to use in computation, one thread will be reserved as manager.")
    opts = parser.parse_args(args)
    main_ranker(opts)

def text2binary_matrix(args=None):
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
    parser.add_argument('-B', '--binarize', dest="binarize", default=None, type = float,
    	help='Binarize matrix changin x >= thr to one and any other to zero into matrix given')
    parser.add_argument('-c', '--cutoff', dest="cutoff", default=None, type = float,
    	help='Cutoff matrix values keeping just x >= and setting any other to zero into matrix given')
    parser.add_argument('-s', '--get_stats', dest="stats", default=None,
    	help='Get stats from the processed matrix')
    parser.add_argument('-O', '--output_type', dest="output_type", default='bin',
    	help='Set output format file. "bin" for binary (default) or "mat" for tabulated text file matrix')
    
    opts = parser.parse_args(args)
    main_text2binary_matrix(opts)

def main_integrate_kernels(options):
    kernels = Kernels()

    if not options.kernel_ids:
        options.kernel_ids = list(range(0,len(options.kernel_files)))
        options.kernel_ids = [str(k) for k in options.kernel_ids]

    # TODO: Consider adding more options to integrate to accept other formats
    if options.input_format == "bin":
        kernels.load_kernels_by_bin_matrixes(options.kernel_files, options.node_files, options.kernel_ids)
        kernels.create_general_index()

    if options.integration_type is not None:
        print(options.n_workers)
        kernels.integrate_matrix(options.integration_type, options.n_workers, options.symmetric)

    if options.output_matrix_file is not None:
        kernel, names = kernels.integrated_kernel
        np.save(options.output_matrix_file, kernel)

        with open(options.output_matrix_file +'.lst', 'w') as f:
            for name in names:
                f.write(name + "\n")  

def main_netanalyzer(options):
    print("Loading network data")
    opts = vars(options)
    # FRED: Remove this part of vars and modify the loads methods (Tlk wth PSZ)
    fullNet = Net_parser.load(opts)
    fullNet.set_compute_pairs(options.use_pairs, not options.no_autorelations)
    fullNet.threads = options.processes

    fullNet.reference_nodes = options.reference_nodes
    for ont_data in opts['ontologies']:
        layer_name, ontology_file_path = ont_data
        fullNet.link_ontology(ontology_file_path, layer_name)
        fullNet.ontologies

    if options.group_nodes:
        fullNet.set_groups(options.group_nodes)

    if options.delete_nodes:
        node_list = CmdTabs.load_input_data(options.delete_nodes[0])
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
                fullNet.generate_adjacency_matrix(
                    options.use_layers[0][0], options.use_layers[1][0])
            if not (options.use_layers[1][0], options.use_layers[0][1]) in fullNet.matrices["adjacency_matrices"]:
                fullNet.generate_adjacency_matrix(
                    options.use_layers[1][0], options.use_layers[0][1])

            fullNet.get_association_values(
                (options.use_layers[0][0], options.use_layers[1][0]),
                (options.use_layers[1][0], options.use_layers[0][1]),
                "transference")
        else:
            fullNet.get_association_values(
                options.use_layers[0],
                options.use_layers[1][0],
                options.meth)

        with open(options.assoc_file, 'w') as f:
            for val in fullNet.association_values[options.meth][:-1]:
                f.write("\t".join(map(str, val)) + "\n")
            f.write(
                "\t".join(map(str, fullNet.association_values[options.meth][-1])))
        print(f"End of analysis: {options.meth}")

        if options.control_file != None:
            with open(options.control_file, "r") as f:
                control = [control.append(line.rstrip().split("\t")) for line in f]
            Performancer.load_control(control)
            predictions = fullNet.association_values[options.meth]
            performance = Performancer.get_pred_rec(predictions)
            with open(options.performance_file, 'r') as f:
                f.write("\t".join(['cut', 'prec', 'rec', 'meth']) + "\n")
                for item in performance:
                    item.append(options['meth'])
                    f.write("\t".join(item) + "\n")

    if options.kernel is not None:
        # This allows inject custom arguments for each embedding method
        embedding_kwargs = eval('{' +options.embedding_add_options +'}')
        if len(options.use_layers) == 1:
            # we use only a layer to perform the kernel, so only one item it is selected.
            layers2kernel = (options.use_layers[0][0], options.use_layers[0][0])
        else:
            layers2kernel = tuple(options.use_layers[0])


        fullNet.get_kernel(layers2kernel, options.kernel, options.normalize_kernel,
                           options.coords2sim_type, embedding_kwargs, add_to_object=True)
        fullNet.write_kernel(layers2kernel, options.kernel, options.kernel_file)

    if options.graph_file is not None:
        options.graph_options['output_file'] = options.graph_file
        fullNet.plot_network(options.graph_options)

    # Group creation
    if options.build_cluster_alg is not None:
        clust_kwargs = eval('{' + options.build_clusters_add_options +'}')
        fullNet.discover_clusters(
            options.build_cluster_alg, clust_kwargs, **{'seed': options.seed})

        if options.output_build_clusters is None:
            options.output_build_clusters = options.build_cluster_alg + \
                '_' + 'discovered_clusters.txt'

        with open(options.output_build_clusters, 'w') as out_file:
            for cl_id, nodes in fullNet.group_nodes.items():
                for node in nodes:
                    out_file.write(f"{cl_id}\t{node}\n")

    # Group metrics by cluster.
    if options.group_metrics:
        fullNet.compute_group_metrics(
            output_filename=options.output_metrics_by_cluster, metrics=options.group_metrics)

    # Group metrics summarized.
    if options.summarize_metrics:
        fullNet.compute_summarized_group_metrics(
            output_filename=options.output_summarized_metrics, metrics=options.summarize_metrics)

    # Comparing Group Families (Two by now)
    if options.compare_clusters_reference is not None:
        res = fullNet.compare_partitions(options.compare_clusters_reference)
        print(str(res.score))

    # Group Expansion
    if options.expand_clusters is not None:
        expanded_clusters = fullNet.expand_clusters(
            options.expand_clusters, options.one_sht_pairs)
        with open(options.output_expand_clusters, 'w') as out_file:
            for cl_id, nodes in expanded_clusters.items():
                for node in nodes:
                    out_file.write(f"{cl_id}\t{node}\n")

    if len(options.get_attributes) > 0:
        fullNet.get_node_attributes(
            options.get_attributes, summary=options.attributes_summarize, output_filename="node_attributes.txt")
    if len(options.get_graph_attributes) > 0:
        fullNet.get_graph_attributes(
            options.get_graph_attributes, output_filename="graph_attributes.txt")

def main_randomize_clustering(options):
    options = vars(options)
    clusters = load_clusters(options)

    nodes = [ n for cl in clusters.values() for n in cl ] # Get list of nodes
    if options['random_type'][0] != "full_size": nodes = pxc.uniq(nodes)

    if "size" in options['random_type'][0] and len(options['random_type']) == 1:
        all_sizes = [ len(nodes) for cluster_id, nodes in clusters.items() ]
    elif options['random_type'][0] == "fixed" and len(options['random_type']) == 3:
        all_sizes = [int(options['random_type'][2])] * int(options['random_type'][2]) # cluster_size * n clusters

    random_clusters = random_sample(nodes, options['replacement'], all_sizes, options['seed']) 
    write_clusters(random_clusters, options['output_file'], options['aggregate_sep'])
     

def main_randomize_network(options):
    fullNet = Net_parser.load(vars(options)) 
    randomNet = fullNet.randomize_network(options.type_random, **{"seed":options.seed})

    with open(options.output_file, "w") as outfile:
        for e in randomNet.graph.edges:
            outfile.write(f"{e[0]}\t{e[1]}\n")

def main_ranker(options):
    ranker = Ranker()
    ranker.matrix = np.load(options.kernel_file)
    if options.normalize_matrix is not None:
        ranker.normalize_matrix(mode=options.normalize_matrix)
    ranker.load_nodes_from_file(options.input_nodes)
    if options.whitelist is not None:
        ranker.filter_matrix(options.whitelist)
    ranker.load_seeds(options.genes_seed, sep=options.seed_sep) # TODO: Add when 3 columns is needed for weigths
    options.filter is not None and ranker.load_references(options.filter, sep=",")
    print(options.propagate_options)
    propagate_options = eval('{' + options.propagate_options +'}')
    ranker.do_ranking(cross_validation=options.cross_validation, propagate=options.propagate,
                      k_fold=options.k_fold, options=propagate_options)
    rankings = ranker.ranking

    discarded_seeds = [seed_name for seed_name,
                       ranks in rankings.items() if not ranks]

    if discarded_seeds:
        with open(options.output_name + "_discarded", "w") as f:
            for seed_name in discarded_seeds:
                f.write(
                    f"{seed_name}\t{options.seed_sep.join(ranker.seeds[seed_name])}")

    if options.top_n is not None:
        top_n = ranker.get_top(options.top_n)
        if options.output_top is None:
            rankings = top_n
        else:
            write_ranking(options.output_top, top_n)

    if options.filter is not None:
        rankings = ranker.get_reference_ranks()

    if rankings:
        if options.type_of_candidates:
            for seed_name, rankings_by_seed in rankings.items():
                added_ranking_column = []
                for ranking in rankings_by_seed:
                    if ranking[0] in ranker.seeds[seed_name]:
                        ranking.insert(5, "seed")
                    else:
                        ranking.insert(5, "new")
                    added_ranking_column.append(ranking)
                rankings[seed_name] = added_ranking_column

        write_ranking(f"{options.output_name}_all_candidates", rankings)

def main_text2binary_matrix(options):
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
        matrix = pxc.filter_cutoff_mat(matrix, options.binarize)
        matrix = pxc.binarize_mat(matrix)

    if options.cutoff is not None and options.binarize is None:
        matrix = pxc.filter_cutoff_mat(matrix, options.cutoff)

    if options.stats is not None:
        stats = pxc.get_stats_from_matrix(matrix)
        with open(options.stats, 'w') as f:
            for row in stats:
                f.write("\t".join([str(item) for item in row]) + "\n")

    if options.output_type == 'bin':
        np.save(options.output_matrix_file, matrix)
    elif options.output_type == 'mat':
        np.savetxt(options.output_matrix_file, matrix, delimiter='\t')

# METHODS FOR NETANALYZER
#########################

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
               
# METHODS FOR RANDOMIZE CLUSTERING
###################################

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
    uniq_node_list = pxc.uniq(nodes)
    random.seed(seed)
    for counter, cluster_size in enumerate(all_sizes):
        if cluster_size > len(uniq_node_list) and not replacement: sys.exit("Not enough nodes to generate clusters. Please activate replacement or change random mode") 
        random_nodes = random.sample(uniq_node_list, cluster_size)
        if not replacement: uniq_node_list = [n for n in uniq_node_list if n not in random_nodes]
        random_clusters[f"{counter}_random"] = random_nodes
    return random_clusters

def write_clusters(clusters, output_file, sep): #2expcalc
	with open(output_file, "w") as outfile:
		for cluster, nodes in clusters.items():
			if sep != None: nodes = [sep.join(nodes)]
			for node in nodes:
				outfile.write(f"{cluster}\t{node}\n")
                        
# METHODS FOR RANKER
#####################

def write_ranking(file, ranking_list): #2expcalc
  with open(file, 'w') as f:
    for seed_name, ranking in ranking_list.items():
      for ranked_gene in ranking:
        f.write('\t'.join(map(str,ranked_gene)) + "\t" + f"{seed_name}" + "\n")     

def open_whitelist(file):
  whitelist = []
  with open(file, "r") as f:
    for line in f:
      node = line.strip()
      whitelist.append(node)
  return whitelist

# METHODS FOR TEXT2BINARY
#########################

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
		pxc.add_nested_record(connections, node_a, node_b, weight)
		pxc.add_nested_record(connections, node_b, node_a, weight)

	matrix, names = dicti2wmatrix_squared(connections)
	return matrix, names

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