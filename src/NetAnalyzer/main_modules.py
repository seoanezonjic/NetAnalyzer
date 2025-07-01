import sys
import os
import warnings
import numpy as np
import random
import copy
from multiprocessing import Process, Manager, Lock
from py_cmdtabs.cmdtabs import CmdTabs
import py_exp_calc.exp_calc as pxc
from py_report_html import Py_report_html
from NetAnalyzer import Net_parser, NetAnalyzer
from NetAnalyzer import Kernels
from NetAnalyzer import Ranker
#from NetAnalyzer import Graph2sim
from NetAnalyzer import Adv_mat_calc
from NetAnalyzer.performancer import Performancer
from NetAnalyzer.seed_parser import SeedParser
import networkx as nx
from scipy.sparse import bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dia_matrix, dok_matrix, lil_matrix
from scipy.sparse import save_npz

def main_net_explorer(options, test = False):
    # loading gene seeds.
    options = vars(options)
    if options["seed_nodes"]: seeds2explore, _ = SeedParser.load_nodes_by_group(options["seed_nodes"], sep=options["seed_sep"])
    if options["group_nodes"]: 
        groups2explore = load_clusters(options)
    else:
        groups2explore = None

    # Loading multinet operations
    multinet = {}
    for net_id, net_file in options["input_file"].items():
        options_single_net = {'input_format': options['input_format'], 'input_file': net_file,
                              'node_files': [options['node_files'][net_id]] if options['node_files'] else None,
                              'layers': [['layer', '-']],
                              'split_char': options['split_char'],
                              'load_both': True}
        multinet[net_id] = Net_parser.load(options_single_net)
        cutoff = options["layer_cutoff"].get(net_id)
        if options["no_autorelations"]: 
            np.fill_diagonal(multinet[net_id].matrices['adjacency_matrices'][('layer','layer')][0], 0)
            multinet[net_id].adjMat2netObj('layer','layer')
        if cutoff:
            cutoff = float(cutoff)
            multinet[net_id].filter_matrix( mat_keys=('adjacency_matrices',('layer','layer')),operation='filter_cutoff', options={'cutoff': cutoff}, add_to_object=True)
            multinet[net_id].adjMat2netObj('layer','layer')

    # extract a subgraph for each
    if options["seed_nodes"]:
        seeds2subgraph = {}
        seeds2lcc = {}
        for seed, nodes in seeds2explore.items():
            seeds2subgraph[seed] = {}
            seeds2lcc[seed] = {}
            for net_id, net in multinet.items():
                # get neighbor from node
                nodes_with_neigh = set(nodes)
                if options["neigh_level"].get(net_id):
                    neigh_level = int(options["neigh_level"].get(net_id))
                    for i in range(0, neigh_level): nodes_with_neigh = get_neigh_set(net, nodes_with_neigh)
                    seeds2subgraph[seed][net_id] = net.graph.subgraph(nodes_with_neigh)
                else:
                    seeds2subgraph[seed][net_id] = net.graph
                largest_cc = len(max(nx.connected_components(seeds2subgraph[seed][net_id]), key=len))
                seeds2lcc[seed][net_id] = largest_cc

    # # If mention, add node2vec coordinates with a tnse proyection.
    net2embedding_proj = None
    if options["embedding_proj"]:
        from NetAnalyzer.graph2sim import Graph2sim # for optimization in loading times
        net2embedding_proj = {}
        for net_id, net in multinet.items():
            adj_mat, embedding_nodes, _ = net.matrices["adjacency_matrices"][("layer", "layer")]
            emb_coords = Graph2sim.get_embedding(adj_mat, embedding = "node2vec", embedding_nodes=embedding_nodes, embedding_kwargs= {'workers': options["threads"]})
            umap_coords = pxc.data2umap(emb_coords,  n_neighbors = 15, min_dist = 0.1, n_components = 2, metric = 'euclidean', random_seed = 123456)
            net2embedding_proj[net_id] = [umap_coords, embedding_nodes]

    # Comparing nets:
    net_sims = None
    if options["compare_nets"]:
        network_ids = list(options["input_file"].keys())
        num_nets = len(network_ids)
        sim_nets = np.zeros((num_nets,num_nets))
        for i, net_i in enumerate(network_ids):
            for j,net_j in enumerate(network_ids[i:len(network_ids)]):
                edges_i = set(multinet[net_i].graph.edges)
                edges_j = set(multinet[net_j].graph.edges)
                sim_i_j = len(edges_i & edges_j) / min(len(edges_i), len(edges_j))
                sim_nets[i,j+i] = sim_i_j
                sim_nets[j+i,i] = sim_i_j
        net_sims = [sim_nets, network_ids]
               
    # Execute the reports in the process.
    template = open(str(os.path.join(os.path.dirname(__file__), 'templates','net_explorer.txt'))).read()
    container = {"seeds2explore": seeds2explore, 
                "seeds2subgraph": seeds2subgraph, 
                "seeds2lcc":seeds2lcc, 
                "net2embedding_proj": net2embedding_proj, 
                "net_sims": net_sims, 
                "groups2explore": groups2explore,
                "plot_options": options["graph_options"]}

    report = Py_report_html(container, 'Network explorer')
    report.build(template)
    report.write(options["graph_file"]+".html")

    if test: return container

def get_neigh_set(net, nodes):
    neigh = set(nodes)
    for i,n in enumerate(nodes):
        try:
            neigh = neigh | set(net.graph.neighbors(n))
        except:
            continue
    return list(neigh)


def main_integrate_kernels(options):
    kernels = Kernels()

    if not options.kernel_ids:
        options.kernel_ids = list(range(0,len(options.kernel_files)))
        options.kernel_ids = [str(k) for k in options.kernel_ids]

    # TODO: Consider adding more options to integrate to accept other formats
    if options.input_format == "bin":
        kernels.load_kernels_by_bin_matrixes(options.kernel_files, options.node_files, options.kernel_ids)
        kernels.create_general_index()

    if not options.raw_values:
        kernels.move2zero_reference()

    if options.integration_type is not None:
        print(options.threads)
        kernels.integrate_matrix(options.integration_type, options.threads, options.symmetric)

    if options.output_file is not None:
        kernel, names = kernels.integrated_kernel
        np.save(options.output_file, kernel)

        with open(options.output_file +'.lst', 'w') as f:
            for name in names:
                f.write(name + "\n")  

def main_netanalyzer(options):
    print("Loading network data")
    opts = vars(options)
    # FRED: Remove this part of vars and modify the loads methods (Tlk wth PSZ)
    fullNet = Net_parser.load(opts)
    fullNet.set_compute_pairs(options.use_pairs, not options.no_autorelations)
    fullNet.threads = options.threads

    fullNet.reference_nodes = options.reference_nodes
    for ont_data in opts['ontologies']:
        layer_name, ontology_file_path = ont_data
        fullNet.link_ontology(ontology_file_path, layer_name)
        fullNet.ontologies

    if options.group_nodes:
        fullNet.set_groups(load_clusters(opts))
        if not fullNet.group_nodes:
            raise Exception("No nodes in the groups were detected")

    if options.delete_nodes:
        node_list = CmdTabs.load_input_data(options.delete_nodes[0])
        node_list = [item for sublist in node_list for item in sublist]
        mode = options.delete_nodes[1] if len(options.delete_nodes) > 1 else 'd'
        fullNet.delete_nodes(node_list, mode)

    if options.filter_connected_components:
        fullNet.delete_connected_component_by_size(options.filter_connected_components)

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

        if options.embedding_coords:
            fullNet.get_embedding_coords(layers2kernel, options.kernel, input_type = "matrix", embedding_kwargs=embedding_kwargs, output_filename=options.kernel_file)
        else:
            fullNet.get_kernel(layers2kernel, options.kernel, options.normalize_kernel,
                               options.coords2sim_type, embedding_kwargs, add_to_object=True)
            fullNet.write_kernel(layers2kernel, options.kernel, options.kernel_file)

    if options.graph_file is not None:
        options.graph_options['output_file'] = options.graph_file
        fullNet.plot_network(options.graph_options)

    # Group creation
    if options.build_cluster_alg is not None:
        clust_kwargs = eval('{' + options.build_clusters_add_options +'}')
        import networkx as nx
        if fullNet.group_nodes:
            first_to_add = True
            for group_id, nodes in fullNet.group_nodes.items():
                subNet = fullNet.clone()
                subNet.delete_nodes(nodes,"r")
                subNet.group_nodes = {}
                subNet.delete_isolated_nodes()
                if first_to_add:
                    discover_and_write_cluster(subNet, options.build_cluster_alg, clust_kwargs, options.seed, options.output_build_clusters, group_id)
                else:
                    discover_and_write_cluster(subNet, options.build_cluster_alg, clust_kwargs, options.seed, options.output_build_clusters, group_id, 'a')
                first_to_add = False
        else:
            discover_and_write_cluster(fullNet, options.build_cluster_alg, clust_kwargs, options.seed, options.output_build_clusters, None)

    # Group metrics by cluster.
    if options.group_metrics:
        fullNet.compute_group_metrics(
            output_filename=options.output_metrics_by_cluster, metrics=options.group_metrics)

    # Group metrics summarized.
    if options.summarize_metrics:
        fullNet.compute_summarized_group_metrics(
            output_filename=options.output_summarized_metrics, metrics=options.summarize_metrics)

    # Family clusters metric.
    if options.partition_metrics:
        partition_metrics = {}
        if options.external_metadata_cluster:
            if options.external_metadata_cluster.get("sim"):
                partition_metrics["community_quality"] = fullNet.community_quality(options.external_metadata_cluster["sim"])
            if options.external_metadata_cluster.get("metadata_classify"):
                partition_metrics["overlap_quality"] = fullNet.overlap_quality(options.external_metadata_cluster["metadata_classify"])
        if options.overlapping_communities:
            partition_metrics["overlap_coverage"] = fullNet.overlap_coverage()
            # partition_metrics["overlap_coverage_no_triads"] = fullNet.overlap_coverage(minimum_size=4)
        partition_metrics["community_coverage"] = fullNet.community_coverage()
        partition_metrics["community_coverage_no_triads"] = fullNet.community_coverage(minimum_size=4)
        partition_metrics["triad_number"]=fullNet.get_triad_numbers()
        for metric_name, metric_value in partition_metrics.items():
            print(f"{metric_name}\t{metric_value}")

    # Comparing Group Families (Two by now)
    if options.compare_clusters_reference is not None:
        fullNet.group_reference = options.compare_clusters_reference
        fullNet.join_clusters(options.compare_clusters_join)
        results = fullNet.compare_partitions(options.overlapping_communities)
        for metric_name, metric_value in results.items():
            print(f"{metric_name}\t{metric_value}")

    # Group Expansion
    if options.expand_clusters is not None:
        expanded_clusters = fullNet.expand_clusters(
            options.expand_clusters, options.one_sht_pairs)
        with open(options.output_expand_clusters, 'w') as out_file:
            for cl_id, nodes in expanded_clusters.items():
                for node in nodes:
                    out_file.write(f"{cl_id}\t{node}\n")

    if options.extract_subgraphs:
        fullNet.write_subgraphs_from_communities()

    if len(options.get_attributes) > 0:
        fullNet.get_node_attributes(
            options.get_attributes, summary=options.attributes_summarize, output_filename="node_attributes.txt")
    if len(options.get_graph_attributes) > 0:
        fullNet.get_graph_attributes(
            options.get_graph_attributes, output_filename="graph_attributes.txt")

    if options.output_network:
        fullNet.write_network(options.output_network)

def main_randomize_clustering(options):
    options = vars(options)
    if options["seed"]: random.seed(options["seed"])
    cluster2nodes = load_clusters(options)
    cluster2nodes = {cluster: pxc.uniq(nodes) for cluster, nodes in cluster2nodes.items()} # ensuring a cluster doesnt cotain replicates.
    random_clusters = {}
    # Reverse should go to expcalc
    node2clusters = {}
    for cluster, nodes in cluster2nodes.items():
        for node in nodes:
            if not node2clusters.get(node):
                node2clusters[node] = [cluster]
            else:
                node2clusters[node].append(cluster) 
    node2clusters = list(node2clusters.items())
    node2clusters.sort(key=lambda x: -len(x[1]))
    if options["random_type"][0] == "hard_fixed" or options["random_type"] == "soft_fixed":
        # Setting universe com and exiled:
        cluster_universe = list(cluster2nodes.keys())
        for node, clusters in node2clusters:
            if options["random_type"][0] == "hard_fixed":
                number_clust = len(clusters)
                if len(set(clusters)) < number_clust: raise Exception("A node is being defined two or more times to the same cluster")
            elif options["random_type"][1] == "soft_fixed":
                u_size = len(cluster_universe)
                k = len(pxc.intersection(clusters,cluster_universe))
                p = k/u_size
                number_clust = random.binomial(u_size, p) 
            sampled_clusters = random.sample(cluster_universe, number_clust)
            for clust in sampled_clusters:
                if random_clusters.get(clust):
                    random_clusters[clust].append(node)
                else:
                    random_clusters[clust] = [node]
                if len(random_clusters[clust]) == len(cluster2nodes[clust]): cluster_universe.remove(clust)
    elif options["random_type"][0] == "not_fixed":
        all_nodes = [ n for cl in cluster2nodes.values() for n in cl ] 
        uniq_nodes = pxc.uniq(all_nodes)
        if len(uniq_nodes) < len(all_nodes):
            for cluster, nodes in cluster2nodes.items():
                random_clusters[cluster] = random.sample(uniq_nodes, len(nodes))
        else:
            random.shuffle(all_nodes)
            i = 0
            for cluster, nodes in cluster2nodes:
                random_clusters[cluster] = all_nodes[i, i+len(nodes)]
                i += len(nodes)
    else:
        all_sizes = [int(options['random_type'][2])] * int(options['random_type'][1])
        all_nodes = [node for node, _ in node2clusters]
        random_clusters = random_sample(all_nodes, options['random_type'][3] == "r", all_sizes, options['seed']) 

    write_clusters(random_clusters, options['output_file'], options['aggregate_sep'])
     

def main_randomize_network(options):
    fullNet = Net_parser.load(vars(options)) 
    randomNet = fullNet.randomize_network(options.type_random, **{"seed":options.seed})

    with open(options.output_file, "w") as outfile:
        for e in randomNet.graph.edges:
            outfile.write(f"{e[0]}\t{e[1]}\n")

def worker_ranker(seed_groups, seed_weight, opts, nodes, all_rankings, matrix, lock):
    ranker = Ranker()
    if opts["seed_presence"] == "remove":
        ranker.seed_presence = False
    ranker.nodes = nodes
    for sg_id, sg in seed_groups: ranker.seeds[sg_id] = sg
    for sg_id, ws in seed_weight:
        if ws is not None: ranker.weights[sg_id] = ws
    # LOAD KERNEL
    #load_kernel(ranker, opts)
    ranker.matrix = matrix
    # DO RANKING
    propagate_options = eval('{' + opts["propagate_options"] +'}')
    ranker.do_ranking(cross_validation=opts.get("cross_validation"), propagate=opts["propagate"],
                      k_fold=opts.get("k_fold"), metric = opts["representation_seed_metric"], options=propagate_options)

    with lock: # lock avoids that several processes write at same time in the dictionary
        data_package = {} # Create chunks of results to reduce using too much RAM in pickle process and process piping overload
        added_records = 0
        for key, vals in ranker.ranking.items():
            data_package[key] = vals
            added_records += len(vals)
            if added_records > 10000:
                all_rankings.update(data_package)
                data_package = {} 
        if len(data_package) > 0: all_rankings.update(data_package) # Write buffered records not writed during loop execution


def load_kernel(ranker, opts):
    ranker.matrix = np.load(opts["kernel_files"])
    if opts.get('normalize_matrix') is not None:
        ranker.normalize_matrix(mode=opts["normalize_matrix"])
    if opts.get('whitelist') is not None:
        ranker.filter_matrix(opts["whitelist"])
        ranker.clean_seeds()

def sort_records_by_load(records):
    recs = []
    slices = int(chunk_size/2)
    r = chunk_size % 2
    while len(records) > 0:
        recs.append(records.pop(0))
        if len(records) > 0: recs.append(records.pop())
    return recs

def main_ranker(options):
    # LOAD RANKER
    ranker = Ranker()
    if options.seed_presence == "remove": # TODO: Probably, this is not necessary right here but on worker_ranker
        ranker.seed_presence = False
    # LOAD SEEDS
    ranker.load_nodes_from_file(options.node_files)
    ranker.load_seeds(options.seed_nodes, sep=options.seed_sep) # TODO: Add when 3 columns is needed for weigths

    ranker.clean_seeds(options.minimum_size)
    discarded_seeds = [ [seed_name, seed] for seed_name, seed in ranker.discarded_seeds.items()]
    if discarded_seeds:
        with open(options.output_file + "_discarded", "w") as f:
            for seed_name, seed in discarded_seeds: f.write(f"{seed_name}\t{options.seed_sep.join(seed)}"+"\n")
    load_kernel(ranker, vars(options))
    # LOAD TRAINING DATASET
    if options.score2pvalue:
        if options.score2pvalue == "logistic":
            if options.training_dataset:
                ranker.load_training_dataset(options.training_dataset)
            else:
                ranker.network = Net_parser.load_network_by_bin_matrix(options.adj_matrix, [options.node_files], [['layer', '-']])
                ranker.network.adjMat2netObj('layer','layer')
                ranker.generate_training_dataset("edges")
        ranker.score2pvalue_matrix(options.score2pvalue)
        
    # DO PARALLEL RANKING
    chunk_size = int(len(ranker.seeds)/options.threads)
    seeds = list(ranker.seeds.items())
    opts = vars(options)
    lock = Lock()
    if options.cross_validation and options.k_fold is None:
        header = ["candidates", "score", "normalized_rank", "rank"]
    else:
        header = ["candidates", "score", "normalized_rank", "rank", "uniq_rank"]

    with Manager() as manager:
        all_rankings = manager.dict()
        processes = []
        if options.threads > 1:
            worker_threads = options.threads - 1
            seeds.sort(reverse=True, key=lambda x: len(x[1]))
            seeds = sort_records_by_load(seeds)
        else:
            worker_threads = options.threads 
        for i in range(worker_threads):
            length = len(seeds)
            offset = length-chunk_size
            records = seeds[offset:length]
            seeds = seeds[0:offset]
            if len(seeds) < chunk_size: records.extend(seeds) # There is no enough record for other chunk so we merge the remanent records t this chunk
            records_weight = [ (record[0], ranker.weights.get(record[0])) for record in records ]
            p = Process(target=worker_ranker, args=(records, records_weight, opts, ranker.nodes, all_rankings, ranker.matrix, lock))
            processes.append(p)
            p.start()
        for p in processes: p.join() # first we have to start ALL the processes before ask to wait for their termination. For this reason, the join MUST be in an independent loop
        ranker.ranking.update(all_rankings) # COPY RESULTS OUT OF THE MEMORY MAP MANAGER!!!
        ranker.attributes["header"] = header

    # WRITE RANKING
    if options.seed_presence == "annotate": 
        ranker.add_candidate_tag_types()

    if options.top_n is not None:
        if options.output_top is None:
            ranker.ranking = ranker.get_top(options.top_n)
        else:
            ranker.write_ranking(options.output_top, add_header=options.header, top_n=options.top_n)

    if options.filter is not None:
        ranker.load_references(options.filter, sep=",")
        ranker.ranking = ranker.get_filtered_ranks_by_reference(cross_validation=options.cross_validation)

    if options.add_tags is not None:
        tags = load_tags(options.add_tags)
        if options.cross_validation:
            for seed, nodes in ranker.seeds.keys():
                tags[seed] = tags[seed.split("_")[0]]
        for seed, ranking_by_seed in ranker.ranking.items():
            ranker.ranking[seed] = pxc.add_tags(ranking_by_seed, tags[seed], (0,), default_tag = False)

    if ranker.ranking:
        ranker.write_ranking(f"{options.output_file}_all_candidates", add_header=options.header)

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
        pairs = load_pair_file(source)
        matrix, rowIds, colIds = pxc.pairs2matrix(pairs, symm=True)
        with open(options.output_file + "_rowIds" ".lst", 'w') as f:
            f.write("\n".join(rowIds))
        with open(options.output_file + "_colIds" ".lst", 'w') as f:
            f.write("\n".join(colIds))

    # bsr, coo, csc, csr, dia, dok, lil

    if options.input_type != "pair" and options.node_files:
        rowIds = Net_parser.load_input_list(options.node_files[0])
        colIds = Net_parser.load_input_list(options.node_files[1])

    source.close()
    if options.umap:
        matrix = pxc.data2umap(matrix, n_neighbors = 30, min_dist = 0.1, n_components = 2, 
        metric = 'euclidean', random_seed = 123)

    if options.coords2kernel:
        matrix = pxc.coords2sim(matrix, sim = options.coords2kernel)
    
    if options.normalize_by: 
        matrix = pxc.normalize_matrix(matrix, method=options.normalize_by)

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
        # bsr, coo, csc, csr, dia, dok, lil
        if options.sparse_type:
            if options.sparse_type == "bsr":
                sparse_matrix = bsr_matrix(matrix)
            elif options.sparse_type == "coo":
                sparse_matrix = coo_matrix(matrix)
            elif options.sparse_type == "csc":
                sparse_matrix = csc_matrix(matrix)
            elif options.sparse_type == "csr":
                sparse_matrix = csr_matrix(matrix)
            elif options.sparse_type == "dia":
                sparse_matrix = dia_matrix(matrix)
            elif options.sparse_type == "dok":
                sparse_matrix = dok_matrix(matrix)
            elif options.sparse_type == "lil":
                sparse_matrix = lil_matrix(matrix)
            save_npz(f"{options.output_file}.npz", sparse_matrix)
        else:
            np.save(options.output_file, matrix)
    elif options.output_type == 'matrix':
        np.savetxt(options.output_file, matrix, delimiter='\t')
    elif options.output_type == "pair":
        pairs = pxc.matrix2relations(matrix, rowIds, colIds, symm = False)
        with open(options.output_file, "w") as f:
            for pair in pairs:
                pair[2] = str(pair[2])
                f.write("\t".join(pair) + "\n")


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

def discover_and_write_cluster(net, cluster_alg, clust_kwargs, seed, output_build_clusters, group_id, type_write='w'):
        net.discover_clusters(
            cluster_alg, clust_kwargs, **{'seed': seed})

        if output_build_clusters is None:
            output_build_clusters = cluster_alg + \
                '_' + 'discovered_clusters.txt'

        with open(output_build_clusters, type_write) as out_file:
            for cl_id, nodes in net.group_nodes.items():
                for node in nodes:
                    if group_id:
                        out_file.write(f"{group_id}\t{cl_id}\t{node}\n")
                    else:
                        out_file.write(f"{cl_id}\t{node}\n")
               
# METHODS FOR RANDOMIZE CLUSTERING
###################################

def load_clusters(options):
    clusters = {}
    if os.path.isfile(options['group_nodes']):
        with open(options['group_nodes'], "r") as f:
            for line in f:
                line = line.rstrip().split("\t")
                cluster = line[options['group_cluster_index']]
                node = line[options['group_node_index']]
                if options.get('node_sep') != None: 
                    node = node.split(options['node_sep'])
                    clusters[cluster] = node
                else:
                    query = clusters.get(cluster)
                    if query == None: 
                        clusters[cluster] = [node] 
                    else:
                        query.append(node)
    else:
        for i, group in enumerate(options['group_nodes'].split(";")):
            clusters[i] = group.split(',')
    return clusters

def random_sample(nodes, replacement, all_sizes, seed):
    random_clusters = {}
    node_list = copy.deepcopy(nodes)  
    random.seed(seed)
    for counter, cluster_size in enumerate(all_sizes):
        if cluster_size > len(node_list) and not replacement: sys.exit("Not enough nodes to generate clusters. Please activate replacement or change random mode") 
        random_nodes = random.sample(node_list, cluster_size)
        if not replacement: 
            node_list = pxc.diff(node_list, random_nodes)
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

def open_whitelist(file):
  whitelist = []
  with open(file, "r") as f:
    for line in f:
      node = line.strip()
      whitelist.append(node)
  return whitelist

def load_tags(file):
    tags = {}
    with open(file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            pxc.add_nested_value(tags, (line[0],line[1]), line[2])
    return tags

# METHODS FOR TEXT2BINARY
#########################

def load_matrix_file(source, splitChar = "\t"):
    matrix = None
    counter = 0
    col_names = []
    row_names = []
    for line in source:
        line = line.strip()
        row = [float(c) for c in line.split(splitChar)]
        if matrix is None:
            matrix = np.zeros((len(row), len(row)))
        for i, val in enumerate(row):
            matrix[counter, i] = val    
        counter += 1

    return matrix

def load_pair_file(source): 
    # Not used byte_forma parameter
    pairs = []
    for line in source:
        line = line.strip().split("\t")
        if len(line) == 3:
            node_a, node_b, weight = line
            weight = float(weight) 
        else:
            node_a, node_b = line
            weight = 1.0
        pairs.append([node_a, node_b, weight])
    return pairs