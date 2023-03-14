import random
import sys 
import re
import copy
import networkx as nx
import math
import numpy
import scipy.stats as stats
import statsmodels
import itertools
from cdlib import algorithms, viz, evaluation
from NetAnalyzer.adv_mat_calc import Adv_mat_calc
from NetAnalyzer.net_plotter import Net_plotter
# https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
# http://mkivela.com/pymnet
class NetAnalyzer:

    def __init__(self, layers):
        self.graph = nx.Graph()
        self.layers = layers
        self.association_values = {}
        self.compute_autorelations = True
        self.compute_pairs = 'conn'
        self.adjacency_matrices = {}
        self.kernels = {}
        self.group_nodes = {} # Communities are networkx objects {community_id : networkx obj}
        self.reference_nodes = []

    def __eq__(self, other): # https://igeorgiev.eu/python/tdd/python-unittest-assert-custom-objects-are-equal/
        return nx.utils.misc.graphs_equal(self.graph, other.graph) and \
            self.layers == other.layers and \
            self.association_values == other.association_values and \
            self.compute_autorelations == other.compute_autorelations and \
            self.compute_pairs == other.compute_pairs and \
            self.adjacency_matrices == other.adjacency_matrices and \
            self.kernels == other.kernels and \
            self.group_nodes == other.group_nodes and \
            self.reference_nodes == other.reference_nodes

    def clone(self):
        network_clone = NetAnalyzer(copy.copy(self.layers))
        network_clone.graph = copy.deepcopy(self.graph)
        network_clone.association_values = self.association_values.copy()
        network_clone.set_compute_pairs(self.compute_pairs, self.compute_autorelations)
        network_clone.adjacency_matrices = self.adjacency_matrices.copy()
        network_clone.kernels = self.kernels.copy()
        network_clone.group_nodes = copy.deepcopy(self.group_nodes)
        network_clone.reference_nodes = self.reference_nodes.copy()
        return network_clone

    # THE PREVIOUS METHODS NEED TO DEFINE/ACCESS THE VERY SAME ATTRIBUTES, WATCH OUT ABOUT THIS !!!!!!!!!!!!!

    def set_compute_pairs(self, use_pairs, get_autorelations):
        self.compute_pairs = use_pairs
        self.compute_autorelations = get_autorelations

    def add_node(self, nodeID, layer):
        self.graph.add_node(nodeID, layer=layer)

    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    def set_layer(self, layer_definitions, node_name):
        layer = None
        if len(layer_definitions) > 1:
            for layer_name, regexp in layer_definitions:
                if re.search(regexp, node_name):
                    layer = layer_name
                    break
            if layer == None: raise Exception("The node '" + node_name + "' not match with any layer regex")
        else:
            layer = layer_definitions[0][0]
        if layer not in self.layers: self.layers.append(layer)
        return layer

    def generate_adjacency_matrix(self, layerA, layerB):
        layerAidNodes = [ node[0] for node in self.graph.nodes('layer') if node[1] == layerA]
        layerBidNodes = [ node[0] for node in self.graph.nodes('layer') if node[1] == layerB]
        matrix = numpy.zeros((len(layerAidNodes), len(layerBidNodes)))

        for i, nodeA in enumerate(layerAidNodes):
            for j, nodeB in enumerate(layerBidNodes):
                if nodeA in self.graph.neighbors(nodeB):
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0

        all_info_matrix = [matrix, layerAidNodes, layerBidNodes]

        if layerA == layerB:
            self.adjacency_matrices[(layerA)] = all_info_matrix
        else:
            self.adjacency_matrices[(layerA, layerB)] = all_info_matrix

        return all_info_matrix

    def delete_nodes(self, node_list, mode='d'):
        if mode == 'd':
            self.graph.remove_nodes_from(node_list)
        elif mode == 'r': # reverse selection
            self.graph.remove_nodes_from(list(n for n in self.graph.nodes if n not in node_list ))

    def get_connected_nodes(self, node_id, from_layer):
        return [n for n in self.graph.neighbors(node_id) if self.graph.nodes[n]['layer'] == from_layer ]

    def get_bipartite_subgraph(self, from_layer_node_ids, from_layer, to_layer):
        bipartite_subgraph = {}
        for from_layer_node_id in from_layer_node_ids:
            connected_nodes = self.graph.neighbors(from_layer_node_id)
            for connected_node in connected_nodes:
                if self.graph.nodes[connected_node]['layer'] == to_layer:
                    query = bipartite_subgraph.get(connected_node)
                    if query == None:
                        bipartite_subgraph[connected_node] = self.get_connected_nodes(connected_node, from_layer)
        return bipartite_subgraph

    def get_nodes_by_attr(self, attrib, value):
        return [nodeID for nodeID, attr in self.graph.nodes(data=True) if attr[attrib] == value]

    def get_nodes_layer(self, layers):
        nodes = []
        for layer in layers:
            nodes.extend(self.get_nodes_by_attr('layer', layer))
        return nodes

    def get_node_layer(self, node_id):
        return self.graph.nodes(data=True)[node_id]['layer']

    def get_edge_number(self):
        return len(self.graph.edges())

    def get_degree(self, zscore = True):
        degree = dict(self.graph.degree())
        if zscore:
            deg = numpy.array([d for n, d in degree.items()])
            deg_z = (deg - numpy.mean(deg)) / numpy.std(deg)
            degree_z = {}
            count = 0
            for n, d in degree.items():
                degree_z[n] = deg_z[count]
                count += 1
            degree = degree_z
        return degree

    def collect_nodes(self, layers = 'all'):
        nodeIDsA = []
        nodeIDsB = []
        if self.compute_autorelations:
            if layers == 'all':
                nodeIDsA = self.graph.nodes
            else:
                nodeIDsA = self.get_nodes_layer(layers)
        else:
            if layers != 'all': # layers contains two layer IDs
                nodeIDsA = self.get_nodes_layer([layers[0]])
                nodeIDsB = self.get_nodes_layer([layers[1]])
        return nodeIDsA, nodeIDsB

    def intersection(self, node1, node2):
        shared_nodes = nx.common_neighbors(self.graph, node1, node2)
        return shared_nodes

    def get_all_intersections(self, layers = 'all'):
        def _(node1, node2):
            node_intersection = self.intersection(node1, node2)
            return len(list(node_intersection))
        intersection_lengths = self.get_all_pairs(_, layers = layers)
        return intersection_lengths

    def connections(self, ids_connected_to_n1, ids_connected_to_n2):
        res = False
        if ids_connected_to_n1 != None and ids_connected_to_n2 != None and len(ids_connected_to_n1 & ids_connected_to_n2) > 0 : # check that at least exists one node that connect to n1 and n2
            res = True
        return res
    
    def get_all_pairs(self, pair_operation = None , layers = 'all'):
        all_pairs = []
        nodeIDsA, nodeIDsB = self.collect_nodes(layers = layers)
        if pair_operation != None:
            if self.compute_autorelations:
                node_list = [ n for n in nodeIDsA]
                while len(node_list) > 0:
                    node1 = node_list.pop(0)
                    if self.compute_pairs == 'all':
                        for node2 in node_list:
                            res = pair_operation(node1, node2)
                            all_pairs.append(res)
                    elif self.compute_pairs == 'conn':
                        ids_connected_to_n1 = set(self.graph.neighbors(node1))
                        for node2 in node_list:
                            ids_connected_to_n2 = set(self.graph.neighbors(node2))
                            if self.connections(ids_connected_to_n1, ids_connected_to_n2):
                                res = pair_operation(node1, node2)
                                all_pairs.append(res)
            else:
                if self.compute_pairs == 'conn': #MAIN METHOD
                    for node1 in nodeIDsA:
                        ids_connected_to_n1 = set(self.graph.neighbors(node1))
                        for node2 in nodeIDsB:
                            ids_connected_to_n2 = set(self.graph.neighbors(node2))
                            if self.connections(ids_connected_to_n1, ids_connected_to_n2):
                                res = pair_operation(node1, node2)
                                all_pairs.append(res)
                elif self.compute_pairs == 'all':
                    raise NotImplementedError('Not implemented')

        return all_pairs

    ## association methods adjacency matrix based
    #---------------------------------------------------------

    def clean_autorelations_on_association_values(self):
        for meth, values in self.association_values.items():
            self.association_values[meth] = [relation for relation in values if self.graph.nodes[relation[0]]["layer"] != self.graph.nodes[relation[1]]["layer"]]

    def get_association_values(self, layers, base_layer, meth):
        relations = [] #node A, node B, val
        if meth == 'counts':
            relations = self.get_counts_associations(layers, base_layer)
        elif meth == 'jaccard': #all networks
            relations = self.get_jaccard_associations(layers, base_layer)
        elif meth == 'simpson': #all networks
            relations = self.get_simpson_associations(layers, base_layer)
        elif meth == 'geometric': #all networks
            relations = self.get_geometric_associations(layers, base_layer)
        elif meth == 'cosine': #all networks
            relations = self.get_cosine_associations(layers, base_layer)
        elif meth == 'pcc': #all networks
            relations = self.get_pcc_associations(layers, base_layer)
        elif meth == 'hypergeometric': #all networks
            relations = self.get_hypergeometric_associations(layers, base_layer)
        elif meth == 'hypergeometric_bf': #all networks
            relations = self.get_hypergeometric_associations(layers, base_layer, pvalue_adj_method = 'bonferroni')
        elif meth == 'hypergeometric_bh': #all networks
            relations = self.get_hypergeometric_associations(layers, base_layer, pvalue_adj_method = 'benjamini_hochberg')
        elif meth == 'csi': #all networks
            relations = self.get_csi_associations(layers, base_layer)
        elif meth == 'transference': #tripartite networks
            relations = self.get_association_by_transference_resources(layers, base_layer)
        return relations

    def get_association_by_transference_resources(self, firstPairLayers, secondPairLayers, lambda_value1 = 0.5, lambda_value2 = 0.5):
        relations = []
        matrix1 = self.adjacency_matrices[firstPairLayers][0]
        matrix2 = self.adjacency_matrices[secondPairLayers][0]
        finalMatrix = Adv_mat_calc.tranference_resources(matrix1, matrix2, lambda_value1 = lambda_value1, lambda_value2 = lambda_value2)
        rowIds = self.adjacency_matrices[firstPairLayers][1]
        colIds =  self.adjacency_matrices[secondPairLayers][2]
        relations = self.matrix2relations(finalMatrix, rowIds, colIds)
        self.association_values['transference'] = relations
        return relations

    def get_associations(self, layers, base_layer, compute_association): # BASE METHOD
        base_nodes = set(self.get_nodes_layer([base_layer]))
        def _(node1, node2): 
            associatedIDs_node1 = set(self.graph.neighbors(node1))
            associatedIDs_node2 = set(self.graph.neighbors(node2))
            intersectedIDs = (associatedIDs_node1 & associatedIDs_node2) & base_nodes
            associationValue = compute_association(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2)
            return [node1, node2, associationValue]  
        associations = self.get_all_pairs(_, layers = layers)
        return associations

    #https://stackoverflow.com/questions/55063978/ruby-like-yield-in-python-3
    def get_counts_associations(self, layers, base_layer):
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            return len(intersectedIDs)
        relations = self.get_associations(layers, base_layer, _)
        self.association_values['counts'] = relations
        return relations

    def get_jaccard_associations(self, layers, base_layer):
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            unionIDS = associatedIDs_node1 | associatedIDs_node2
            return len(intersectedIDs)/len(unionIDS)		
        relations = self.get_associations(layers, base_layer, _)
        self.association_values['jaccard'] = relations
        return relations

    def get_simpson_associations(self, layers, base_layer):
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            minLength = min([len(associatedIDs_node1), len(associatedIDs_node2)])
            return len(intersectedIDs)/minLength
        relations = self.get_associations(layers, base_layer, _)
        self.association_values['simpson'] = relations
        return relations

    def get_geometric_associations(self, layers, base_layer):
        #wang 2016 method
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            intersectedIDs = len(intersectedIDs)**2
            productLength = math.sqrt(len(associatedIDs_node1) * len(associatedIDs_node2))
            return intersectedIDs/productLength
        relations = self.get_associations(layers, base_layer, _)
        self.association_values['geometric'] = relations
        return relations

    def get_cosine_associations(self, layers, base_layer):
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            productLength = math.sqrt(len(associatedIDs_node1) * len(associatedIDs_node2))
            return len(intersectedIDs)/productLength
        relations = self.get_associations(layers, base_layer, _)
        self.association_values['cosine'] = relations
        return relations

    def get_pcc_associations(self, layers, base_layer):
        #for Ny calcule use get_nodes_layer
        base_layer_nodes = self.get_nodes_layer([base_layer])
        ny = len(base_layer_nodes)
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            intersProd = len(intersectedIDs) * ny
            nodesProd = len(associatedIDs_node1) * len(associatedIDs_node2)
            nodesSubs = intersProd - nodesProd
            nodesAInNetwork = ny - len(associatedIDs_node1)
            nodesBInNetwork = ny - len(associatedIDs_node2)
            return numpy.float64(nodesSubs) / math.sqrt(nodesProd * nodesAInNetwork * nodesBInNetwork) # TODO: numpy.float64 is used to handle division by 0. Fix the implementation/test to avoid this case
        relations = self.get_associations(layers, base_layer, _)
        self.association_values['pcc'] = relations
        return relations

    def get_csi_associations(self, layers, base_layer):
        pcc_relations = self.get_pcc_associations(layers, base_layer)
        pcc_relations = [row for row in pcc_relations if not math.isnan(row[2])]
        if len(layers) > 1:
            self.clean_autorelations_on_association_values()

        nx = len(self.get_nodes_layer(layers))
        pcc_vals = {}
        node_rels = {}

        for node1, node2, assoc_index in pcc_relations:
            self.add_nested_record(pcc_vals, node1, node2, numpy.abs(assoc_index))
            self.add_nested_record(pcc_vals, node2, node1, numpy.abs(assoc_index))
            self.add_record(node_rels, node1, node2)
            self.add_record(node_rels, node2, node1)
        
        relations = []
        for node1, node2, assoc_index in pcc_relations:
            pccAB = assoc_index - 0.05
            valid_nodes = 0

            significant_nodes_from_node1 = set([node for node in node_rels[node1] if pcc_vals[node1][node] >= pccAB])
            significant_nodes_from_node2 = set([node for node in node_rels[node2] if pcc_vals[node2][node] >= pccAB])
            all_significant_nodes = significant_nodes_from_node2 | significant_nodes_from_node1
            all_nodes = set(node_rels[node1]) | set(node_rels[node2])
            
            csiValue = 1 - (len(all_significant_nodes))/(len(all_nodes)) 		
            relations.append([node1, node2, csiValue])
        
        self.association_values['csi'] = relations
        return relations

    def get_hypergeometric_associations(self, layers, base_layer, pvalue_adj_method= None):
        ny = len(self.get_nodes_layer([base_layer]))
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            # Analogous formulation with stats.fisher_exact(data, alternative='greater')
            intersection_lengths = len(intersectedIDs)
            if intersection_lengths > 0:
                n1_items = len(associatedIDs_node1)
                n2_items = len(associatedIDs_node2)
                p_value = stats.hypergeom.sf(intersection_lengths-1, ny, n1_items, n2_items)

            return p_value
        relations = self.get_associations(layers, base_layer, _)

        if pvalue_adj_method == 'bonferroni':
            meth = 'hypergeometric_bf'
            self.adjust_pval_association(relations, 'bonferroni')
        elif pvalue_adj_method == 'benjamini_hochberg':
            meth = 'hypergeometric_bh'
            self.adjust_pval_association(relations, 'fdr_bh')
        else:
            meth = 'hypergeometric'
        relations = [[assoc[0], assoc[1], -numpy.log10(assoc[2])] for assoc in relations]
        self.association_values[meth] = relations
        return relations

    def adjust_pval_association(self, associations, method): # TODO TEST
        pvals = numpy.array([val[2] for val in associations])
        adj_pvals = statsmodels.stats.multitest.multipletests(pvals, method=method, is_sorted=False, returnsorted=False)[1]
        for idx, adj_pval in enumerate(adj_pvals):
            associations[idx][2] = adj_pval

    def get_kernel(self, layer2kernel, kernel, normalization=False):
        matrix, node_names_x, node_names_y = self.adjacency_matrices[layer2kernel]
        matrix_result = Adv_mat_calc.get_kernel(matrix, node_names_x, kernel, normalization=normalization)
        self.kernels[layer2kernel] = matrix_result

    def write_kernel(self, layer2kernel, output_file):
        numpy.save(output_file, self.kernels[layer2kernel])

    def shortest_path(self, source, target):
        return nx.shortest_path(self.graph, source, target)

    def average_shortest_path_length(self, community):
        return nx.average_shortest_path_length(community)

    def shortest_paths(self, community):
        return nx.all_pairs_shortest_path(community)

    def get_node_attributes(self, attr_names):
        attrs = []
        for attr_name in attr_names:
            if attr_name == 'get_degree':
                attrs.append(self.get_degree(zscore=False))
            elif attr_name == 'get_degreeZ':
                attrs.append(self.get_degree())
        node_ids = attrs[0].keys()
        node_attrs = []
        for n in node_ids:
            n_attrs = [ at[n] for at in attrs ]
            node_attrs.append([n] + n_attrs)
        return node_attrs

    def plot_network(self, options = {}):
        net_data = {
            'group_nodes': self.group_nodes,
            'reference_nodes': self.reference_nodes,
            'graph': self.graph,
            'layers': self.layers
        }
        Net_plotter(net_data, options)

    # COMMUNITY METHODS

    # Cluster (community) dicovery

    def discover_clusters(self, cluster_method, cluster_additional_options):
        communities = self.get_clusters_by_algorithm(cluster_method, cluster_additional_options)
        communities = { cluster_method + "_" + str(idx): self.graph.subgraph(community) for idx, community in enumerate(communities)}
        self.group_nodes.update(communities) # If external added, thay will not be removed!

    def get_clusters_by_algorithm(self, cluster_method, cluster_additional_options):
        exec('clust_kwargs = {' + options.additional_options +'}') # This allows inject custom arguments for each clustering method
        if(cluster_method == 'leiden'):
            communities = algorithms.leiden(g, weights='weight', **clust_kwargs)
        elif(cluster_method == 'louvain'):
            communities = algorithms.louvain(g, weight='weight', **clust_kwargs)
        elif(cluster_method == 'cpm'):
            communities = algorithms.cpm(g, weights='weight', **clust_kwargs)
        elif(cluster_method == 'der'):
            communities = algorithms.der(g, **clust_kwargs)
        elif(cluster_method == 'edmot'):
            communities = algorithms.edmot(g, **clust_kwargs)
        elif(cluster_method == 'eigenvector'):
            communities = algorithms.eigenvector(g, **clust_kwargs)
        elif(cluster_method == 'gdmp2'):
            communities = algorithms.gdmp2(g, **clust_kwargs)
        elif(cluster_method == 'greedy_modularity'):
            communities = algorithms.greedy_modularity(g, weight='weight', **clust_kwargs)
        elif(cluster_method == 'label_propagation'):
            communities = algorithms.label_propagation(g, **clust_kwargs)
        elif(cluster_method == 'markov_clustering'):
            communities = algorithms.markov_clustering(g, **clust_kwargs)
        elif(cluster_method == 'rber_pots'):
            communities = algorithms.rber_pots(g, weights='weight', **clust_kwargs)
        elif(cluster_method == 'rb_pots'):
            communities = algorithms.rb_pots(g, weights='weight', **clust_kwargs)
        elif(cluster_method == 'significance_communities'):
            communities = algorithms.significance_communities(g, **clust_kwargs)
        elif(cluster_method == 'spinglass'):
            communities = algorithms.spinglass(g, **clust_kwargs)
        elif(cluster_method == 'surprise_communities'):
            communities = algorithms.surprise_communities(g, **clust_kwargs)
        elif(cluster_method == 'walktrap'):
            communities = algorithms.walktrap(g, **clust_kwargs)
        elif(cluster_method == 'lais2'):
            communities = algorithms.lais2(g, **clust_kwargs)
        elif(cluster_method == 'big_clam'):
            communities = algorithms.big_clam(g, **clust_kwargs)
        elif(cluster_method == 'danmf'):
            communities = algorithms.danmf(g, **clust_kwargs)
        elif(cluster_method == 'ego_networks'):
            communities = algorithms.ego_networks(g, **clust_kwargs)
        elif(cluster_method == 'egonet_splitter'):
            communities = algorithms.egonet_splitter(g, **clust_kwargs)
        elif(cluster_method == 'nmnf'):
            communities = algorithms.nmnf(g, **clust_kwargs)
        elif(cluster_method == 'nnsed'):
            communities = algorithms.nnsed(g, **clust_kwargs)
        elif(cluster_method == 'slpa'):
            communities = algorithms.slpa(g, **clust_kwargs)
        elif(cluster_method == 'bimlpa'):
            communities = actlgorithms.bimlpa(g, **clust_kwargs)
        elif(cluster_method == 'wcommunity'):
            communities = algorithms.wCommunity(g, **clust_kwargs)
        elif(cluster_method == 'aslpaw'):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                communities = algorithms.aslpaw(g)

        return communities.communities # To return a list of list with each of the nodes names for each communities.
    # Metrics
    def compute_comparative_degree(self, com): # see Girvan-Newman Benchmark control parameter in http://networksciencebook.com/chapter/9#testing (communities chapter)
        internal_degree = 0
        external_degree = 0
        com_nodes = set(com.nodes)
        for nodeID in com_nodes:
            nodeIDneigh = set(self.graph.neighbors(nodeID))
            if nodeIDneigh == None: next
            internal_degree += len(nodeIDneigh & com_nodes)
            external_degree += len(nodeIDneigh - com_nodes)
        comparative_degree = external_degree / (external_degree + internal_degree)
        return comparative_degree
    
    def compute_node_com_assoc(self, com, ref_node):
        ref_edges = 0
        ref_secondary_edges = 0
        secondary_nodes = {}
        other_edges = 0
        other_nodes = {}

        refNneigh = set(self.graph.neighbors(ref_node))
        for nodeID in com.nodes:
            nodeIDneigh = set(self.graph.neighbors(nodeID))
            if nodeIDneigh == None: next
            if ref_node in nodeIDneigh: ref_edges += 1
            if refNneigh != None:
                common_nodes = nodeIDneigh & refNneigh
                for id in common_nodes: secondary_nodes[id] = True
                ref_secondary_edges += len(common_nodes) 
            specific_nodes = nodeIDneigh - refNneigh - {ref_node}
            for id in specific_nodes: other_nodes[id] = True
            other_edges += len(specific_nodes)
        by_edge = (ref_edges + ref_secondary_edges) / other_edges
        by_node = (ref_edges + len(secondary_nodes)) / len(other_nodes)
        return [by_edge, by_node]

    # Iterative community methods
    def communities_avg_sht_path(self, coms):
        return [ self.average_shortest_path_length(com) for com_id, com in coms.items()]

    def communities_comparative_degree(self, coms):
        return [ self.compute_comparative_degree(com) for com_id, com in coms.items()]

    def communities_node_com_assoc(self, coms, ref_node):
        return [ self.compute_node_com_assoc(com, ref_node) for com_id, com in coms.items()]

    def compute_summarized_group_metrics(self, output_filename, metrics = ['size', 'avg_transitivity', 'internal_edge_density',
     'conductance', 'triangle_participation_ratio', 'max_odf', 'avg_odf', 'avg_embeddedness', 'average_internal_degree','cut_ratio',
     'fraction_over_median_degree', 'scaled_density']):
        communities = NodeClustering(list(self.group_nodes.values()), self.graph, "external", overlap=True)
        results = []
        for metric in metrics:
            # https://www.kite.com/python/answers/how-to-call-a-function-by-its-name-as-a-string-in-python
            class_method = getattr(evaluation, metric)
            res = class_method(graph, communities)
            results.append(res)

        with open(output_filename, 'w') as out_file:
            count = 0
            for res in results:
                metric_name = metrics[count]
                f.write("\t".join([metric_name, str(res.score), str(res.max), str(res.min), str(res.std)]) + "\n")
                count += 1

    def compute_group_metrics(self, output_filename, metrics = ['comparative_degree', 'avg_sht_path', 'node_com_assoc']): #metics by each clusters
        output_metrics = [[k] for k in self.group_nodes.keys()]
        header = ['group']

        for metric in metrics: 
            self.add_metrics(header, output_metrics, metric)

        with open(output_filename, 'w') as out_file:
            out_file.write("\t".join(header) + "\n")
            for line in output_metrics:
                out_file.write("\t".join(list(map(str, line))) + "\n")

    def add_metrics(self, header, output_metrics, metric):
        # Fusion of cdlib stats methods with NetAnalyzer "original" methods.
        if metric == 'comparative_degree':
            comparative_degree = self.communities_comparative_degree(self.group_nodes)
            for i, val in enumerate(comparative_degree): output_metrics[i].append(self.replace_none_vals(val)) # Add to metrics
            header.append(metric)
        elif metric == 'avg_sht_path': 
            avg_sht_path = self.communities_avg_sht_path(self.group_nodes)
            for i, val in enumerate(avg_sht_path): output_metrics[i].append(self.replace_none_vals(val)) # Add to metrics
            header.append(metric)
        elif metric == 'node_com_assoc':
            if len(self.reference_nodes) > 0:
                header.extend(['node_com_assoc_by_edge', 'node_com_assoc_by_node'])
                node_com_assoc = self.communities_node_com_assoc(self.group_nodes, self.reference_nodes[0]) # Assume only obe reference node
                for i, val in enumerate(node_com_assoc): output_metrics[i].extend(val) # Add to metrics
        else:
            # https://www.kite.com/python/answers/how-to-call-a-function-by-its-name-as-a-string-in-python
            communities = NodeClustering(list(self.group_nodes.values()), self.graph, "external", overlap=True)
            class_method = getattr(evaluation, metric)
            res = class_method(self.graph, communities, summary=False)
            for i, val in enumerate(res): output_metrics[i].append(self.replace_none_vals(val))
            header.append(metric)

    def expand_clusters(self, expand_method):
        clusters = {}
        for id, com in self.group_nodes.items():
            if expand_method == 'sht_path':
                new_nodes = set()
                all_sht_paths = (nx.all_shortest_paths(self.graph, NodeA, NodeB) for NodeA, NodeB in itertools.combinations(com.nodes, 2))
                for node_pair_sht_paths in all_sht_paths:
                    for path in node_pair_sht_paths:
                        new_nodes = new_nodes.union(set(path))
                com.add_nodes_from(list(new_nodes))
                clusters[id] = com
        return clusters

    ## RAMDOMIZATION METHODS
    ############################################################
    def randomize_monopartite_net_by_nodes(self):
        nodeIds = list(self.graph.nodes)
        random.shuffle(nodeIds)
        new_mapping = dict(zip(self.graph.nodes, nodeIds))
        random_network = self.clone() # TODO # Change to new instance with only an empty graph and layers defined
        random_network.graph = nx.relabel_nodes(self.graph, new_mapping)
        return random_network

    def randomize_monopartite_net_by_links(self):
        source = []
        target = []
        weigth = []
        for e, datadict in self.graph.edges.items():
            source.append(e[0])
            target.append(e[1])
            w = datadict.get('weigth')
            if w != None: weigth.append(w)
        random.shuffle(target)
        random_network = self.clone() # TODO # Change to new instance with only an empty graph and layers defined
        random_network.graph.clear()
        for src in source:
            i = 0
            while src == target[i] or (random_network.graph.has_node(src) and target[i] in random_network.graph[src]):
                i += 1
            targ = target.pop(i)
            if len(weigth) > 0:
                random_network.graph.add_edge(src, targ, {'weigth' : weigth.pop()})
            else:
                random_network.graph.add_edge(src, targ)
        return random_network


    def randomize_network(self, random_type):
        if random_type == 'nodes':
            if len(self.layers) == 1:
                random_network = self.randomize_monopartite_net_by_nodes()
            elif len(self.layers) == 2:
                random_network = self.randomize_bipartite_net_by_nodes()
        elif random_type == 'links':
            if len(self.layers) == 1:
                random_network = self.randomize_monopartite_net_by_links()
            elif len(self.layers) == 2:
                random_network = self.randomize_bipartite_net_by_links()
        else:
            raise(f"ERROR: The randomization is not available for {random_type} types of nodes")
        return random_network

    ## AUXILIAR METHODS
    #######################################################################################

    def add_record(self, hash, node1, node2):
        query = hash.get(node1)
        if query is None:
            hash[node1] = [node2]
        else:
            query.append(node2)

    def matrix2relations(self,finalMatrix, rowIds, colIds):
        relations = []
        for rowPos, rowId in enumerate(rowIds):
            for colPos, colId in enumerate(colIds):
                associationValue = finalMatrix[rowPos, colPos]
                if associationValue >= 0: relations.append([rowId, colId, associationValue])
        return relations

    def add_nested_record(self, hash, node1, node2, val):
        query_node1 = hash.get(node1)
        if query_node1 is None:
            hash[node1] = {node2: val}
        else:
            query_node1[node2] = val

    def replace_none_vals(self, val):
        return 'NULL' if val == None else val
