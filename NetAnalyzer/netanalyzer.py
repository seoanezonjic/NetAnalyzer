import random
import sys 
import re
import copy
import networkx as nx
import math
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels
import itertools
import warnings
import logging
from cdlib import algorithms, viz, evaluation
from cdlib import NodeClustering
import py_semtools # For external_data
from py_semtools import Ontology
from NetAnalyzer.adv_mat_calc import Adv_mat_calc
from NetAnalyzer.net_plotter import Net_plotter
from NetAnalyzer.graph2sim import Graph2sim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
# http://mkivela.com/pymnet

class NestedDict(dict):
    def dig(self, *keys):
        try:
            for key in keys:
                self = self[key]
            return self
        except KeyError:
            return None



class NetAnalyzer:

    def __init__(self, layers):
        self.threads = 2
        self.graph = nx.Graph() # Talk with PSZ, problem with directed graphs on loading edges.
        self.layers = layers
        self.association_values = {}
        self.compute_autorelations = True
        self.compute_pairs = 'conn'
        self.matrices = NestedDict(
        {"adjacency_matrices": {}, # {layers => Mat, rowIds, colIds}
         "kernels": {},            # {layers => {method_type => Mat, rowIds, colIds}}
         "associations": {},       # {layers => {method_type => Mat, rowIds, colIds}}
         "semantic_sims": {},      # {layers => {method_type => Mat, rowIds, colIds}}
         })
        self.embedding_coords = {} 
        self.group_nodes = {} # Communities are lists {community_id : [Node1, Node2,...]}
        self.reference_nodes = []
        self.loaded_obos = []
        self.ontologies = []
        self.layer_ontologies = {}

    def __eq__(self, other): # https://igeorgiev.eu/python/tdd/python-unittest-assert-custom-objects-are-equal/
        return nx.utils.misc.graphs_equal(self.graph, other.graph) and \
            self.layers == other.layers and \
            self.association_values == other.association_values and \
            self.compute_autorelations == other.compute_autorelations and \
            self.compute_pairs == other.compute_pairs and \
            self.matrices == other.matrices and \
            self.embedding_coords == other.embedding_coords and \
            self.group_nodes == other.group_nodes and \
            self.reference_nodes == other.reference_nodes and \
            self.loaded_obos == other.loaded_obos and \
            self.ontologies == other.ontologies and \
            self.layer_ontologies == other.layer_ontologies

    def clone(self):
        network_clone = NetAnalyzer(copy.copy(self.layers))
        network_clone.graph = copy.deepcopy(self.graph)
        network_clone.association_values = self.association_values.copy()
        network_clone.set_compute_pairs(self.compute_pairs, self.compute_autorelations)
        network_clone.embedding_coords = self.embedding_coords.copy()
        network_clone.matrices = self.matrices.copy()
        network_clone.group_nodes = copy.deepcopy(self.group_nodes)
        network_clone.reference_nodes = self.reference_nodes.copy()
        network_clone.loaded_obos = self.loaded_obos.copy()
        network_clone.ontologies = self.ontologies.deepcopy() if self.ontologies != [] else []
        network_clone.layer_ontologies = self.layer_ontologies.deepcopy() if self.layer_ontologies != {} else {}
        return network_clone

    # THE PREVIOUS METHODS NEED TO DEFINE/ACCESS THE VERY SAME ATTRIBUTES, WATCH OUT ABOUT THIS !!!!!!!!!!!!!

    def set_compute_pairs(self, use_pairs, get_autorelations):
        self.compute_pairs = use_pairs
        self.compute_autorelations = get_autorelations

    def add_node(self, nodeID, layer):
        self.graph.add_node(nodeID, layer=layer)

    def add_edge(self, node1, node2, **attribs):
        self.graph.add_edge(node1, node2, **attribs) # Talk with PSZ, problem with directed graphs on loading edges.

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

    def set_groups(self, groups):
        for group_id, nodes in groups.items():
            for node in nodes:
                if node in self.graph.nodes:
                    if self.group_nodes.get(group_id) is None:
                        self.group_nodes[group_id] = [node]
                    else:
                        self.group_nodes[group_id].append(node)
                else:
                    #print("Group id: " + str(group_id) + " with member not in network:" + str(node), file=sys.stderr)
                    logging.warning("Group id: " + str(group_id) + " with member not in network: " + str(node))

    

    def generate_adjacency_matrix(self, layerA, layerB): 
        layerAidNodes = [ node[0] for node in self.graph.nodes('layer') if node[1] == layerA] 
        layerBidNodes = [ node[0] for node in self.graph.nodes('layer') if node[1] == layerB]
        matrix = np.zeros((len(layerAidNodes), len(layerBidNodes)))

        has_weight = 'weight' if nx.get_edge_attributes(self.graph, 'weight') else None

        if layerA == layerB:
            # The method biadjacency matrix for this cases, fill the triangular upper matrix.
            matrix_triu = np.array(nx.bipartite.biadjacency_matrix(self.graph, row_order=layerAidNodes, column_order=layerBidNodes, weight=has_weight, format='csr').todense())
            matrix_tril = np.array(nx.bipartite.biadjacency_matrix(self.graph, row_order=layerBidNodes, column_order=layerAidNodes, weight=has_weight, format='csr').todense())
            matrix = np.triu(matrix_triu) + np.tril(np.transpose(matrix_tril), k = -1)
        else:
            matrix = np.array(nx.bipartite.biadjacency_matrix(self.graph, row_order=layerAidNodes, column_order=layerBidNodes, weight=has_weight, format='csr').todense())

        all_info_matrix = [matrix, layerAidNodes, layerBidNodes]

        self.matrices["adjacency_matrices"][(layerA, layerB)] = all_info_matrix

        return all_info_matrix

    def generate_all_biadjs(self):
        for layerA, layerB in itertools.product(self.layers, self.layers):
            self.generate_adjacency_matrix(layerA, layerB)

    def adjMat2netObj(self, layerA, layerB):
        matrix, rowIds, colIds = self.matrices["adjacency_matrices"][(layerA, layerB)] 

        self.graph = nx.Graph()
        for rowId in rowIds: self.add_node(rowId, layerA)
        for colId in colIds: self.add_node(colId, layerB)

        for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds):
                        associationValue = matrix[rowPos, colPos]
                        if associationValue > 0: self.graph.add_edge(rowId, colId, weight=associationValue)
        return self.graph

    def delete_nodes(self, node_list, mode='d'):
        if mode == 'd':
            self.graph.remove_nodes_from(node_list)
        elif mode == 'r': # reverse selection
            self.graph.remove_nodes_from(list(n for n in self.graph.nodes if n not in node_list ))

    def get_connected_nodes(self, node_id, from_layer):
        return [n for n in self.graph.neighbors(node_id) if self.graph.nodes[n]['layer'] == from_layer ]

    def get_layers_as_dict(self, from_layers, to_layer):
        relations = {}
        from_nodes = self.get_nodes_layer(from_layers)
        for fr_node in from_nodes:
            relations[fr_node] = self.get_connected_nodes(fr_node, to_layer)
        return relations

    def link_ontology(self, ontology_file_path, layer_name):
        if ontology_file_path not in self.loaded_obos: #Load new ontology
            ontology = Ontology(file = ontology_file_path, load_file = True)
            ontology.precompute()
            ontology.threads = self.threads
            self.loaded_obos.append(ontology_file_path)
            self.ontologies.append(ontology)
        else: #Link loaded ontology to current layer
            ontology = self.ontologies[self.loaded_obos.index(ontology_file_path)]
        self.layer_ontologies[layer_name] = ontology

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
            degree = self.znormalize_dic_by_values(degree)
        return degree

    def get_betweenness_centrality(self, zscore = True):
        betweenness_centrality = dict(nx.betweenness_centrality(self.graph))
        if zscore:
            betweenness_centrality = self.znormalize_dic_by_values(betweenness_centrality)
        return betweenness_centrality


    def znormalize_dic_by_values(self, dic2znormalize):
        data = np.array([d for n, d in dic2znormalize.items()])
        data_z = Adv_mat_calc.zscore_normalize(data)
        znormalized_dic = {}
        count = 0
        for n, d in dic2znormalize.items():
            znormalized_dic[n] = data_z[count]
            count += 1
        dic2znormalize = znormalized_dic
        return dic2znormalize


    def collect_nodes(self, layers = 'all'):
        nodeIDsA = []
        nodeIDsB = []
        if self.compute_autorelations: # TODO: remove the compute_autorrelations attrib.
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
                node_list = [ n for n in nodeIDsA] # Is this conversion needed?
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

    def get_association_values(self, layers, base_layer, meth, output_filename=None, outFormat='pair', add_to_object= False, **options): #TODO: Talk with PSZ about **optinos or options= {}

        default_options = {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2, "metric": 'euclidean', 
        "corr_type": "pearson", "pvalue": 0.05, "pvalue_adj_method": None, "alternative": 'greater', "coords2sim_type":  'dotProduct' }
        default_options.update(options)

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
        elif meth == "correlation":
            relations = self.get_corr_associations(layers, base_layer, corr_type = default_options["corr_type"], pvalue = default_options["pvalue"], pvalue_adj_method = default_options["pvalue_adj_method"], alternative = default_options["alternative"])
        elif meth == "umap":
            relations = self.get_umap_associations(layers, base_layer, n_neighbors = default_options["n_neighbors"], min_dist = default_options["min_dist"], n_components = default_options["n_components"], metric = default_options["metric"])
        elif meth == "pca":
            relations = self.get_pca_associations(layers, base_layer,  n_components = default_options["n_components"], coords2sim_type = default_options["coords2sim_type"])
        elif meth == "bicm":
            relations = self.get_bicm_associations(layers, base_layer, pvalue = default_options["pvalue"])

        if len(layers) == 1: layers = (layers[0], layers[0])
        self.control_output(values = relations, output_filename=output_filename, inFormat="pair",
         outFormat=outFormat, add_to_object=add_to_object, matrix_keys= ("associations", layers, meth))

        return relations

    def get_matrix_from_keys(self, matrix_keys, symm = True): # TODO: Implement this option in the code.
        layers = matrix_keys[1]
        matrix = self.matrices.dig(*matrix_keys)
        if matrix is None and symm: 
            matrix_keys = list(matrix_keys)
            matrix_keys[1] = (layers[1], layers[0])
            matrix_keys = tuple(matrix_keys)
            matrix = self.matrices.dig(*matrix_keys)
        return matrix

    def get_bicm_associations(self, layers, base_layer, pvalue = 0.05, pvalue_adj_method = 'fdr'):
        # TODO: Need test.
        biadj_matrix = self.matrices.dig(("adjacency_matrices",(*layers,base_layer)))
        if biadj_matrix is None:
            biadj_matrix = self.generate_adjacency_matrix(*layers, base_layer)

        matrix, rowIds, _ = biadj_matrix

        from bicm.graph_classes import BipartiteGraph
        miGraph = BipartiteGraph(biadjacency=matrix)
        relations = miGraph.get_rows_projection(
                            alpha=pvalue,
                            method='poisson',
                            progress_bar=False,
                            fmt='edgelist',
                            validation_method=pvalue_adj_method)
        relations = [[rowIds[relation[0]], rowIds[relation[1]], 1] for relation in relations] # TODO: Check 0-based numeration.
        return relations


    def get_corr_associations(self, layers, base_layer, corr_type = "pearson", pvalue = 0.05, pvalue_adj_method = None, alternative = 'greater'): 
        biadj_matrix = self.matrices.dig(("adjacency_matrices",(*layers,base_layer)))
        if biadj_matrix is None:
            biadj_matrix = self.generate_adjacency_matrix(*layers, base_layer)

        matrix, rowIds, _ = biadj_matrix
        
        if corr_type == "pearson": 
            corr_mat, corr_pvalue = Adv_mat_calc.get_corr(x=matrix.T, alternative=alternative, corr_type = "pearson")
        elif corr_type == "spearman":
            corr_mat, corr_pvalue = Adv_mat_calc.get_corr(x=matrix.T, alternative= alternative, corr_type = "spearman")

        relations = self.matrixes2pairs([corr_pvalue, corr_mat], rowIds, rowIds, symm = True)
        relations = [ relation for relation in relations if relation[0] != relation[1] and not np.isnan(relation[2]) and relation[3] != 0 ]
        if pvalue_adj_method is not None: self.adjust_pval_association(relations, pvalue_adj_method)
        relations = [[relation[0], relation[1], relation[3]] for relation in relations if relation[2] < pvalue]
        return relations

    def get_umap_associations(self, layers, base_layer, n_neighbors = 15, min_dist = 0.1, n_components = 2, metric = 'euclidean', random_seed = None):
        biadj_matrix = self.matrices.dig(("adjacency_matrices",(*layers,base_layer)))
        if biadj_matrix is None:
            biadj_matrix = self.generate_adjacency_matrix(*layers, base_layer)
        data, rowIds, _ = biadj_matrix
        umap_coords = Adv_mat_calc.data2umap(data, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_seed = random_seed)
        umap_sims = np.triu(Adv_mat_calc.coords2sim(umap_coords, sim="euclidean"), k = 1)
        relations = self.matrix2relations(umap_sims, rowIds, rowIds)
        relations = [relation for relation in relations if relation[2] != 0]
        return relations

    def get_pca_associations(self, layers, base_layer, n_components = 2, coords2sim_type = "dotProduct"):
        # TODO: Try to select the correct number of n_componentes (automatically)
        biadj_matrix = self.matrices.dig(("adjacency_matrices",(*layers,base_layer)))
        if biadj_matrix is None:
            biadj_matrix = self.generate_adjacency_matrix(*layers, base_layer)

        matrix, rowIds, _ = biadj_matrix

        x = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(x)
        pca_matrix_sim = Adv_mat_calc.coords2sim(pca_coords, sim = coords2sim_type)
        relations = self.matrix2relations(pca_matrix_sim , rowIds, rowIds)
        return relations


    def get_association_by_transference_resources(self, firstPairLayers, secondPairLayers, lambda_value1 = 0.5, lambda_value2 = 0.5):
        relations = []
        matrix1 = self.matrices["adjacency_matrices"][firstPairLayers][0]
        matrix2 = self.matrices["adjacency_matrices"][secondPairLayers][0]
        finalMatrix = Adv_mat_calc.tranference_resources(matrix1, matrix2, lambda_value1 = lambda_value1, lambda_value2 = lambda_value2)
        rowIds = self.matrices["adjacency_matrices"][firstPairLayers][1]
        colIds =  self.matrices["adjacency_matrices"][secondPairLayers][2]
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

    def get_pcc_associations(self, layers, base_layer, weighted = False ):
        #for Ny calcule use get_nodes_layer
        base_layer_nodes = self.get_nodes_layer([base_layer])
        ny = len(base_layer_nodes)
        def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
            intersProd = len(intersectedIDs) * ny
            nodesProd = len(associatedIDs_node1) * len(associatedIDs_node2)
            nodesSubs = intersProd - nodesProd
            nodesAInNetwork = ny - len(associatedIDs_node1)
            nodesBInNetwork = ny - len(associatedIDs_node2)
            return np.float64(nodesSubs) / math.sqrt(nodesProd * nodesAInNetwork * nodesBInNetwork) # TODO: np.float64 is used to handle division by 0. Fix the implementation/test to avoid this case
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
            self.add_nested_record(pcc_vals, node1, node2, np.abs(assoc_index))
            self.add_nested_record(pcc_vals, node2, node1, np.abs(assoc_index))
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
        relations = [[assoc[0], assoc[1], -np.log10(assoc[2])] for assoc in relations if assoc[2] > 0]
        self.association_values[meth] = relations
        return relations

    def adjust_pval_association(self, associations, method): # TODO TEST
        pvals = np.array([val[2] for val in associations])
        adj_pvals = statsmodels.stats.multitest.multipletests(pvals, method=method, is_sorted=False, returnsorted=False)[1]
        for idx, adj_pval in enumerate(adj_pvals):
            associations[idx][2] = adj_pval

    ## filter methods
    #----------------

    def get_filter(self, layers, method="cutoff", options={}):
        default_options = {"cutoff": None, "compute_autorelations": False, "binarize": False}
        default_options.update(options)

        if method == "cutoff":
            filtered_function = self.filter_cutoff
        else:
            raise Exception('Not defined method')

        edges_with_filtered_values = []
        for layers_pairs in itertools.pairwise(layers):
            edges_with_filtered_values += filtered_function(layers_pairs, cutoff= default_options["cutoff"], compute_autorelations = default_options["compute_autorelations"])

        if default_options["binarize"] == True:
            edges_with_filtered_values = [[edge[0], edge[1], float(edge[2]>0)] for edge in edges_with_filtered_values]

        self.update_edges_with(edges_with_filtered_values) 


    def filter_cutoff(self, layers, cutoff=0.5, compute_autorelations= False):  
        edges_with_filtered_values = []

        def _(node1, node2):
            edges_attr = self.graph.edges()[node1,node2]
            weight = edges_attr["weight"] if edges_attr.get("weight") else 1
            if weight < cutoff:
                weight = 0
            return [node1, node2, weight]

        edges_with_filtered_values = self.get_direct_conns(_, layers = layers, compute_autorelations= compute_autorelations)

        return edges_with_filtered_values

    def write_subgraph(self, layers, output_filename, outFormat='pair'): 
        if outFormat == "pair":
            nodes = [node for node, data in self.graph.nodes(data=True) if data.get("layer") in layers]
            subgraph = self.graph.subgraph(nodes)
            with open(output_filename, "w") as f:
                for nodeA, nodeB, data in subgraph.edges(data=True):
                    weight = data.get("weight")
                    if weight is not None:
                        f.write(f"{nodeA}\t{nodeB}\t{str(weight)}" + "\n")
                    else:
                        f.write(f"{nodeA}\t{nodeB}" + "\n")
        elif outFormat == "matrix":
            if self.matrices["adjacency_matrices"].get(layers) is None:
                self.generate_adjacency_matrix(layers[0], layers[1])
            matrix, rowIds, colIds = self.matrices.dig("adjacency_matrices", layers)
            self.write_obj(matrix, output_filename=output_filename, Format=outFormat, rowIds=rowIds, colIds=colIds)


    def get_direct_conns(self, pair_operation = None, layers = None, compute_autorelations = False): 
        direct_edges = []
        nodeIDsA = self.get_nodes_layer([layers[0]])
        if layers[0] == layers[1]:
            nodeIDsB = nodeIDsA
        else:
            nodeIDsB = self.get_nodes_layer([layers[1]])


        if compute_autorelations:
            all_nodes = list(set(nodeIDsA).union(set(nodeIDsB)))
            for nodeA, nodeB in itertools.combinations(all_nodes,2): # Watchout! Just valids for non directed graphs
                if self.graph.has_edge(nodeA, nodeB):
                    res = pair_operation(nodeA, nodeB)
                    direct_edges.append(res)
        else:
            for nodeA in nodeIDsA:
                for nodeB in nodeIDsB:
                    if self.graph.has_edge(nodeA, nodeB):
                        res = pair_operation(nodeA, nodeB)
                        direct_edges.append(res)

        return direct_edges


    def update_edges_with(self, relations):
        for nodeA, nodeB, weight in relations:
            if weight > 0:
                self.graph.add_edge(nodeA, nodeB, weight=weight)
            elif self.graph.has_edge(nodeA, nodeB):
                self.graph.remove_edge(nodeA, nodeB)


    ## Kernel and similarity methods
    #------------------------------------

    def get_kernel(self, layers, method, normalization=False, sim_type= "dotProduct", embedding_kwargs={}, output_filename=None, outFormat='matrix', add_to_object= False):
        #embedding_kwargs accept: dimensions, walk_length, num_walks, p, q, workers, window, min_count, seed, quiet, batch_words

        if method in Graph2sim.allowed_embeddings:
            embedding_nodes = [node for node, layer in self.graph.nodes('layer') if layer in list(layers)] 
            subgraph2embed = self.graph.subgraph(embedding_nodes)
            emb_coords = Graph2sim.get_embedding(subgraph2embed, embedding = method, **embedding_kwargs)
            
            kernel = Graph2sim.emb_coords2kernel(emb_coords, normalization, sim_type= sim_type)
            rowIds = list(subgraph2embed.nodes())
            colIds = rowIds
        elif method[0:2] in Graph2sim.allowed_kernels:
            adj_mat, rowIds, colIds = self.matrices["adjacency_matrices"][(layers[0],layers[0])]
            kernel = Graph2sim.get_kernel(adj_mat, method, normalization=normalization)

        self.control_output(values = kernel, rowIds = rowIds, colIds = colIds, output_filename=output_filename, inFormat="matrix",
         outFormat=outFormat, add_to_object=add_to_object, matrix_keys= ("kernels", layers, method))
        return kernel, rowIds, colIds

    def write_kernel(self, layers, kernel_type, output_file):
        kernel, rowIds, colIds = self.matrices["kernels"][layers][kernel_type]
        np.save(output_file, kernel)
        self.write_nodelist(rowIds, output_file + "_rowIds")
        self.write_nodelist(rowIds, output_file + "_colIds")

    def get_similarity(self, layers, base_layer, sim_type='lin', options={}, output_filename=None, outFormat='pair', add_to_object= False):
        # options--> options['term_filter'] = GO:00001
        ontology = self.layer_ontologies[base_layer]
        relations = self.get_layers_as_dict(layers, base_layer)
        ontology.load_profiles(relations)
        ontology.clean_profiles(store = True,options=options)
        similarity_pairs = ontology.compare_profiles(sim_type = sim_type)

        if len(layers) == 1: layers = (layers[0],layers[0])
        self.control_output(values = similarity_pairs, output_filename=output_filename, inFormat="nested_pairs",
         outFormat=outFormat, add_to_object=add_to_object, matrix_keys= ("semantic_sims", layers, sim_type))
        return similarity_pairs



    def shortest_path(self, source, target):
        return nx.shortest_path(self.graph, source, target)

    def average_shortest_path_length(self, community):
        weight_attr_name = "weight" if nx.get_edge_attributes(self.graph, 'weight') else None
        try:
            com = community.copy()
            path_lens = []
            while len(com) > 1:
                source = com.pop()
                for target in com:
                    path_lens.append(nx.shortest_path_length(self.graph, source, target, weight= weight_attr_name))
            asp_com = np.mean(path_lens)
        except nx.exception.NetworkXNoPath:
            asp_com = None
        return asp_com 

    def shortest_paths(self, community):
        return nx.all_pairs_shortest_path(community)


    def get_node_attributes(self, attr_names, layers = "all", summary = False, output_filename = None):
        if layers == "all": layers = self.layers
        node_universe = self.get_nodes_layer(layers)

        attrs = {}
        for attr_name in attr_names:
            if attr_name == 'get_degree':
                attrs["get_degree"] = self.get_degree(zscore=False)
            elif attr_name == 'get_degreeZ':
                attrs["get_degreeZ"] = self.get_degree()
            elif attr_name == "betweenness_centrality":
                attrs["betweenness_centrality"] = self.get_betweenness_centrality(zscore=False)
            elif attr_name == "betweenness_centralityZ":
                attrs["betweenness_centralityZ"] = self.get_betweenness_centrality()

        node_ids = attrs[attr_names[0]].keys() # TODO: This line of code should be replaced for an option to select node for each attr.
        node_ids = [node_id for node_id in node_ids if node_id in node_universe]
        
        node_attrs = []
        if summary: 
            for at in attr_names:
                stats = Adv_mat_calc.get_stats_from_list(list(attrs[at].values()))
                node_attrs += [[at] + stat for stat in stats]
        else:
            for n in node_ids:
                n_attrs = [ attrs[at][n] for at in attr_names ]
                node_attrs.append([n] + n_attrs)

        self.control_output(values = node_attrs, output_filename=output_filename, inFormat="pair", outFormat="pair", add_to_object=False)

        return node_attrs

    ## Ploting method 
    #----------------

    def plot_network(self, options = {}):
        net_data = {
            'group_nodes': self.group_nodes,
            'reference_nodes': self.reference_nodes,
            'graph': self.graph,
            'layers': self.layers
        }
        Net_plotter(net_data, options)

    ## Matrix information and manipulation
    #-------------------------------------

    def write_matrix(self, mat_keys, output_filename):
        matrix_row_col = self.matrices.dig(*mat_keys)
        if matrix_row_col is not None:
            mat, rowIds, colIds = matrix_row_col
            self.write_obj(mat, output_filename, Format= "matrix", rowIds=rowIds, colIds=colIds)
        else:              
            raise Exception("keys for matrices which dont exist yet")

    def write_stats_from_matrix(self, mat_keys, output_filename="stats_from_matrix"): 
        matrix_data = self.matrices.dig(*mat_keys)
        if matrix_data == None: raise Exception("keys for matrices which dont exist yet")
        matrix, _, _ = matrix_data

        stats = Adv_mat_calc.get_stats_from_matrix(matrix)
        self.write_obj(stats, output_filename, Format="pair")

    def normalize_matrix(self, mat_keys, by= "rows_cols"):
        matrix_data = self.matrices.dig(*mat_keys)
        if matrix_data == None: raise Exception("keys for matrices which dont exist yet")
        matrix, rowIds, colIds = matrix_data

        if by == "rows_cols":
            matrix = Adv_mat_calc.row_col_normalization(matrix)
        elif by == "cosine":
            matrix = Adv_mat_calc.cosine_normalization(matrix)

        self.control_output(values = matrix, rowIds = rowIds, colIds = colIds, output_filename = None, outFormat = "matrix",
            inFormat = "matrix", add_to_object = True, matrix_keys = mat_keys)


    def mat_vs_mat(self, mat1_rowcol, mat2_rowcol, operation="cutoff", options={"cutoff": 0, "cutoff_type": "greater"}):
        default_options={"cutoff": 0, "cutoff_type": "greater"}
        default_options.update(options)
        mat1, rows1, cols1 = mat1_rowcol
        mat2, rows2, cols2 = mat2_rowcol

        if operation == "filter":
            if default_options["cutoff_type"] == "greater":
                mat2 = mat2 >= default_options["cutoff"]
            elif default_options["cutoff_type"] == "less":
                mat2 = mat2 <= default_options["cutoff"]
            mat_result = mat1 * mat2
            rows_result, cols_result = rows1, cols1
            mat_result, rows_result, cols_result = Adv_mat_calc.remove_zero_lines(mat_result, rows_result, cols_result)

        return mat_result, rows_result, cols_result


    def mat_vs_mat_operation(self, mat1_keys, mat2_keys, operation, options, output_filename=None, outFormat='matrix', add_to_object= False):
        result = (None, None, None)

        mat1 = self.matrices.dig(*mat1_keys)
        mat2 = self.matrices.dig(*mat2_keys)

        if mat1 is None or mat2 is None:
            raise Exception("keys for matrices which dont exist yet")

        mat_result, rows_result, cols_result = self.mat_vs_mat(mat1, mat2, operation, options)


        self.control_output(values = mat_result, rowIds = rows_result, colIds = cols_result, output_filename = output_filename, 
            inFormat = "matrix", outFormat = outFormat, add_to_object = add_to_object, matrix_keys = mat1_keys)
        return mat_result, rows_result, cols_result


    def filter_matrix(self, mat_keys, operation, options, output_filename=None, outFormat='matrix', add_to_object= False):
        result = (None, None, None)

        mat1 = self.matrices.dig(*mat_keys)

        if mat1 is None:      
            raise Exception("keys for matrices which dont exist yet")
        else:
            mat1, rows1, cols1 = mat1
            layers = mat_keys[1]

        if operation == "filter_cutoff":
            filtered_mat = mat1 >= options["cutoff"]
            mat_result = mat1 * filtered_mat
            Adv_mat_calc.filter_cutoff_mat(mat1, cutoff = options["cutoff"])
            rows_result, cols_result = rows1, cols1
        elif operation == "filter_disparity":
            mat_result, rows_result, cols_result = Adv_mat_calc.disparity_filter_mat(mat1, rows1, cols1, pval_threshold = options["pval_threshold"])
        elif operation == "filter_by_percentile":
            mat_result = Adv_mat_calc.percentile_filter(mat1, options["percentile"]) # TODO: Check is this is valid for non-square matrix
            rows_result, cols_result = rows1, cols1

        if options.get("binarize"):
            mat_result = Adv_mat_calc.binarize_mat(mat_result)

        mat_result, rows_result, cols_result = Adv_mat_calc.remove_zero_lines(mat_result, rows_result, cols_result)

        self.control_output(values = mat_result, rowIds = rows_result, colIds = cols_result, output_filename = output_filename, 
            inFormat = "matrix", outFormat = outFormat, add_to_object = add_to_object, matrix_keys = mat_keys)
        return mat_result, rows_result, cols_result


    ## Community Methods 
    #-------------------

    # Cluster (community) dicovery #

    def get_communities_as_cdlibObj(self,communities,overlaping=False): # communites is a hash like group_nodes
        coms = [list(c) for c in communities.values()]
        communities = NodeClustering(coms, self.graph, "external", method_parameters={}, overlap=overlaping)
        return communities

    def discover_clusters(self, cluster_method, clust_kwargs, **user_options):
        if user_options.get("seed") != None: self.set_seed(user_options.get("seed"))
        communities = self.get_clusters_by_algorithm(cluster_method, clust_kwargs)
        if cluster_method in ['hlc', 'hlc_f']: communities = self.link_to_node_communities(communities)
        communities = { str(idx): community for idx, community in enumerate(communities)}
        self.group_nodes.update(communities) # If external coms added, thay will not be removed!

    def link_to_node_communities(self, communities):
        comm_nodes = []
        for com in communities:
            if len(com) > 1 :
                nodes = []
                for e in com:
                    a, b = e
                    if not a in nodes: nodes.append(a)
                    if not b in nodes: nodes.append(b)
                if len(nodes) > 2: comm_nodes.append(nodes)
        return comm_nodes

    def get_clusters_by_algorithm(self, cluster_method, clust_kwargs={}):
        if(cluster_method == 'leiden'):
            communities = algorithms.leiden(self.graph, weights='weight', **clust_kwargs)
        elif(cluster_method == 'louvain'):
            communities = algorithms.louvain(self.graph, weight='weight', **clust_kwargs)
        elif(cluster_method == 'cpm'):
            communities = algorithms.cpm(self.graph, weights='weight', **clust_kwargs)
        elif(cluster_method == 'der'):
            communities = algorithms.der(self.graph, **clust_kwargs)
        elif(cluster_method == 'edmot'):
            communities = algorithms.edmot(self.graph, **clust_kwargs)
        elif(cluster_method == 'eigenvector'):
            communities = algorithms.eigenvector(self.graph, **clust_kwargs)
        elif(cluster_method == 'gdmp2'):
            communities = algorithms.gdmp2(self.graph, **clust_kwargs)
        elif(cluster_method == 'greedy_modularity'):
            communities = algorithms.greedy_modularity(self.graph, weight='weight', **clust_kwargs)
        elif(cluster_method == 'label_propagation'):
            communities = algorithms.label_propagation(self.graph, **clust_kwargs)
        elif(cluster_method == 'markov_clustering'):
            communities = algorithms.markov_clustering(self.graph, **clust_kwargs)
        elif(cluster_method == 'rber_pots'):
            communities = algorithms.rber_pots(self.graph, weights='weight', **clust_kwargs)
        elif(cluster_method == 'rb_pots'):
            communities = algorithms.rb_pots(self.graph, weights='weight', **clust_kwargs)
        elif(cluster_method == 'significance_communities'):
            communities = algorithms.significance_communities(self.graph, **clust_kwargs)
        elif(cluster_method == 'spinglass'):
            communities = algorithms.spinglass(self.graph, **clust_kwargs)
        elif(cluster_method == 'surprise_communities'):
            communities = algorithms.surprise_communities(self.graph, **clust_kwargs)
        elif(cluster_method == 'walktrap'):
            communities = algorithms.walktrap(self.graph, **clust_kwargs)
        elif(cluster_method == 'lais2'):
            communities = algorithms.lais2(self.graph, **clust_kwargs)
        elif(cluster_method == 'big_clam'):
            communities = algorithms.big_clam(self.graph, **clust_kwargs)
        elif(cluster_method == 'danmf'):
            communities = algorithms.danmf(self.graph, **clust_kwargs)
        elif(cluster_method == 'ego_networks'):
            communities = algorithms.ego_networks(self.graph, **clust_kwargs)
        elif(cluster_method == 'egonet_splitter'):
            communities = algorithms.egonet_splitter(self.graph, **clust_kwargs)
        elif(cluster_method == 'mnmf'):
            communities = algorithms.mnmf(self.graph, **clust_kwargs)
        elif(cluster_method == 'nnsed'):
            communities = algorithms.nnsed(self.graph, **clust_kwargs)
        elif(cluster_method == 'slpa'):
            communities = algorithms.slpa(self.graph, **clust_kwargs)
        elif(cluster_method == 'bimlpa'):
            communities = algorithms.bimlpa(self.graph, **clust_kwargs)
        elif(cluster_method == 'wcommunity'):
            communities = algorithms.wCommunity(self.graph, **clust_kwargs)
        elif(cluster_method == 'kclique'):
            communities = algorithms.kclique(self.graph, **clust_kwargs)
        elif(cluster_method == 'hlc'):
            communities = algorithms.hierarchical_link_community(self.graph, **clust_kwargs)
        elif(cluster_method == 'hlc_f'):
            communities = algorithms.hierarchical_link_community_full(self.graph, **clust_kwargs)
        elif(cluster_method == 'aslpaw'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                communities = algorithms.aslpaw(self.graph)
        else:
            raise Exception('Not defined method')
        print(communities.method_parameters, file=sys.stderr)
        print(communities.overlap, file=sys.stderr)
        print(communities.node_coverage, file=sys.stderr)

        return communities.communities # To return a list of list with each of the nodes names for each communities.
    
    # Metrics

    # Evaluating one community

    def compute_comparative_degree(self, com): # see Girvan-Newman Benchmark control parameter in http://networksciencebook.com/chapter/9#testing (communities chapter)
        internal_degree = 0
        external_degree = 0
        com_nodes = set(com)
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
        for nodeID in com: # Change this to put as a list of nodes
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

    # Evaluating all communities

    def communities_avg_sht_path(self, coms):
        asp_coms = []
        for com_id, com in coms.items():
            asp_com = self.average_shortest_path_length(com)
            asp_coms.append(asp_com)
        return asp_coms

    def communities_comparative_degree(self, coms):
        return [ self.compute_comparative_degree(com) for com_id, com in coms.items()]

    def communities_node_com_assoc(self, coms, ref_node):
        return [ self.compute_node_com_assoc(com, ref_node) for com_id, com in coms.items()]

    def compute_summarized_group_metrics(self, output_filename, metrics = ['size', 'avg_transitivity', 'internal_edge_density',
     'conductance', 'triangle_participation_ratio', 'max_odf', 'avg_odf', 'avg_embeddedness', 'average_internal_degree','cut_ratio',
     'fraction_over_median_degree', 'scaled_density']):
            # HAS NOT SUMMARY: 'surprise', 'significance', 'comparative_degree', 'avg_sht_path', 'node_com_assoc'
        communities = NodeClustering(list(self.group_nodes.values()), self.graph, "external", overlap=True)
        results = []
        for metric in metrics:
            # https://www.kite.com/python/answers/how-to-call-a-function-by-its-name-as-a-string-in-python
            class_method = getattr(evaluation, metric)
            res = class_method(self.graph, communities)
            results.append(res)

        with open(output_filename, 'w') as out_file:
            out_file.write("\t".join(["Metric", "Mean", "Max", "Min", "Std"]) + "\n")
            count = 0
            for res in results:
                metric_name = metrics[count]
                out_file.write("\t".join([metric_name, str(res.score), str(res.max), str(res.min), str(res.std)]) + "\n")
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
            communities = NodeClustering(list(self.group_nodes.values()), self.graph, "external", overlap=True) # TODO Maybe this is not the most efficient way (?)
            class_method = getattr(evaluation, metric)
            res = class_method(self.graph, communities, summary=False)
            for i, val in enumerate(res): output_metrics[i].append(self.replace_none_vals(val))
            header.append(metric)

    # Evaluating comparison between partitions (EXTERNAL EVALUATION IN CDLIB)
    # Note: Partitions are non overlapped communities (Must be).

    def compare_partitions(self, communities_ref):
        communities = self.get_communities_as_cdlibObj(self.group_nodes)
        ref_communities = self.get_communities_as_cdlibObj(communities_ref)
        res = evaluation.adjusted_mutual_information(ref_communities,communities) # This could be easily extended
        return(res)

    # TODO: Add ranker evalutation for set of clusterings (This is told to be added in a posterior expansion phase of lib)

    # Cluster Expansions #
    def expand_clusters(self, expand_method, one_sht_paths = False):
        clusters = {}

        if one_sht_paths:
            get_sht_path = lambda G, nodeA, nodeB: [nx.shortest_path(G, NodeA, NodeB)]
        else:
            get_sht_path = lambda G, nodeA, nodeB: nx.all_shortest_paths(G, NodeA, NodeB)

        for id, com in self.group_nodes.items():
            if expand_method == 'sht_path':
                new_nodes = set(com) 
                #Community nodes are included in the set above and then this set is expanded with shortest path nodes
                # between community nodes and assigned as the new cluster nodes list, otherwise updating the current list 
                # could potentially add original community nodes again if they are found in the shortest path between other community nodes. 
                sht_paths = []

                for NodeA, NodeB in itertools.combinations(com, 2):
                    if NodeA in self.graph.nodes and NodeB in self.graph.nodes:
                        sht_path = None
                        try:
                            sht_path = get_sht_path(self.graph, NodeA, NodeB)
                        except nx.exception.NetworkXNoPath:
                            continue 
                        sht_paths.append(sht_path)

                for node_pair_sht_paths in sht_paths:
                    for path in node_pair_sht_paths:
                        new_nodes = new_nodes.union(set(path))
                self.group_nodes[id] = new_nodes #Originally it was modified inplace with "com.add_nodes_from(list(new_nodes))" because "com" were networkx objects
                clusters[id] = new_nodes
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


    def randomize_network(self, random_type, **user_options):
        if user_options.get("seed") != None: self.set_seed(user_options.get("seed"))

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

    def add_nested_record(self, hash, node1, node2, val):
        query_node1 = hash.get(node1)
        if query_node1 is None:
            hash[node1] = {node2: val}
        else:
            query_node1[node2] = val

    
    # TODO: Talk with PSZ: The next three methods are redundant, but i dont see an efficiente manner of making one generic function (Fred)
    def matrix2relations(self, finalMatrix, rowIds, colIds, symm = False):
        relations = []
        if symm:
            for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds[rowPos:]):
                    colPos += rowPos
                    associationValue = finalMatrix[rowPos, colPos]
                    if associationValue >= 0: relations.append([rowId, colId, associationValue])
        else:
            for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds):
                    associationValue = finalMatrix[rowPos, colPos]
                    if associationValue >= 0: relations.append([rowId, colId, associationValue])
        return relations

    def matrix2pairs(self, matrix, rowIds, colIds, symm = False):
        relations = []
        if symm:
            for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds[rowPos:]):
                    colPos += rowPos
                    relations.append([rowId, colId, matrix[rowPos, colPos]])
        else:
            for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds):
                    relations.append([rowId, colId, matrix[rowPos, colPos]])
        return relations

    def matrixes2pairs(self, matrixes, rowIds, colIds, symm = False):
        # When there are multiple matrix with the same rows and cols.
        relations = []
        if symm:
            for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds[rowPos:]):
                    colPos += rowPos
                    associationValues = [matrix[rowPos, colPos] for matrix in matrixes]
                    relations.append([rowId, colId, *associationValues])
        else:
            for rowPos, rowId in enumerate(rowIds):
                for colPos, colId in enumerate(colIds):
                    associationValues = [matrix[rowPos, colPos] for matrix in matrixes]
                    relations.append([rowId, colId, associationValue])
        return relations


    def update_index(self, index, element, count):
        if index.get(element) is None:
            index[element] = count
            count += 1
        return count

    def pairs2matrix(self, pairs, symm= True):
        count_A = 0
        index_A = {}
        count_B = 0
        index_B = {}
        if symm:
            for pair in pairs:
                elementA, elementB, val = pair
                count_A = self.update_index(index_A, elementA, count_A)
                count_A = self.update_index(index_A, elementB, count_A)
            index_B = index_A
        else:
            for pair in pairs:
                elementA, elementB, val = pair
                count_A = self.update_index(index_A, elementA, count_A)
                count_B = self.update_index(index_B, elementB, count_B)

        elementA_names = list(index_A.keys())
        elementB_names = list(index_B.keys())

        matrix = np.zeros((len(elementA_names), len(elementB_names)))
        for pair in pairs:
            elementA, elementB, val = pair
            i = index_A[pair[0]] 
            j = index_B[pair[1]] 
            matrix[i, j] = val
            if symm: matrix[j, i] = val

        return matrix, elementA_names, elementB_names

    def transform2obj(self, obj, inFormat=None, outFormat=None, rowIds= None, colIds= None):
        if outFormat == 'pair':
            if inFormat == 'matrix': 
                obj = self.matrix2pairs(obj, rowIds=rowIds, colIds=colIds)
            if inFormat == 'nested_pairs':
                obj = self.nested_pairs2pairs(obj)
        elif outFormat == 'matrix':
            if inFormat == 'pair': 
                obj, rowIds, colIds = self.pairs2matrix(obj)
            if inFormat == 'nested_pairs':
                obj, rowIds, colIds = self.pairs2matrix(self.nested_pairs2pairs(obj)) # Talk with PSZ about the trade-off combinatorial vs optimization
        return obj, rowIds, colIds

    def control_output(self, values, output_filename = None, inFormat = "pair", outFormat = "matrix" , add_to_object = False, matrix_keys = None, rowIds = None, colIds = None):
        if add_to_object: 
            matrix, rowIds, colIds = self.transform2obj(values, inFormat= inFormat, outFormat= "matrix", rowIds = rowIds, colIds = colIds) 
            path_keys = matrix_keys[:-1]
            final_key = matrix_keys[-1]
            result_dic = self.matrices.dig(*path_keys)
            if result_dic is not None:
                result_dic[final_key] = [matrix, rowIds, colIds]
            else:
                # This is posible just when len(matrix_keys) == 3
                self.matrices[path_keys[0]][path_keys[1]] = {final_key: [matrix, rowIds, colIds]}
        elif output_filename != None: 
            obj, rowIds, colIds = self.transform2obj(values, inFormat= inFormat, outFormat= outFormat, rowIds = rowIds, colIds = colIds)
            self.write_obj(obj, output_filename, Format=outFormat, rowIds=rowIds, colIds=colIds)

    def nested_pairs2pairs(self, nested_dic_pairs):
        pairs = []
        for item_a, dat in nested_dic_pairs.items(): 
            for item_b, val in dat.items(): pairs.append([item_a, item_b, val])
        return pairs


    def write_obj(self, obj, output_filename, Format=None, rowIds=None, colIds=None):
        if Format == 'pair':
            with open(output_filename, 'w') as f:
                for pair in obj: f.write("\t".join([str(item) for item in pair]) + "\n")
        elif Format == 'matrix':
            np.save(output_filename, obj)
            if rowIds != None:
                with open(output_filename + '_rowIds', 'w') as f:
                    for item in rowIds: f.write(item + "\n")
            if colIds != None:
                with open(output_filename + '_colIds', 'w') as f:
                    for item in colIds: f.write(item + "\n")  
        

    def write_nodelist(self, nodes, file_name):
        with open(file_name, "w") as f:
            for node in nodes:
                f.write(node + "\n")

    def replace_none_vals(self, val):
        return 'NULL' if val == None else val

    def set_seed(self, seed):
        try: 
            random.seed(int(seed))
            np.random.seed(int(seed))
        except ValueError:
            #np seed cannot used something else but integers, and although random allows it, 200, 200.0 and "200" gives different results, so in order to avoid weird results, we force the seed to be an integer
            raise(f"ERROR: The seed must be a valid number")