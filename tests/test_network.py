import sys 
import unittest
import os
import math
import numpy as np
import networkx as nx
import random
import py_exp_calc.exp_calc as pxc
from NetAnalyzer import NetAnalyzer
from NetAnalyzer import Net_parser
from statsmodels.stats.multitest import multipletests
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

class BaseNetTestCase(unittest.TestCase):
	def setUp(self):
		self.tripartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+'], ['salient', 'S[0-9]+']]
		self.tripartite_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'tripartite_network_for_validating.txt'), self.tripartite_layers)
		self.tripartite_network.generate_adjacency_matrix(self.tripartite_layers[0][0], self.tripartite_layers[1][0])
		self.tripartite_network.generate_adjacency_matrix(self.tripartite_layers[1][0], self.tripartite_layers[2][0])

		self.tripartite_network_weights = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'tripartite_network_weighted_for_validating.txt'), self.tripartite_layers)
		self.tripartite_network_weights.generate_adjacency_matrix(self.tripartite_layers[0][0], self.tripartite_layers[1][0])
		self.tripartite_network_weights.generate_adjacency_matrix(self.tripartite_layers[1][0], self.tripartite_layers[2][0])
		
		self.bipartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]
		self.network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), self.bipartite_layers)
		self.network_obj.generate_adjacency_matrix(self.bipartite_layers[0][0], self.bipartite_layers[1][0])
		
		self.monopartite_layers = [['main', '\w'], ['main', '\w']]
		self.monopartite_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'monopartite_network_for_validating.txt'), self.monopartite_layers)
		self.monopartite_network.generate_adjacency_matrix(self.monopartite_layers[0][0], self.monopartite_layers[0][0])

		self.monopartite_network_weights = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'monopartite_network_weights_for_validating.txt'), self.monopartite_layers)
		self.monopartite_network_weights.generate_adjacency_matrix(self.monopartite_layers[0][0], self.monopartite_layers[0][0])

		self.comunities_network_layers = [['main', '\w'], ['main', '\w']]
		self.comunities_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'comunities_network_for_validating.txt'), self.comunities_network_layers)
		self.comunities_network_no_coms = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'comunities_network_for_validating.txt'), self.comunities_network_layers)

		self.comunities_network.generate_adjacency_matrix(self.comunities_network_layers[0][0], self.comunities_network_layers[0][0])
		self.comunities_network.reference_nodes = ["X"]
		com1 = [node for node in self.comunities_network.graph.nodes if node in "ABCDEF"]		
		com2 = [node for node in self.comunities_network.graph.nodes if node in "ZLNM"]
		self.comunities_network.group_nodes = {'com1': com1, 'com2': com2}

		self.clusters_network_layers = [['main', '\w'], ['main', '\w']]
		self.clusters_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'clusters_network_for_validating.txt'), self.clusters_network_layers)
		self.clusters_network.generate_adjacency_matrix(self.clusters_network_layers[0][0], self.clusters_network_layers[0][0])
		clust1 = [node for node in self.clusters_network.graph.nodes if node in "ABCDEFGHIJ"]
		clust2 = [node for node in self.clusters_network.graph.nodes if node in "MNOPQR"]
		self.clusters_network.group_nodes = {'clust1': clust1, 'clust2': clust2}	

	def test_clone(self):
		network_clone = self.network_obj.clone()
		self.assertEqual(self.network_obj, network_clone)

	def test_clone_change(self):
		network_clone = self.network_obj.clone()
		network_clone.add_node('M8', network_clone.set_layer(self.bipartite_layers, 'M8'))
		self.assertNotEqual(self.network_obj, network_clone	)

	def test_rows_cols_normalization(self):
		self.monopartite_network.normalize_matrix(("adjacency_matrices",("main","main")), by="rows_cols")
		test_result = self.monopartite_network.matrices["adjacency_matrices"][("main","main")][0]
		expected_result = np.array([[0.        , 0.70710678, 0.5       , 0.        , 0.        ],
       								[0.70710678, 0.        , 0.        , 0.        , 0.        ],
       								[0.5       , 0.        , 0.        , 0.        , 0.5       ],
       								[0.        , 0.        , 0.        , 0.        , 0.70710678],
       								[0.        , 0.        , 0.5       , 0.70710678, 0.        ]])
		self.assertTrue((expected_result.round(decimals=5) == test_result.round(decimals=5)).all())
	
	def test_generate_adjacency_matrix_monopartite(self):
		test_values = self.monopartite_network.matrices["adjacency_matrices"]
		matrix_values = np.array([[0, 1, 1, 0, 0],[1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 0]],dtype='float')
		expected_values = { ('main', 'main') : [matrix_values, ['A', 'C', 'E', 'B', 'D'], ['A', 'C', 'E', 'B', 'D']]} 
		self.assertEqual(expected_values[('main', 'main')][0].tolist(), test_values[('main', 'main')][0].tolist())
		self.assertEqual(expected_values[('main', 'main')][1], test_values[('main', 'main')][1])

	def test_generate_adjacency_matrix_bipartite(self):
		test_values = self.network_obj.matrices["adjacency_matrices"]
		matrix_values = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
			[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
			[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
			[1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
			[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
		expected_values = {('main', 'projection') : [matrix_values, ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'], ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']]} 
		self.assertEqual(expected_values[('main', 'projection')][0].tolist(), test_values['main', 'projection'][0].tolist())
		self.assertEqual(expected_values[('main', 'projection')][1], test_values[('main', 'projection')][1])	
	
	def test_delete_nodes_d_mono(self):
		network_clone = self.monopartite_network.clone()
		network_clone.delete_nodes(['E'])
		self.assertEqual(4, len(network_clone.get_nodes_layer(['main'])))
		self.assertEqual(2, network_clone.get_edge_number())

	def test_delete_nodes_d_bi(self):
		network_clone = self.network_obj.clone()
		network_clone.delete_nodes(['M1', 'M2'])
		self.assertEqual(4, len(network_clone.get_nodes_layer(['main'])))
		self.assertEqual(20, network_clone.get_edge_number())

	def test_delete_nodes_r_mono(self):
		network_clone = self.monopartite_network.clone()
		network_clone.delete_nodes(['E', 'D'], mode = 'r')
		self.assertEqual(2, len(network_clone.get_nodes_layer(['main'])))
		self.assertEqual(1, network_clone.get_edge_number())

	def test_delete_nodes_r_bi(self):
		network_clone = self.network_obj.clone()
		network_clone.delete_nodes(['M1', 'P1'], mode = 'r')
		self.assertEqual(2, len(network_clone.get_nodes_layer(['main', 'projection'])))
		self.assertEqual(1, network_clone.get_edge_number())

	def test_get_connected_nodes(self):
		test_result = self.monopartite_network.get_connected_nodes('A', 'main')
		self.assertEqual(['C', 'E'], test_result)

	def test_get_edge_number(self): 
		edge_number_test = self.monopartite_network.get_edge_number()
		self.assertEqual(4, edge_number_test)

	def test_get_degree(self):
		degree_test = self.monopartite_network.get_degree(zscore = False)
		expected_result = {'A' : 2, 'C' : 1, 'E' : 2, 'B' : 1, 'D' : 2}
		self.assertEqual(expected_result, degree_test)

	def test_get_connected_nodes(self):
		test_result = self.monopartite_network.get_connected_nodes('A', 'main')
		self.assertEqual(['C', 'E'], test_result)

	def test_get_nodes_from_layer(self):
		test_result = self.network_obj.get_nodes_layer(['main'])
		expected_result = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
		self.assertEqual(expected_result, test_result)

	def test_get_bipartite_subgraph(self): 
		bipartirte_test = self.tripartite_network.get_bipartite_subgraph(['M1', 'M2', 'M3'], 'salient', 'projection')
		expected_result = {'P1': ['S1'], 'P2': ['S1', 'S2', 'S3']}
		self.assertEqual(expected_result, bipartirte_test)

	def	test_get_all_intersections_autorr_all_layers_conn(self): 
		network_clone = self.tripartite_network.clone()
		test_result = network_clone.get_all_intersections()
		expected_result = [1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1]
		self.assertEqual(expected_result, test_result)

	def test_get_all_intersections_no_autorr_all(self):
		# how to test raising exceptions? https://www.pythonclear.com/unittest/python-unittest-assertraises/ 
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("all", False)
		fun = lambda node1,node2 : [node1, node2]
		with self.assertRaises(NotImplementedError) as e:
			network_clone.get_all_pairs(pair_operation= fun)
		self.assertEqual(str(e.exception),'Not implemented')

	def test_get_all_pairs_autorr_all_layers_conn(self):
		network_clone = self.tripartite_network.clone()
		fun = lambda node1,node2 : [node1, node2]
		test_result = network_clone.get_all_pairs(pair_operation= fun)
		self.assertEqual(16, len(test_result))

	def test_get_all_pairs_autorr_all_layers_all(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("all", True)
		fun = lambda node1,node2 : [node1, node2]
		test_result = network_clone.get_all_pairs(pair_operation= fun)
		self.assertEqual(36, len(test_result))

	def test_get_all_pairs_autorr_some_layers_conn(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("conn", True)
		fun = lambda node1,node2 : [node1, node2]
		test_result = network_clone.get_all_pairs(layers = ["main", "salient"], pair_operation= fun)
		self.assertEqual(13, len(test_result))

	def test_get_all_pairs_autorr_some_layers_all(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("all", True)
		fun = lambda node1,node2 : [node1, node2]
		test_result = network_clone.get_all_pairs(layers = ["main", "salient"], pair_operation= fun)
		self.assertEqual(15, len(test_result))

	def test_get_all_pairs_no_autorr_some_layers_conn(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("conn", False)
		fun = lambda node1,node2 : [node1, node2]
		test_result = network_clone.get_all_pairs(layers = ["main", "salient"], pair_operation= fun)
		self.assertEqual(7, len(test_result))
	
	def test_get_all_pairs_no_autorr_all_layers_conn(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("conn", False)
		test_result = network_clone.get_all_pairs()
		self.assertEqual([], test_result)

	def test_collect_nodes_autorr_some_layers(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("conn", True)
		nodesA_test, nodesB_test = network_clone.collect_nodes(layers = ["main", "salient"])
		expected_result_nodesA = ["M1", "M2", "M3", "S1", "S2", "S3"]
		self.assertEqual(expected_result_nodesA, nodesA_test)
		self.assertEqual([], nodesB_test)
	
	def test_collect_nodes_no_autorr_some_layers(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("all", False)
		nodesA_test, nodesB_test = network_clone.collect_nodes(layers = ["main", "salient"])
		expected_result_nodesA = ["M1", "M2", "M3"]
		expected_result_nodesB = ["S1", "S2", "S3"]
		self.assertEqual(expected_result_nodesA, nodesA_test)
		self.assertEqual(expected_result_nodesB, nodesB_test)

	def test_collect_nodes_no_autorr_all_layers(self):
		network_clone = self.tripartite_network.clone()
		network_clone.set_compute_pairs("conn", False)
		nodesA_test, nodesB_test = network_clone.collect_nodes(layers = "all")
		self.assertEqual([], nodesA_test)
		self.assertEqual([], nodesB_test)

	def test_get_nodes_layer(self):
		nodes_from_layers_test = self.tripartite_network.get_nodes_layer(["main", "salient"])
		expected_result = ["M1", "M2", "M3", "S1", "S2", "S3"]
		self.assertEqual(expected_result, nodes_from_layers_test)

	def test_intersection(self):
		test_result = self.network_obj.intersection('M3', 'M6')
		expected_result = ['P1', 'P2']
		self.assertEqual(sorted(['P1', 'P2']), sorted(list(test_result)))

	def test_get_node_attributes(self):
		node_attribute_test = self.monopartite_network.get_node_attributes(['get_degree'])
		expected_result = [['A', 2], ['C', 1], ['E', 2],['B', 1], ['D', 2]]
		self.assertEqual(expected_result, node_attribute_test)

	def test_get_node_attributes_zscore(self):
		node_attribute_test = self.monopartite_network.get_node_attributes(['get_degreeZ', 'get_degree'])
		expected_result = [['A', 0.8164965809277259, 2], ['C', -1.2247448713915894, 1], ['E', 0.8164965809277259, 2],['B', -1.2247448713915894, 1], ['D', 0.8164965809277259, 2]]
		self.assertEqual(expected_result, node_attribute_test)

	def test_get_graph_attributes(self):
		test = self.monopartite_network.get_graph_attributes(['size','edge_density','global_clustering', 'assorciativity'])
		expected_values = [['size', 5], ['edge_density', 0.4], ['global_clustering', 0], ['assorciativity', -0.3333333333333333]]
		self.assertEqual(expected_values, test)

	def test_get_corr_association(self):
		test_association = self.network_obj.get_corr_associations(['main'], 'projection', corr_type = "pearson", pvalue = 0.5)
		test_association = [ [association[0], association[1], np.round(association[2],4)] for association in test_association ]
		expected_result = [['M3', 'M4', 0.6124], ['M3', 'M5', 0.4082], ['M3', 'M6', 0.25], ['M4', 'M5', 0.6667], ['M4', 'M6', 0.4082], ['M5', 'M6', 0.6124]]
		self.assertEqual(expected_result, test_association)

		test_association = self.network_obj.get_corr_associations(['main'], 'projection', corr_type = "pearson", pvalue = 0.5, pvalue_adj_method = "fdr_bh")
		test_association = [ [association[0], association[1], np.round(association[2],4)] for association in test_association ]
		expected_result = [['M3', 'M4', 0.6124], ['M3', 'M5', 0.4082], ['M3', 'M6', 0.25], ['M4', 'M5', 0.6667], ['M4', 'M6', 0.4082], ['M5', 'M6', 0.6124]]
		self.assertEqual(expected_result, test_association)

	def test_get_umap_association(self):
		test_association = self.network_obj.get_umap_associations(['main'], 'projection', n_neighbors = 2, random_seed = 47)
		expected_result =[['M1', 'M2', 0.6036601371481928], ['M1', 'M3', 0.5609716303441513],
		 ['M1', 'M4', 0.45384044421210484], ['M1', 'M5', 0.36606013513275965],
		 ['M1', 'M6', 0.30427961425589395], ['M2', 'M3', 0.4591326510409939],
		 ['M2', 'M4', 0.3547936225526581], ['M2', 'M5', 0.2951290158950945], 
		 ['M2', 'M6', 0.25372040357498143], ['M3', 'M4', 0.5392249401850056],
		 ['M3', 'M5', 0.3777011696740734], ['M3', 'M6', 0.3234901016988017],
		 ['M4', 'M5', 0.54646615122644], ['M4', 'M6', 0.44706939227564624], ['M5', 'M6', 0.6329591835808437]]
		print(test_association)
		self.assertEqual(expected_result, test_association)

	def test_get_pca_association(self):
		test_association = self.network_obj.get_pca_associations(['main'], 'projection', n_components=3, coords2sim_type="euclidean")
		expected_result = [['M1', 'M1', 1.0], ['M1', 'M2', 0.9999999999999996], ['M1', 'M3', 0.2632638076057898], ['M1', 'M4', 0.19952554249100315],
		 ['M1', 'M5', 0.1654311160526798], ['M1', 'M6', 0.1359425751823058], ['M2', 'M1', 0.9999999999999996],
		 ['M2', 'M2', 1.0], ['M2', 'M3', 0.2632638076057898], ['M2', 'M4', 0.1995255424910032], ['M2', 'M5', 0.16543111605267982],
		 ['M2', 'M6', 0.1359425751823058], ['M3', 'M1', 0.2632638076057898], ['M3', 'M2', 0.2632638076057898], ['M3', 'M3', 1.0], 
		 ['M3', 'M4', 0.33700718436347943], ['M3', 'M5', 0.19572890826736966], ['M3', 'M6', 0.15376771104889048], ['M4', 'M1', 0.19952554249100315],
		  ['M4', 'M2', 0.1995255424910032], ['M4', 'M3', 0.33700718436347943], ['M4', 'M4', 1.0], ['M4', 'M5', 0.2876176797272343], 
		  ['M4', 'M6', 0.17428658536674838], ['M5', 'M1', 0.1654311160526798], ['M5', 'M2', 0.16543111605267982], ['M5', 'M3', 0.19572890826736966], 
		  ['M5', 'M4', 0.2876176797272343], ['M5', 'M5', 1.0], ['M5', 'M6', 0.2115677806952532], ['M6', 'M1', 0.1359425751823058], 
		  ['M6', 'M2', 0.1359425751823058], ['M6', 'M3', 0.15376771104889048], ['M6', 'M4', 0.17428658536674838], ['M6', 'M5', 0.2115677806952532], ['M6', 'M6', 1.0]]
		self.assertEqual(expected_result, test_association)

	def test_get_counts_association(self):	
		test_association = self.network_obj.get_counts_associations(['main'], 'projection')
		test_association = [[a[0], a[1], a[2]] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'counts_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = int(fields.pop())
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_get_jaccard_association(self):
		test_association = self.network_obj.get_jaccard_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'jaccard_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_get_simpson_association(self):
		test_association = self.network_obj.get_simpson_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'simpson_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_get_geometric_association(self):
		test_association = self.network_obj.get_geometric_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'geometric_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_get_cosine_associations(self):
		test_association = self.network_obj.get_cosine_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'cosine_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_pcc_associations(self):
		test_association = self.network_obj.get_pcc_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'pcc_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values = [row for row in expected_values if not math.isnan(row[2])]
		expected_values.sort()
		test_association = [row for row in test_association if not math.isnan(row[2])]
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_clean_autorelations_on_association_values(self):
		self.tripartite_network.get_pcc_associations(['main','salient'], 'projection')
		self.tripartite_network.clean_autorelations_on_association_values()
		expected_values = [['M1', 'S1', float("nan")],
						   ['M2', 'S1', float("nan")],
  						   ['M2', 'S2', -0.5],
  						   ['M2', 'S3', 0.5],
  						   ['M3', 'S1', float("nan")],
  						   ['M3', 'S2', -0.5],
  						   ['M3', 'S3', 0.5]]
		test_values = self.tripartite_network.association_values['pcc']
		expected_values = [row for row in expected_values if not math.isnan(row[2])]
		expected_values.sort()
		test_values = [row for row in test_values if not math.isnan(row[2])]
		test_values.sort()
		self.assertEqual(expected_values, test_values)

	def test_hypergeometric_associations(self):
		test_association = self.network_obj.get_hypergeometric_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'hyi_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_csi_associations(self):
		test_association = self.network_obj.get_csi_associations(['main'], 'projection')
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'csi_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)

	def test_transference_associations(self):
		test_association = self.tripartite_network.get_association_by_transference_resources(('main','projection'), ('projection','salient'))
		test_association = [[a[0], a[1], round(a[2], 6)] for a in test_association]
		expected_values = []
		f = open(os.path.join(DATA_TEST_PATH, 'transference_results.txt'), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			association_value = round(float(fields.pop()), 6)
			expected_values.append([fields[0], fields[1], association_value])
		f.close()
		expected_values.sort()
		test_association.sort()
		self.assertEqual(expected_values, test_association)
	
	def test_adjust_pval_association_bonferroni(self):
		mock_hypergeo_assoc = [["A", "B", 0.01], ["A", "C", 0.05], ["A", "D", 0.6]]
		p_adjusted = multipletests([pvalue[2] for pvalue in mock_hypergeo_assoc], 
							 	method="bonferroni", is_sorted=False, returnsorted=False)[1]
		expected = [[nodeA, nodeB, p_adjusted[idx]] for idx, (nodeA, nodeB, pvalue) in enumerate(mock_hypergeo_assoc)]
		
		self.network_obj.adjust_pval_association(mock_hypergeo_assoc, "bonferroni") # This method modifies the list in place
		self.assertEqual(expected, mock_hypergeo_assoc)
		
	def test_adjust_pval_association_benjamini(self):
		mock_hypergeo_assoc = [["A", "B", 0.01], ["A", "C", 0.05], ["A", "D", 0.6]]
		p_adjusted = multipletests([pvalue[2] for pvalue in mock_hypergeo_assoc], 
							 	method="fdr_bh", is_sorted=False, returnsorted=False)[1]
		expected = [[nodeA, nodeB, p_adjusted[idx]] for idx, (nodeA, nodeB, pvalue) in enumerate(mock_hypergeo_assoc)]
		
		self.network_obj.adjust_pval_association(mock_hypergeo_assoc, "fdr_bh") # Same as benjamini in the test before
		self.assertEqual(expected, mock_hypergeo_assoc)

	def test_filter(self):
		# With same two layers
		monopartite_weighs_graph = self.monopartite_network_weights.clone()
		test_value = monopartite_weighs_graph.graph.edges(data=True)
		monopartite_weighs_graph.get_filter(("main","main"), method="cutoff", options={"cutoff": 5})
		test_value = monopartite_weighs_graph.graph.edges(data=True)
		expected = [('M3', 'M1', {'weight': 7.0})]
		self.assertEqual(expected, list(test_value))

		# Binarize option
		test_value = self.monopartite_network_weights.graph.edges(data=True)
		self.monopartite_network_weights.get_filter(("main","main"), method="cutoff", options={"cutoff": 5, "binarize": True})
		test_value = self.monopartite_network_weights.graph.edges(data=True)
		expected = [('M3', 'M1', {'weight': 1.0})]
		self.assertEqual(expected, list(test_value))

		# With different three layers
		self.tripartite_network_weights.graph.add_edge("M1","M2",weight=6)
		self.tripartite_network_weights.graph.add_edge("P1","P2",weight=3)
		self.tripartite_network_weights.graph.add_edge("S1","S2",weight=4)
		self.tripartite_network_weights.get_filter(("main","projection","salient"), method="cutoff", options={"cutoff": 5})
		expected = [('M1', 'P1', {'weight': 7.0}), ('P2', 'S1', {'weight': 5.0}), ('P2', 'S2', {'weight': 5.0}), ('S2', 'P3', {'weight': 6.0})]
		test_value = list(self.tripartite_network_weights.graph.edges(data=True))
		test_value.remove(("M1","M2",{'weight':6}))
		test_value.remove(("P1","P2",{'weight':3}))
		test_value.remove(("S1","S2",{'weight':4}))
		self.assertEqual(expected, test_value)

		# With different layers and autorelations
		self.tripartite_network_weights.get_filter(("main","projection","salient"), method="cutoff", options={"cutoff": 5, "compute_autorelations": True})
		expected = [('M1', 'P1', {'weight': 7.0}), ('M1', 'M2', {'weight': 6}), ('P2', 'S1', {'weight': 5.0}), ('P2', 'S2', {'weight': 5.0}), ('S2', 'P3', {'weight': 6.0})]
		test_value = list(self.tripartite_network_weights.graph.edges(data=True))
		self.assertEqual(expected, list(test_value))


	def test_cutoff_filter(self):
		self.tripartite_network_weights.graph.add_edge("P1","P2",weight=3)
		filtered_relations = self.tripartite_network_weights.filter_cutoff(("projection", "salient"), cutoff=5, compute_autorelations= False)
		expected_relations =   [['P1', 'S1', 0],
								['P2', 'S1', 5.0],
								['P2', 'S2', 5.0],
								['P2', 'S3', 0],
								['P3', 'S1', 0],
								['P3', 'S2', 6.0]]
		self.assertEqual(expected_relations, filtered_relations)

		filtered_relations = self.tripartite_network_weights.filter_cutoff(("projection", "salient"), cutoff=5, compute_autorelations= True)
		expected_relations = [['S3', 'P2', 0], ['P1', 'S1', 0], ['P1', 'P2', 0], ['S2', 'P3', 6.0], ['S2', 'P2', 5.0], ['S1', 'P3', 0], ['S1', 'P2', 5.0]]
		self.assertEqual(expected_relations.sort(), filtered_relations.sort())


	def test_write_subgraph(self):
		output_filename = "subgraph.txt"
		outFormat = "pair"
		self.monopartite_network_weights.write_subgraph(("main","main"), output_filename, outFormat)
		graph_to_test = Net_parser.load_network_by_pairs(output_filename, self.monopartite_layers)
		self.assertEqual(list(self.monopartite_network_weights.graph.edges(data=True)), list(graph_to_test.graph.edges(data=True)))
		os.remove(output_filename)

		outFormat = "matrix"
		output_filename = "subgraph"
		self.monopartite_network_weights.write_subgraph(("main","main"), output_filename, outFormat)	
		self.assertEqual(pxc.dig(self.monopartite_network_weights.matrices,*("adjacency_matrices",("main","main")))[0].tolist(), np.load("subgraph.npy").tolist())
		self.assertTrue(os.path.exists(output_filename+"_colIds"))
		self.assertTrue(os.path.exists(output_filename+"_rowIds"))
		os.remove(output_filename+"_colIds")
		os.remove(output_filename+"_rowIds")
		os.remove(output_filename+".npy")

	def test_get_direct_conns(self):
		self.network_obj.graph.add_edge("M1", "M2")
		# No autorelations
		test_value = self.network_obj.get_direct_conns(pair_operation = lambda x,y: (x,y), layers = ('main','projection'), compute_autorelations = False)
		test_value = sorted([sorted(set(pair)) for pair in test_value])
		expected = list(self.network_obj.graph.edges())
		expected.remove(("M1","M2"))
		expected = sorted([sorted(set(pair)) for pair in expected])
		self.assertEqual(expected, test_value)
		# With autorelations
		test_value = self.network_obj.get_direct_conns(pair_operation = lambda x,y: (x,y), layers = ('main','projection'), compute_autorelations = True)
		test_value = sorted(set([tuple(sorted(set(pair))) for pair in test_value]))
		expected = list(self.network_obj.graph.edges())
		expected = sorted(set([tuple(sorted(set(pair))) for pair in expected]))
		self.assertEqual(expected, test_value)

	# Testing matrix operations: Filters
	def test_filter_matrix(self):
		mat_keys = ("adjacency_matrices", ("main","main"))
		# Checking cutoff filter.
		operation = "filter_cutoff"
		expected = [[0.0, 2.0, 7.0], [2.0, 0.0, 3.0], [7.0, 3.0, 0.0]]
		options = {"cutoff": 2}
		expected_cols = ['M3', 'M2', 'M1']
		expected_rows = ['M3', 'M2', 'M1']
		test_value, row_value, col_value = self.monopartite_network_weights.filter_matrix(mat_keys, operation, options)
		self.assertEqual(expected, test_value.tolist())
		self.assertListEqual(expected_cols, col_value)
		self.assertListEqual(expected_rows, row_value)
		# Checking disparity filter.
		operation = "filter_disparity"
		options = {"pval_threshold": 0.3}
		mat_keys = ("adjacency_matrices", ("new","new"))
		expected = [[0., 0, 7, 0],
					[0, 0, 0, 7],
					[7, 0, 0, 0],
					[0, 7, 0, 0]]
		expected_rows = ['M1', 'M2','M3', 'M4']
		expected_cols = ['M1', 'M2','M3', 'M4']
		self.monopartite_network_weights.matrices["adjacency_matrices"][("new","new")] = [np.array([[0,3,7,0],[3,0,2,7],[7,2,0,1],[0,7,1,0]]),["M1","M2","M3","M4"],["M1","M2","M3","M4"]]
		
		test_value, row_value, col_value = self.monopartite_network_weights.filter_matrix(mat_keys, operation, options)
		self.assertEqual(expected, test_value.round(6).tolist())
		self.assertListEqual(expected_cols, col_value)
		self.assertListEqual(expected_rows, row_value)

		# Checking disparity filter.
		operation = "filter_disparity"
		options = {"pval_threshold": 0.3}
		mat_keys = ("adjacency_matrices", ("new","new"))
		expected = [[0, 2, 7], [2, 0, 0], [7, 0, 0]]
		expected_rows = ['M2','M3', 'M4']
		expected_cols = ['M2','M3', 'M4'] 
		self.monopartite_network_weights.matrices["adjacency_matrices"][("new","new")] = [np.array([[0,1,1,0],[1,0,2,7],[1,2,0,1],[0,7,1,0]]),["M1","M2","M3","M4"],["M1","M2","M3","M4"]]
		
		test_value, row_value, col_value = self.monopartite_network_weights.filter_matrix(mat_keys, operation, options)
		self.assertEqual(expected, test_value.round(6).tolist())
		self.assertListEqual(expected_cols, col_value)
		self.assertListEqual(expected_rows, row_value)


	
	# Testing community discovery
	def test_community_discovery(self):
		self.comunities_network_no_coms.discover_clusters("louvain",{})
		returned = list(self.comunities_network_no_coms.group_nodes.items())
		expected = [('0', ['A', 'B', 'C', 'D', 'E', 'F']), ('1', ['X', 'Y', 'V', 'W']), ('2', ['Z', 'L', 'M', 'N'])]
		self.assertEqual(expected, returned)


	# Testing Family community comparation
	def test_compute_comparative_community(self):
		com1_alt = [node for node in self.comunities_network.graph.nodes if node in "ABDF"]		
		com2_alt = [node for node in self.comunities_network.graph.nodes if node in "ZLCENM"]
		ref_com = {'com1_alt': com1_alt, 'com2_alt': com2_alt}
		self.comunities_network.group_reference = ref_com
		res = self.comunities_network.compare_partitions()
		returned = round(res["mutual_information"],5)
		expected = 0.43254
		self.assertEqual(expected, returned)
	

	# Testing comunities methods
	def test_compute_comparative_degree(self):
		internal, external = (5+5+5+5+5+5, 2+1+1+0+0+0) 
		expected = external / (internal + external)
		returned = self.comunities_network.compute_comparative_degree(self.comunities_network.group_nodes["com1"])
		
		internal2, external2 = (3+3+3+3, 0+0+0+2)
		expected2 = external2 / (internal2 + external2)
		returned2 = self.comunities_network.compute_comparative_degree(self.comunities_network.group_nodes["com2"])

		self.assertEqual(expected, returned)
		self.assertEqual(expected2, returned2)

	
	def test_compute_node_com_assoc(self):
		ref_edges, secondary_edges, other_edges = (1+0+0+0+0+0, 1+2+2+1+1+1, 5+4+4+4+4+4)		
		secondary_nodes, other_nodes = (2, 5)
		by_edge, by_node = ((ref_edges + secondary_edges) / other_edges, (ref_edges + secondary_nodes) / other_nodes) 
		expected = [by_edge, by_node]
		returned = self.comunities_network.compute_node_com_assoc(self.comunities_network.group_nodes["com1"], "X")

		ref_edges2, secondary_edges2, other_edges2 = (1+0+0+0, 1+1+1+1, 3+2+2+2)		
		secondary_nodes2, other_nodes2 = (2, 3)
		by_edge2, by_node2 = ((ref_edges2 + secondary_edges2) / other_edges2, (ref_edges2 + secondary_nodes2) / other_nodes2)
		expected2 = [by_edge2, by_node2]
		returned2 = self.comunities_network.compute_node_com_assoc(self.comunities_network.group_nodes["com2"], "X")

		self.assertEqual(set(expected), set(returned))
		self.assertEqual(set(expected2), set(returned2))		


	# Testing iterative comunities methods
	
	def test_communities_avg_sht_path(self):
		# TODO: Check this examples manually.
		paths_clust1 = 1.73
		paths_clust2 =  2.64
		
		expected = [paths_clust1, paths_clust2] #Cluster1 and Cluster2 average shortest path
		returned = self.clusters_network.communities_avg_sht_path(self.clusters_network.group_nodes)
		self.assertEqual(sorted(expected), sorted([round(returned[0],2),round(returned[1],2)]))
	
	def test_communities_comparative_degree(self):
		internal, external = (5+5+5+5+5+5, 2+1+1+0+0+0) 
		expected_com1 = external / (internal + external)
		internal2, external2 = (3+3+3+3, 0+0+0+2)
		expected_com2 = external2 / (internal2 + external2)
		
		returned = self.comunities_network.communities_comparative_degree(self.comunities_network.group_nodes)
		self.assertEqual(set([expected_com1, expected_com2]), set(returned))

	def test_communities_node_com_assoc(self):
		ref_edges, secondary_edges, other_edges = (1+0+0+0+0+0, 1+2+2+1+1+1, 5+4+4+4+4+4)		
		secondary_nodes, other_nodes = (2, 5)
		by_edge, by_node = ((ref_edges + secondary_edges) / other_edges, (ref_edges + secondary_nodes) / other_nodes) 
		expected = [by_edge, by_node]
	
		ref_edges2, secondary_edges2, other_edges2 = (1+0+0+0, 1+1+1+1, 3+2+2+2)		
		secondary_nodes2, other_nodes2 = (2, 3)
		by_edge2, by_node2 = ((ref_edges2 + secondary_edges2) / other_edges2, (ref_edges2 + secondary_nodes2) / other_nodes2)
		expected2 = [by_edge2, by_node2]

		returned = self.comunities_network.communities_node_com_assoc(self.comunities_network.group_nodes, "X")
		self.assertEqual([expected, expected2], returned)


	def test_compute_group_metrics(self):
		expected = {
			"com1": {
			  'average_internal_degree': 5.0,
			  "comparative_degree": self.comunities_network.compute_comparative_degree(self.comunities_network.group_nodes["com1"]), 
			  "avg_sht_path": self.comunities_network.average_shortest_path_length(self.comunities_network.group_nodes["com1"]), 
			  "node_com_assoc_by_edge":	self.comunities_network.compute_node_com_assoc(self.comunities_network.group_nodes["com1"], "X")[0],
			  "node_com_assoc_by_node": self.comunities_network.compute_node_com_assoc(self.comunities_network.group_nodes["com1"], "X")[1]}, 
			"com2": {
			  'average_internal_degree': 3.0,
			  "comparative_degree":	self.comunities_network.compute_comparative_degree(self.comunities_network.group_nodes["com2"]), 
			  "avg_sht_path": self.comunities_network.average_shortest_path_length(self.comunities_network.group_nodes["com2"]), 
			  "node_com_assoc_by_edge":	self.comunities_network.compute_node_com_assoc(self.comunities_network.group_nodes["com2"], "X")[0],
			  "node_com_assoc_by_node":	self.comunities_network.compute_node_com_assoc(self.comunities_network.group_nodes["com2"], "X")[1]}
			}
		
		self.comunities_network.compute_group_metrics(os.path.join(DATA_TEST_PATH, "group_metrics_test.tsv"),['comparative_degree', 'avg_sht_path', 'node_com_assoc','average_internal_degree'])
		
		reread_metrics = {}
		with open(os.path.join(DATA_TEST_PATH, "group_metrics_test.tsv"), "r") as f:
			header = []
			comunity_id = ""
			for line, content in enumerate(f):
				if line == 0:
					header = content.strip().split("\t")
				else:
					values = content.strip().split("\t")
					for pos, value in enumerate(values):
						if pos == 0:
							comunity_id = value
							reread_metrics[comunity_id] = {}
						else:
							reread_metrics[comunity_id][header[pos]] = float(value)
		
		self.assertEqual(expected, reread_metrics)
		os.remove(os.path.join(DATA_TEST_PATH, "group_metrics_test.tsv"))

	def test_compute_group_metrics_summarized(self):
		expected = {'max_odf': {'Max': 2.0, 'Mean': 2.0, 'Min': 2.0, 'Std': 0.0}, 
		               'size': {'Max': 6.0, 'Mean': 5.0, 'Min': 4.0, 'Std': 1.0}}
		
		self.comunities_network.compute_summarized_group_metrics(os.path.join(DATA_TEST_PATH, "group_metrics_test.tsv"),['size','max_odf'])
		
		reread_metrics = {}
		with open(os.path.join(DATA_TEST_PATH, "group_metrics_test.tsv"), "r") as f:
			header = []
			comunity_id = ""
			for line, content in enumerate(f):
				if line == 0:
					header = content.strip().split("\t")
				else:
					values = content.strip().split("\t")
					for pos, value in enumerate(values):
						if pos == 0:
							comunity_id = value
							reread_metrics[comunity_id] = {}
						else:
							reread_metrics[comunity_id][header[pos]] = float(value)
		
		self.assertEqual(expected, reread_metrics)
		os.remove(os.path.join(DATA_TEST_PATH, "group_metrics_test.tsv"))

	def test_expand_clusters(self):
		expected_expanded_cluster_1 = set(["A","B","C","D","E","F","G","H","I","J"]) | set(["X","Y","Z"])
		expected_expanded_cluster_2 = set(["M","N","O","P","Q","R","W"]) | set(["W", "V"])
		
		returned = self.clusters_network.expand_clusters("sht_path")
		self.assertEqual(expected_expanded_cluster_1, set(returned["clust1"]))
		self.assertEqual(expected_expanded_cluster_2, set(returned["clust2"]))	

	# Random network generation
	def test_randomize_monopartite_net_by_nodes(self):
		random.seed(1)
		random_net = self.monopartite_network.randomize_monopartite_net_by_nodes()
		self.assertNotEqual(self.monopartite_network.get_degree(zscore = False), random_net.get_degree(zscore = False)) # Node degree is diferent from original tu random
		self.assertEqual( # but the node distribution is the same
			list(self.monopartite_network.get_degree(zscore = False).values()).sort(), 
			list(random_net.get_degree(zscore = False).values()).sort())
		
	def test_randomize_monopartite_net_by_links(self):
		random.seed(1)
		previous_degree = self.monopartite_network.get_degree(zscore = False)
		random_net = self.monopartite_network.randomize_monopartite_net_by_links()
		random_degree = random_net.get_degree(zscore = False)
		self.assertEqual(previous_degree, random_degree) # Degree for each node must be the same
		self.assertNotEqual(self.monopartite_network.graph.edges, random_net.graph.edges) # But the overall edge set must be different

	# def test_randomize_bipartite_net_by_nodes(self): # TODO PSZ: Generalize from monopartite and study the best netowrkx object manipulation
	# 	layerA_nodes = len(self.network_obj.get_nodes_layer([self.bipartite_layers[0][0]]))
	# 	layerB_nodes = len(self.network_obj.get_nodes_layer([self.bipartite_layers[1][0]]))
	# 	edges = self.network_obj.get_edge_number()
	# 	random_net = self.network_obj.randomize_bipartite_net_by_nodes()
	# 	random_layerA_nodes = len(random_net.get_nodes_layer([self.bipartite_layers[0][0]]))
	# 	random_layerB_nodes = len(random_net.get_nodes_layer([self.bipartite_layers[1][0]]))
	# 	random_edges = random_net.get_edge_number()
	# 	self.assertEqual([layerA_nodes, layerB_nodes, edges], [random_layerA_nodes, random_layerB_nodes, random_edges])
	# 	self.assertNotEqual(self.monopartite_network.get_degree(zscore = False), random_net.get_degree(zscore = False))

	# def test_randomize_bipartite_net_by_links(self): # TODO PSZ: Generalize from monopartite and study the best netowrkx object manipulation
	# 	previous_degree = self.network_obj.get_degree(zscore = False)
	# 	self.network_obj.randomize_bipartite_net_by_links([ l[0] for l in self.bipartite_layers ])
	# 	random_degree = self.network_obj.get_degree(zscore = False)
	# 	self.assertEqual(previous_degree, random_degree)
	