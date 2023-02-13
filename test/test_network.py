import sys 
import unittest
import os
import math
import numpy as np
from NetAnalyzer import NetAnalyzer
from NetAnalyzer import Net_parser
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

class BaseNetTestCase(unittest.TestCase):
	def setUp(self):
		self.tripartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+'], ['salient', 'S[0-9]+']]
		self.tripartite_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'tripartite_network_for_validating.txt'), self.tripartite_layers)
		self.tripartite_network.generate_adjacency_matrix(self.tripartite_layers[0][0], self.tripartite_layers[1][0])
		self.tripartite_network.generate_adjacency_matrix(self.tripartite_layers[1][0], self.tripartite_layers[2][0])
		
		self.bipartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]
		self.network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), self.bipartite_layers)
		self.network_obj.generate_adjacency_matrix(self.bipartite_layers[0][0], self.bipartite_layers[1][0])
		
		self.monopartite_layers = [['main', '\w'], ['main', '\w']]
		self.monopartite_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'monopartite_network_for_validating.txt'), self.monopartite_layers)
		self.monopartite_network.generate_adjacency_matrix(self.monopartite_layers[0][0], self.monopartite_layers[0][0])

	def test_clone(self):
		network_clone = self.network_obj.clone()
		self.assertEqual(self.network_obj, network_clone)

	def test_clone_change(self):
		network_clone = self.network_obj.clone()
		network_clone.add_node('M8', network_clone.set_layer(self.bipartite_layers, 'M8'))
		self.assertNotEqual(self.network_obj, network_clone	)
	
	def test_generate_adjacency_matrix_monopartite(self):
		test_values = self.monopartite_network.adjacency_matrices
		matrix_values = np.array([[0, 1, 1, 0, 0],[1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 0]],dtype='float')
		expected_values = { ('main') : [matrix_values, ['A', 'C', 'E', 'B', 'D'], ['A', 'C', 'E', 'B', 'D']]} 
		self.assertEqual(expected_values[('main')][0].tolist(), test_values[('main')][0].tolist())
		self.assertEqual(expected_values[('main')][1], test_values[('main')][1])

	def test_generate_adjacency_matrix_bipartite(self):
		test_values = self.network_obj.adjacency_matrices
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
		self.assertEqual(['P1', 'P2'], list(test_result))

	def test_get_node_attributes(self):
		node_attribute_test = self.monopartite_network.get_node_attributes(['get_degree'])
		expected_result = [['A', 2], ['C', 1], ['E', 2],['B', 1], ['D', 2]]
		self.assertEqual(expected_result, node_attribute_test)

	def test_get_node_attributes_zscore(self):
		node_attribute_test = self.monopartite_network.get_node_attributes(['get_degreeZ', 'get_degree'])
		expected_result = [['A', 0.8164965809277259, 2], ['C', -1.2247448713915894, 1], ['E', 0.8164965809277259, 2],['B', -1.2247448713915894, 1], ['D', 0.8164965809277259, 2]]
		self.assertEqual(expected_result, node_attribute_test)

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

	# Random network generation
	def test_randomize_monopartite_net_by_nodes(self):
		random_net = self.monopartite_network.randomize_monopartite_net_by_nodes()
		self.assertNotEqual(self.monopartite_network.get_degree(zscore = False), random_net.get_degree(zscore = False)) # Node degree is diferent from original tu random
		self.assertEqual( # but the node distribution is the same
			list(self.monopartite_network.get_degree(zscore = False).values()).sort(), 
			list(random_net.get_degree(zscore = False).values()).sort())

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

	def test_randomize_monopartite_net_by_links(self):
		previous_degree = self.monopartite_network.get_degree(zscore = False)
		random_net = self.monopartite_network.randomize_monopartite_net_by_links()
		random_degree = random_net.get_degree(zscore = False)
		self.assertEqual(previous_degree, random_degree) # Degree for each node must be the same

	# def test_randomize_bipartite_net_by_links(self): # TODO PSZ: Generalize from monopartite and study the best netowrkx object manipulation
	# 	previous_degree = self.network_obj.get_degree(zscore = False)
	# 	self.network_obj.randomize_bipartite_net_by_links([ l[0] for l in self.bipartite_layers ])
	# 	random_degree = self.network_obj.get_degree(zscore = False)
	# 	self.assertEqual(previous_degree, random_degree)
	