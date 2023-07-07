import unittest
import os
import numpy
from NetAnalyzer import Net_parser
import networkx as nx
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

class NetworkParserTestCase(unittest.TestCase):

	def setUp(self):
		self.bipartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]
		self.bipartite_network_path = os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt')
		self.monopartite_layers = [['main', 'M[0-9]+'], ['main', 'M[0-9]+']]
		self.monopartite_network_node_names = os.path.join(DATA_TEST_PATH, 'monopartite_network_node_names.txt')

	def test_load_pairs(self):
		options = {
			'input_format' : 'pair', 'split_char' : "\t",
			'input_file' : self.bipartite_network_path, 
			'layers' : self.bipartite_layers}
		network_obj = Net_parser.load(options)
		test_main_layer = network_obj.get_nodes_layer(['main'])
		self.assertEqual(6, len(test_main_layer))
		test_projection_layer = network_obj.get_nodes_layer(['projection'])
		self.assertEqual(10, len(test_projection_layer))
		test_connections = network_obj.get_edge_number()
		self.assertEqual(40, test_connections)

	def test_load_network_by_pairs(self):
		# Non weighted
		network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), self.bipartite_layers)
		test_main_layer = network_obj.get_nodes_layer(['main'])
		self.assertEqual(6, len(test_main_layer))
		test_projection_layer = network_obj.get_nodes_layer(['projection'])
		self.assertEqual(10, len(test_projection_layer))
		test_connections = network_obj.get_edge_number()
		self.assertEqual(40, test_connections)

		## Weighted
		network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'monopartite_network_weights_for_validating.txt'), self.monopartite_layers)
		test =  nx.get_edge_attributes(network_obj.graph,"weight")
		expected = {('M3', 'M4'): 1.0, ('M3', 'M2'): 2.0, ('M3', 'M1'): 7.0, ('M2', 'M1'): 3.0}
		self.assertEqual(expected, test)


	def test_load_bin_matrix(self):
		options = {'input_format' : 'bin', 'input_file' : os.path.join(DATA_TEST_PATH, 'monopartite_network_bin_matrix.npy'), 'layers' : self.monopartite_layers, 'node_files' : [self.monopartite_network_node_names]}
		monopartite_network_by_bin_matrix = Net_parser.load(options)
		test_adjacency, text_x_axis, text_y_axis = monopartite_network_by_bin_matrix.matrices["adjacency_matrices"][('main', 'main')]
		expected_adjacency_matrix = numpy.array([[0, 1, 1, 0, 0],[1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 0]])
		expected_axis =  ['A', 'B', 'C', 'D', 'E']
		self.assertEqual(expected_axis, text_x_axis)
		self.assertEqual(expected_axis, text_y_axis)
		self.assertTrue((expected_adjacency_matrix == test_adjacency).all())

	def test_load_network_by_bin_matrix(self):
		monopartite_network_by_bin_matrix = Net_parser.load_network_by_bin_matrix(os.path.join(DATA_TEST_PATH, 'monopartite_network_bin_matrix.npy'), [self.monopartite_network_node_names], self.monopartite_layers)
		test_adjacency, text_x_axis, text_y_axis = monopartite_network_by_bin_matrix.matrices["adjacency_matrices"][('main', 'main')]
		expected_adjacency_matrix = numpy.array([[0, 1, 1, 0, 0],[1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 0]])
		expected_axis =  ['A', 'B', 'C', 'D', 'E']
		self.assertEqual(expected_axis, text_x_axis)
		self.assertEqual(expected_axis, text_y_axis)
		self.assertTrue((expected_adjacency_matrix == test_adjacency).all())

	def test_load_plain_matrix(self):
		options = {'input_format' : 'matrix', 'input_file' : os.path.join(DATA_TEST_PATH, 'monopartite_network_matrix'), 'splitChar' : "\t", 'layers' : self.monopartite_layers, 'node_files' : [self.monopartite_network_node_names]}
		monopartite_network_by_plain_matrix = Net_parser.load(options)
		test_adjacency, text_x_axis, text_y_axis = monopartite_network_by_plain_matrix.matrices["adjacency_matrices"][('main', 'main')]
		expected_adjacency_matrix = numpy.array([[0, 0, 1, 0, 1],[0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 1, 0]])
		expected_axis =  ['A', 'B', 'C', 'D', 'E']
		self.assertEqual(expected_axis, text_x_axis)
		self.assertEqual(expected_axis, text_y_axis)
		self.assertTrue((expected_adjacency_matrix == test_adjacency).all())

	def test_load_network_by_plain_matrix(self):
		monopartite_network_by_plain_matrix = Net_parser.load_network_by_plain_matrix(os.path.join(DATA_TEST_PATH, 'monopartite_network_matrix'), [self.monopartite_network_node_names], self.monopartite_layers)
		test_adjacency, text_x_axis, text_y_axis = monopartite_network_by_plain_matrix.matrices["adjacency_matrices"][('main', 'main')]
		expected_adjacency_matrix = numpy.array([[0, 0, 1, 0, 1],[0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 1, 0]])
		expected_axis =  ['A', 'B', 'C', 'D', 'E']
		self.assertEqual(expected_axis, text_x_axis)
		self.assertEqual(expected_axis, text_y_axis)
		self.assertTrue((expected_adjacency_matrix == test_adjacency).all())