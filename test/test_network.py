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
		tripartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+'], ['salient', 'S[0-9]+']]
		self.tripartite_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'tripartite_network_for_validating.txt'), tripartite_layers)
		self.tripartite_network.generate_adjacency_matrix(tripartite_layers[0][0], tripartite_layers[1][0])
		self.tripartite_network.generate_adjacency_matrix(tripartite_layers[1][0], tripartite_layers[2][0])
		
		bipartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]
		self.network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), bipartite_layers)
		self.network_obj.generate_adjacency_matrix(bipartite_layers[0][0], bipartite_layers[1][0])
		
		monopartite_layers = [['main', '\w'], ['main', '\w']]
		self.monopartite_network = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'monopartite_network_for_validating.txt'), monopartite_layers)
		self.monopartite_network.generate_adjacency_matrix(monopartite_layers[0][0], monopartite_layers[0][0])

	def test_clone(self):
		network_clone = self.network_obj.clone()
		self.assertEqual(self.network_obj, network_clone)

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