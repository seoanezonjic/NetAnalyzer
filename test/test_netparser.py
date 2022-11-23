import unittest
import os
from NetAnalyzer import Net_parser
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

class NetworkParserTestCase(unittest.TestCase):

	def test_load_pairs(self):
		options = {
			'input_format' : 'pair', 'split_char' : "\t",
			'input_file' : os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), 
			'layers' : [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]}
		network_obj = Net_parser.load(options)
		test_main_layer = network_obj.get_nodes_layer(['main'])
		self.assertEqual(6, len(test_main_layer))
		test_projection_layer = network_obj.get_nodes_layer(['projection'])
		self.assertEqual(10, len(test_projection_layer))
		test_connections = network_obj.get_edge_number()
		self.assertEqual(40, test_connections)
