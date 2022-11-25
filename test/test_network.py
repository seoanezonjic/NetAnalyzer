import unittest
import os
from NetAnalyzer import NetAnalyzer
from NetAnalyzer import Net_parser
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

class BaseNetTestCase(unittest.TestCase):

	def test_get_counts_association(self):
		bipartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]
		network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), bipartite_layers)
		test_association = network_obj.get_counts_association(['main'], 'projection')
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
