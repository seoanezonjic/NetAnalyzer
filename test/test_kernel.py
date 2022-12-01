import unittest
import os
from NetAnalyzer import NetAnalyzer
from NetAnalyzer import Net_parser
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data/data_kernel')

class KernelTestCase(unittest.TestCase):
	def setUp(self):
		bipartite_layers = [['main', 'M[0-9]+'], ['projection', 'P[0-9]+']]
		self.network_obj = Net_parser.load_network_by_pairs(os.path.join(DATA_TEST_PATH, 'bipartite_network_for_validating.txt'), bipartite_layers)

