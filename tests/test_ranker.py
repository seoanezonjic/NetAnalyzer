import unittest
import os
import math
import numpy as np
from NetAnalyzer import NetAnalyzer
from NetAnalyzer import Net_parser
from NetAnalyzer import Ranker
#from NetAnalyzer import ranker
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data', 'data_ranker')

class BaseNetTestCase(unittest.TestCase):

	def load_results(self, file_name):
		validate_ranked_genes = []
		f = open(os.path.join(DATA_TEST_PATH, file_name), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			validate_ranked_genes.append(fields)
		f.close()
		return validate_ranked_genes

	def ranked_genes2array(self, ranked_genes):
		test_ranked_genes = []
		for seed_name, rank_list in ranked_genes.items():
			for ranked_gene in rank_list:
				test_ranked_genes.append([str(el) for el in ranked_gene] + [seed_name])
		return test_ranked_genes

	def load_ranking(self, file_name):
		ranked_genes = {}
		f = open(os.path.join(DATA_TEST_PATH, file_name), 'r')
		for line in f:
			fields = line.rstrip().split("\t")
			seed_name = fields.pop(0)
			values = [row.split(",") for row in fields[0].split(";")]
			for row in values:
				for idx, val in enumerate(row):
					if idx == 0:
						row[idx] = val 
					elif idx >= 3: 
						row[idx] = int(row[idx])
					else:
						row[idx] = float(row[idx])

			ranked_genes[seed_name] = values
		f.close()
		return ranked_genes

	def setUp(self):
		self.ranker = Ranker()
		self.ranker.matrix = np.load(os.path.join(DATA_TEST_PATH, 'kernel_for_validating'))
		self.ranker.load_seeds(os.path.join(DATA_TEST_PATH, 'seed_genes_for_validating'), sep= ",") # Should be in its test method but modifications are deleted from one test to another
		self.ranker.load_nodes_from_file(os.path.join(DATA_TEST_PATH, 'kernel_for_validating.lst')) # Should be in its test method but modifications are deleted from one test to another

		self.ranker_with_ranking = Ranker()
		self.ranker_with_ranking.load_references(os.path.join(DATA_TEST_PATH, 'genes2filter_for_validating'))
		self.ranker_with_ranking.ranking = self.load_ranking(os.path.join(DATA_TEST_PATH, 'ranked_genes'))

	def test_load_nodes_from_file(self):
		self.assertEqual(["A","B","C","D","E"], self.ranker.nodes)

	def test_load_seeds(self):
		validate_seed_genes_loaded = {'toy_group1': ["A","B"],'toy_group2': ["C","D","E"],'toy_group3': ["A","D"]}
		self.assertEqual(validate_seed_genes_loaded, self.ranker.seeds)

	def test_get_seed_indexes(self):
		validate_seed_indexes={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
		self.assertEqual(validate_seed_indexes, self.ranker.get_seed_indexes())

	def test_leave_one_out(self):
		self.ranker.do_ranking(cross_validation = True, k_fold = None)
		test_ranked_genes = self.ranked_genes2array(self.ranker.ranking)
		self.assertEqual(self.load_results('leave_one_out_by_seedgene_results'), test_ranked_genes)

	def test_10Fold_CV(self):
		self.ranker.do_ranking(cross_validation = True, k_fold = 2)
		test_ranked_genes = self.ranked_genes2array(self.ranker.ranking)
		self.assertEqual(self.load_results('cross_validation_by_seedgene_results'), test_ranked_genes)

	def test_rank_by_seed(self):
		self.ranker.do_ranking()
		test_ranked_genes = self.ranked_genes2array(self.ranker.ranking)
		self.assertEqual(self.load_results('rank_by_seedgene_results'), test_ranked_genes)

	def test_get_filtered(self):
		test_filtered_ranked_genes = self.ranked_genes2array(self.ranker_with_ranking.get_reference_ranks())
		self.assertEqual(self.load_results('filter_results'), test_filtered_ranked_genes)

	def test_get_top(self):
		test_top_ranked_genes = self.ranked_genes2array(self.ranker_with_ranking.get_top(2))
		self.assertEqual(self.load_results('top_results'),test_top_ranked_genes)

	def test_filtered_top_compatibility(self):
		filtered_ranked_genes = self.ranked_genes2array(self.ranker_with_ranking.get_reference_ranks())
		test_top_ranked_genes = self.ranked_genes2array(self.ranker_with_ranking.get_top(2))
		self.assertEqual(self.load_results('top_results'),test_top_ranked_genes)



