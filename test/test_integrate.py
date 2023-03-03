import unittest
import os
from NetAnalyzer import Kernels
import numpy as np
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data/data_integrate')

class KernelTestCase(unittest.TestCase):
	def setUp(self):

		self.kernels = Kernels()
		matrixes_path = [os.path.join(DATA_TEST_PATH, "kernel1.npy"),os.path.join(DATA_TEST_PATH, "kernel2.npy")]
		nodes_path = [os.path.join(DATA_TEST_PATH, "kernel1.lst"),os.path.join(DATA_TEST_PATH, "kernel2.lst")]
		self.kernels.load_kernels_by_bin_matrixes(matrixes_path, nodes_path, ["1", "2"])
	

	def test_load_kernels_by_bin_matrixes(self):
		self.assertTrue((np.array([[1, 1, 1], [1, 3, 0],[1, 0, 0]]) == self.kernels.kernels_raw[0]).all())
		self.assertTrue((np.array([[0, 5],[5, 7]]) == self.kernels.kernels_raw[1]).all())
		self.assertEqual([{'Node1': 0, 'Node2': 1, 'Node3': 2}, {'Node3': 1, 'Node4': 0}], self.kernels.local_indexes)

	def test_create_general_index(self):
		self.kernels.create_general_index()
		self.assertEqual({'Node1': [0, None], 'Node2': [1, None], 'Node3': [2, 1], 'Node4': [None, 0]}, self.kernels.kernels_position_index)

	def test_get_values(self):
		self.kernels.create_general_index()
		self.assertEqual([5], self.kernels.get_values("Node3","Node4"))
		self.assertEqual([1], self.kernels.get_values("Node1","Node2"))
		self.assertEqual([1], self.kernels.get_values("Node1","Node3"))
		self.assertEqual([], self.kernels.get_values("Node2","Node4"))
		self.assertEqual([], self.kernels.get_values("Node1","Node4"))

	def test_integrate(self):
		self.kernels.create_general_index()
		self.kernels.integrate_matrix("mean")
		print(self.kernels.integrated_kernel[0])
		print(self.kernels.integrated_kernel[1])
		self.assertEqual([],self.kernels.integrated_kernel[0])
		self.assertEqual([],self.kernels.integrated_kernel[1])
