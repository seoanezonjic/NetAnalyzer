import unittest
import os
import timeit
from NetAnalyzer import Kernels
import numpy as np
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data/data_integrate')

def profiler(funcion, method, cores, symmetry):
	start = timeit.default_timer()
	funcion(method, n_workers = cores, symmetry = symmetry)
	stop = timeit.default_timer()
	return (stop - start)

class KernelTestCase(unittest.TestCase):
	def setUp(self):

		self.kernels = Kernels()
		self.asym_kernels = Kernels()
		self.negatives_kernels = Kernels()
		matrixes_path = [os.path.join(DATA_TEST_PATH, "kernel1.npy"),os.path.join(DATA_TEST_PATH, "kernel2.npy")]
		asym_matrixes_path = [os.path.join(DATA_TEST_PATH, "asym_kernel1.npy"),os.path.join(DATA_TEST_PATH, "asym_kernel2.npy")]
		negatives_matrixes_path = [os.path.join(DATA_TEST_PATH, "negative_kernel1.npy"),os.path.join(DATA_TEST_PATH, "negative_kernel2.npy")]
		nodes_path = [os.path.join(DATA_TEST_PATH, "kernel1.lst"),os.path.join(DATA_TEST_PATH, "kernel2.lst")]
		self.kernels.load_kernels_by_bin_matrixes(matrixes_path, nodes_path, ["1", "2"])
		self.asym_kernels.load_kernels_by_bin_matrixes(asym_matrixes_path, nodes_path, ["1", "2"])
		self.negatives_kernels.load_kernels_by_bin_matrixes(negatives_matrixes_path, nodes_path, ["1", "2"])
	
	def test_load_kernels_by_bin_matrixes(self):
		#Testing loading symmetric kernels
		self.assertTrue((np.array([[1, 1, 1], [1, 3, 0],[1, 0, 0]]) == self.kernels.kernels_raw[0]).all())
		self.assertTrue((np.array([[0, 5],[5, 7]]) == self.kernels.kernels_raw[1]).all())
		self.assertEqual([{'Node1': 0, 'Node2': 1, 'Node3': 2}, {'Node3': 1, 'Node4': 0}], self.kernels.local_indexes)
		#Testing loading asymmetrical kernels
		self.assertTrue((np.array([[1, 2, 3], [1, 3, 0],[1, 0, 0]]) == self.asym_kernels.kernels_raw[0]).all())
		self.assertTrue((np.array([[0, 4],[5, 7]]) == self.asym_kernels.kernels_raw[1]).all())
		self.assertEqual([{'Node1': 0, 'Node2': 1, 'Node3': 2}, {'Node3': 1, 'Node4': 0}], self.asym_kernels.local_indexes)

	def test_create_general_index(self):
		#Testing creation of general index for symmetric and asymmetric kernels
		self.kernels.create_general_index()
		self.assertEqual({'Node1': [0, None], 'Node2': [1, None], 'Node3': [2, 1], 'Node4': [None, 0]}, self.kernels.kernels_position_index)
		self.asym_kernels.create_general_index()
		self.assertEqual({'Node1': [0, None], 'Node2': [1, None], 'Node3': [2, 1], 'Node4': [None, 0]}, self.asym_kernels.kernels_position_index)

	def test_get_values(self):
		self.kernels.create_general_index()
		self.assertEqual([5], self.kernels.get_values("Node3","Node4"))
		self.assertEqual([1], self.kernels.get_values("Node1","Node2"))
		self.assertEqual([1], self.kernels.get_values("Node1","Node3"))
		self.assertEqual([], self.kernels.get_values("Node2","Node4"))
		self.assertEqual([], self.kernels.get_values("Node1","Node4"))

		self.asym_kernels.create_general_index()
		self.assertEqual([4], self.asym_kernels.get_values("Node4","Node3"))
		self.assertEqual([5], self.asym_kernels.get_values("Node3","Node4"))
		self.assertEqual([3], self.asym_kernels.get_values("Node1","Node3"))
		self.assertEqual([1], self.asym_kernels.get_values("Node3","Node1"))

	def test_integrate_symettric(self):
		self.kernels.create_general_index()
		self.kernels.integrate_matrix("mean")
		self.assertTrue((np.array([[0.5, 0.5, 0.5, 0], [0.5, 1.5, 0, 0], [0.5, 0, 3.5, 2.5], [0, 0, 2.5, 0]]) == self.kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.kernels.integrated_kernel[1])

		self.kernels.integrate_matrix("integration_mean_by_presence")
		self.assertTrue((np.array([[1, 1, 1, 0], [1, 3, 0, 0], [1, 0, 3.5, 5], [0, 0, 5, 0]]) == self.kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.kernels.integrated_kernel[1])

		self.kernels.integrate_matrix("median")
		self.assertTrue((np.array([[1, 1, 1, 0], [1, 3, 0, 0], [1, 0, 3.5, 5], [0, 0, 5, 0]]) == self.kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.kernels.integrated_kernel[1])

		self.kernels.integrate_matrix("max")
		self.assertTrue((np.array([[1, 1, 1, 0], [1, 3, 0, 0], [1, 0, 7, 5], [0, 0, 5, 0]]) == self.kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.kernels.integrated_kernel[1])

		self.kernels.integrate_matrix("geometric_mean")
		#Weird behaviour of the boolean matrix, some values are not equal, probably by numpy numeric representation precision
		#So in this test we are asserting with numpy isclose method instead of ==
		self.assertTrue(np.isclose(np.array([[1, 1, 1, 0], [1, 3, 0, 0], [1, 0, 0, 5], [0, 0, 5, 0]]), self.kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.kernels.integrated_kernel[1])
		
	def test_integrate_asymettric(self):
		#Testing integration mean with asymmetrical matrixes
		self.asym_kernels.create_general_index()
		self.asym_kernels.integrate_matrix("mean", symmetry=False)
		self.assertTrue((np.array([[0.5, 1, 1.5, 0], [0.5, 1.5, 0, 0], [0.5, 0, 3.5, 2.5], [0, 0, 2, 0]]) == self.asym_kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.asym_kernels.integrated_kernel[1])

		self.asym_kernels.integrate_matrix("integration_mean_by_presence", symmetry=False)
		self.assertTrue((np.array([[1, 2, 3, 0], [1, 3, 0, 0], [1, 0, 3.5, 5], [0, 0, 4, 0]]) == self.asym_kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.asym_kernels.integrated_kernel[1])

	def test_integrate_speed(self):
		#Checking that mean integration performs faster with parallelization
		big_kernels = Kernels()
		nodes = [f"M{n}" for n in range(3000)]
		nodes_kernel1 = {node: i for i, node in enumerate(np.random.choice(nodes, 1300, replace=False))}
		nodes_kernel2 = {node: i for i, node in enumerate(np.random.choice(nodes, 400, replace=False))}
		nodes_kernel3 = {node: i for i, node in enumerate(np.random.choice(nodes, 150, replace=False))}
		kernel1 = np.random.randint(0, 10, (len(nodes_kernel1), len(nodes_kernel1)))
		kernel2 = np.random.randint(0, 10, (len(nodes_kernel2), len(nodes_kernel2)))
		kernel3 = np.random.randint(0, 10, (len(nodes_kernel3), len(nodes_kernel3)))
		big_kernels.kernels_raw = [kernel1, kernel2, kernel3]
		big_kernels.local_indexes = [nodes_kernel1, nodes_kernel2, nodes_kernel3] 

		#Testing speed of mean integration with symmetrical matrixes
		big_kernels.create_general_index()
		time1cpu = profiler(big_kernels.integrate_matrix, "mean", 1, True)
		time16cpu  = profiler(big_kernels.integrate_matrix, "mean", 16, True)
		self.assertLessEqual(time16cpu, time1cpu)

	def test_integrate_negatives(self):
		self.negatives_kernels.create_general_index()
		self.negatives_kernels.move2zero_reference()
		self.negatives_kernels.integrate_matrix("mean")
		self.assertTrue((np.array([[0.5, 1.5, 1.5, 0], [1.5, 2.5, 1, 0], [1.5, 1, 6., 4.], [0, 0, 4., 0]]) == self.negatives_kernels.integrated_kernel[0]).all())
		self.assertEqual(['Node1', 'Node2', 'Node3', 'Node4'], self.negatives_kernels.integrated_kernel[1])


"""
	def test_speed_all_combinations(self):
		big_kernels = Kernels()
		nodes = [f"M{n}" for n in range(3000)]
		nodes_kernel1 = {node: i for i, node in enumerate(np.random.choice(nodes, 1300, replace=False))}
		nodes_kernel2 = {node: i for i, node in enumerate(np.random.choice(nodes, 400, replace=False))}
		nodes_kernel3 = {node: i for i, node in enumerate(np.random.choice(nodes, 150, replace=False))}
		kernel1 = np.random.randint(0, 4, (len(nodes_kernel1), len(nodes_kernel1)))
		kernel2 = np.random.randint(0, 4, (len(nodes_kernel2), len(nodes_kernel2)))
		kernel3 = np.random.randint(0, 4, (len(nodes_kernel3), len(nodes_kernel3)))
		big_kernels.kernels_raw = [kernel1, kernel2, kernel3]
		big_kernels.local_indexes = [nodes_kernel1, nodes_kernel2, nodes_kernel3] 
		big_kernels.create_general_index()

		###Testing speed of mean integration with symmetrical matrixes
		big_kernels.create_general_index()
		time1cpu = profiler(big_kernels.integrate_matrix, "mean", 1, True)
		time16cpu  = profiler(big_kernels.integrate_matrix, "mean", 16, True)
		self.assertLessEqual(time16cpu, time1cpu)

		#Testing speed of mean integration with assymmetrical matrixes
		asymtime1cpu = profiler(big_kernels.integrate_matrix, "mean", 1, False)
		asymtime16cpu  = profiler(big_kernels.integrate_matrix, "mean", 16, False)

		print()
		print(f"Time 1 CPU: {time1cpu}")
		print(f"Time 16 CPU: {time16cpu}")
		print(f"Time 1 CPU asym: {asymtime1cpu}")
		print(f"Time 16 CPU asym: {asymtime16cpu}")
		print()

		###Testing speed of geometric mean integration with symmetrical matrixes
		big_kernels.create_general_index()
		time1cpu = profiler(big_kernels.integrate_matrix, "geometric_mean", 1, True)
		time16cpu  = profiler(big_kernels.integrate_matrix, "geometric_mean", 16, True)
		self.assertLessEqual(time16cpu, time1cpu)

		#Testing speed of geometric mean integration with assymmetrical matrixes
		asymtime1cpu = profiler(big_kernels.integrate_matrix, "geometric_mean", 1, False)
		asymtime16cpu  = profiler(big_kernels.integrate_matrix, "geometric_mean", 16, False)

		print()
		print(f"Time 1 CPU: {time1cpu}")
		print(f"Time 16 CPU: {time16cpu}")
		print(f"Time 1 CPU asym: {asymtime1cpu}")
		print(f"Time 16 CPU asym: {asymtime16cpu}")
		print()
"""