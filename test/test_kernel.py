import unittest
import os
from NetAnalyzer import NetAnalyzer
from NetAnalyzer import Net_parser
import numpy as np
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data/data_kernel')

class KernelTestCase(unittest.TestCase):
	def setUp(self):
		self.monopartite_layers = ('main', 'main')
		self.monopartite_network_node_names = os.path.join(DATA_TEST_PATH, 'adj_mat.lst')
		self.monopartite_network_file = os.path.join(DATA_TEST_PATH, 'adj_mat.npy')
		options = {'input_format' : 'bin', 
				   'input_file' : self.monopartite_network_file, 
				   'layers' : [['main', 'M[0-9]+'], ['main', 'M[0-9]+']], 
				   'node_file' : self.monopartite_network_node_names,
				   'load_both': True}
		self.monopartite_network = Net_parser.load(options)
		self.ct_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'ct.npy'))
		self.el_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'el.npy'))
		self.md1_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'md1.npy'))
		self.me_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'me.npy'))
		self.rf_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'rf.npy'))
		self.rl0_5_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'rl0_5.npy'))
		self.vn0_5_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'vn0_5.npy'))
		self.ka_norm_kernel_to_test = np.load(os.path.join(DATA_TEST_PATH, 'ka_normalized.npy'))

		

	def test_get_kernel_ct(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "ct" , add_to_object= True)
		ct_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["ct"]
		self.assertTrue((ct_kernel[0].round(decimals=10) == self.ct_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == ct_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == ct_kernel[2])

	def test_get_kernel_el(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "el", add_to_object= True)
		el_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["el"]
		self.assertTrue((el_kernel[0].round(decimals=10) == self.el_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == el_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == el_kernel[2])

	def test_get_kernel_md1(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "md1", add_to_object= True)
		md1_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["md1"]
		self.assertTrue((md1_kernel[0].round(decimals=10) == self.md1_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == md1_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == md1_kernel[2])

	
	def test_get_kernel_me(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "me", add_to_object= True)
		me_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["me"]
		self.assertTrue((me_kernel[0].round(decimals=10) == self.me_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == me_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == me_kernel[2])
	
	def test_get_kernel_rf(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "rf", add_to_object= True)
		rf_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["rf"]
		self.assertTrue((rf_kernel[0].round(decimals=10) == self.rf_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == rf_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == rf_kernel[2])
	
	def test_get_kernel_rl0_5(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "rl0.5", add_to_object= True)
		rl0_5_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["rl0.5"]
		self.assertTrue((rl0_5_kernel[0].round(decimals=10) == self.rl0_5_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == rl0_5_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == rl0_5_kernel[2])

	
	
	def test_get_kernel_vn0_5(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "vn0.5", add_to_object= True)
		vn0_5_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["vn0.5"]
		self.assertTrue((vn0_5_kernel[0].round(decimals=10) == self.vn0_5_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == vn0_5_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == vn0_5_kernel[2])

	
	
	def test_get_kernel_ka_norm(self):
		self.monopartite_network.get_kernel(layers2kernel = self.monopartite_layers, method = "ka", normalization = True, add_to_object= True)
		ka_norm_kernel = self.monopartite_network.matrices["kernels"][self.monopartite_layers]["ka"]
		self.assertTrue((ka_norm_kernel[0].round(decimals=10) == self.ka_norm_kernel_to_test.round(decimals=10)).all())
		self.assertTrue(list(self.monopartite_network.graph.nodes) == ka_norm_kernel[1])
		self.assertTrue(list(self.monopartite_network.graph.nodes) == ka_norm_kernel[2])
	# def test_get_kernel_ka_norm(self)
	# 	ka_norm_kernel = self.monopartite_network.get_kernel('main', "ka", normalization = True)
	# 	self.assertEqual(ct_kernel, self.ka_norm_kernel_to_test)
