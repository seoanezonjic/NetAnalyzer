import sys 
import re
import networkx as nx
import math
import numpy
import scipy.stats as stats
import statsmodels

# https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
# http://mkivela.com/pymnet
class NetAnalyzer:

	def __init__(self, layers):
		self.graph = nx.Graph()
		self.layers = []
		self.association_values = {}
		self.compute_autorelations = True
		self.compute_pairs = 'conn'
		self.adjacency_matrices = {}
		self.kernels = {}

	def add_node(self, nodeID, layer):
		self.graph.add_node(nodeID, layer=layer)

	def add_edge(self, node1, node2):
		self.graph.add_edge(node1, node2)

	def set_layer(self, layer_definitions, node_name):
		layer = None
		if len(layer_definitions) > 1:
			for layer_name, regexp in layer_definitions:
				if re.search(regexp, node_name):
					layer = layer_name
					break
			if layer == None: raise Exception("The node '" + node_name + "' not match with any layer regex")
		else:
			layer = layer_definitions[0][0]
		if layer not in self.layers: self.layers.append(layer)
		return layer

	def get_nodes_by_attr(self, attrib, value):
		return [nodeID for nodeID, attr in self.graph.nodes(data=True) if attr[attrib] == value]

	def get_nodes_layer(self, layers):
		nodes = []
		for layer in layers:
			nodes.extend(self.get_nodes_by_attr('layer', layer))
		return nodes

	def get_edge_number(self):
		return len(self.graph.edges())

	def collect_nodes(self, layers = 'all'):
		nodeIDsA = None
		nodeIDsB = None
		if self.compute_autorelations:
			if layers == 'all':
				nodeIDsA = self.graph.nodes
			else:
				nodeIDsA = self.get_nodes_layer(layers)
		else:
			if layers != 'all': # layers contains two layer IDs
				nodeIDsA = self.get_nodes_layer(layers[0])
				nodeIDsB = self.get_nodes_layer(layers[1])
		return nodeIDsA, nodeIDsB

	def connections(self, ids_connected_to_n1, ids_connected_to_n2):
		res = False
		if ids_connected_to_n1 != None and ids_connected_to_n2 != None and len(ids_connected_to_n1 & ids_connected_to_n2) > 0 : # check that at least exists one node that connect to n1 and n2
			res = True
		return res
	
	def get_all_pairs(self, pair_operation, layers = 'all'):
		all_pairs = []
		nodeIDsA, nodeIDsB = self.collect_nodes(layers = layers)
		if self.compute_autorelations:
			while len(nodeIDsA) > 0:
				node1 = nodeIDsA.pop(0)
				if self.compute_pairs == 'all':
					for node2 in nodeIDsA:
						res = pair_operation(node1, node2)
						all_pairs.append(res)
				elif self.compute_pairs == 'conn':
					ids_connected_to_n1 = set(self.graph.neighbors(node1))
					for node2 in nodeIDsA:
						ids_connected_to_n2 = set(self.graph.neighbors(node2))
						if self.connections(ids_connected_to_n1, ids_connected_to_n2):
							res = pair_operation(node1, node2)
							all_pairs.append(res)
		else:
			if self.compute_pairs == 'conn': #MAIN METHOD
				for node1 in nodeIDsA:
					ids_connected_to_n1 = set(self.graph.neighbors(node1))
					for node2 in nodeIDsB:
						ids_connected_to_n2 = set(self.graph.neighbors(node2))
						if self.connections(ids_connected_to_n1, ids_connected_to_n2):
							res = pair_operation(node1, node2)
							all_pairs.append(res)
			elif self.compute_pairs == 'all':
				raise Exception('Not implemented')

		return all_pairs

	def get_associations(self, layers, base_layer, compute_association): # BASE METHOD
		base_nodes = set(self.get_nodes_layer([base_layer]))
		def _(node1, node2): 
			associatedIDs_node1 = set(self.graph.neighbors(node1))
			associatedIDs_node2 = set(self.graph.neighbors(node2))
			intersectedIDs = (associatedIDs_node1 & associatedIDs_node2) & base_nodes
			associationValue = compute_association(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2)
			return [node1, node2, associationValue]  
		associations = self.get_all_pairs(_, layers = layers)
		return associations

	#https://stackoverflow.com/questions/55063978/ruby-like-yield-in-python-3
	def get_counts_associations(self, layers, base_layer):
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			return len(intersectedIDs)
		relations = self.get_associations(layers, base_layer, _)
		self.association_values['counts'] = relations
		return relations

	def get_jaccard_associations(self, layers, base_layer):
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			unionIDS = associatedIDs_node1 | associatedIDs_node2
			return len(intersectedIDs)/len(unionIDS)		
		relations = self.get_associations(layers, base_layer, _)
		self.association_values['jaccard'] = relations
		return relations

	def get_simpson_associations(self, layers, base_layer):
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			minLength = min([len(associatedIDs_node1), len(associatedIDs_node2)])
			return len(intersectedIDs)/minLength
		relations = self.get_associations(layers, base_layer, _)
		self.association_values['simpson'] = relations
		return relations

	def get_geometric_associations(self, layers, base_layer):
		#wang 2016 method
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			intersectedIDs = len(intersectedIDs)**2
			productLength = math.sqrt(len(associatedIDs_node1) * len(associatedIDs_node2))
			return intersectedIDs/productLength
		relations = self.get_associations(layers, base_layer, _)
		self.association_values['geometric'] = relations
		return relations

	def get_cosine_associations(self, layers, base_layer):
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			productLength = math.sqrt(len(associatedIDs_node1) * len(associatedIDs_node2))
			return len(intersectedIDs)/productLength
		relations = self.get_associations(layers, base_layer, _)
		self.association_values['cosine'] = relations
		return relations

	def get_pcc_associations(self, layers, base_layer):
		#for Ny calcule use get_nodes_layer
		base_layer_nodes = self.get_nodes_layer([base_layer])
		ny = len(base_layer_nodes)
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			intersProd = len(intersectedIDs) * ny
			nodesProd = len(associatedIDs_node1) * len(associatedIDs_node2)
			nodesSubs = intersProd - nodesProd
			nodesAInNetwork = ny - len(associatedIDs_node1)
			nodesBInNetwork = ny - len(associatedIDs_node2)
			return numpy.float64(nodesSubs) / math.sqrt(nodesProd * nodesAInNetwork * nodesBInNetwork) # TODO: numpy.float64 is used to handle division by 0. Fix the implementation/test to avoid this case
		relations = self.get_associations(layers, base_layer, _)
		self.association_values['pcc'] = relations
		#print(relations)
		return relations

	def get_csi_associations(self, layers, base_layer):
		pcc_relations = self.get_pcc_associations(layers, base_layer)
		pcc_relations = [row for row in pcc_relations if not math.isnan(row[2])]
		if len(layers) > 1:
			self.clean_autorelations_on_association_values()

		nx = len(self.get_nodes_layer(layers))
		pcc_vals = {}
		node_rels = {}

		for node1, node2, assoc_index in pcc_relations:
			self.add_nested_record(pcc_vals, node1, node2, numpy.abs(assoc_index))
			self.add_nested_record(pcc_vals, node2, node1, numpy.abs(assoc_index))
			self.add_record(node_rels, node1, node2)
			self.add_record(node_rels, node2, node1)
		
		relations = []
		for node1, node2, assoc_index in pcc_relations:
			pccAB = assoc_index - 0.05
			valid_nodes = 0

			significant_nodes_from_node1 = set([node for node in node_rels[node1] if pcc_vals[node1][node] >= pccAB])
			significant_nodes_from_node2 = set([node for node in node_rels[node2] if pcc_vals[node2][node] >= pccAB])
			all_significant_nodes = significant_nodes_from_node2 | significant_nodes_from_node1
			all_nodes = set(node_rels[node1]) | set(node_rels[node2])
			
			csiValue = 1 - (len(all_significant_nodes))/(len(all_nodes)) 		
			relations.append([node1, node2, csiValue])
		
		self.association_values['csi'] = relations
		return relations

	def get_hypergeometric_associations(self, layers, base_layer, pvalue_adj_method= None):
		ny = len(self.get_nodes_layer([base_layer]))
		def _(associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2):
			intersection_lengths = len(intersectedIDs)
			if intersection_lengths > 0:
				n1_items = len(associatedIDs_node1)
				n2_items = len(associatedIDs_node2)
				M= ny
				n = n1_items
				N = n2_items
				p_value = stats.hypergeom.sf(intersection_lengths-1, M, n, N)

			return p_value
		relations = self.get_associations(layers, base_layer, _)

		if pvalue_adj_method == 'bonferroni':
			meth = 'hypergeometric_bf'
			self.adjust_pval_association(relations, 'bonferroni')
		elif pvalue_adj_method == 'benjamini_hochberg':
			meth = 'hypergeometric_bh'
			self.adjust_pval_association(relations, 'fdr_bh')
		else:
			meth = 'hypergeometric'
		relations = [[assoc[0], assoc[1], -numpy.log10(assoc[2])] for assoc in relations]
		self.association_values[meth] = relations
		return relations

	def adjust_pval_association(associations, method): # TODO TEST
		pvals = numpy.array([val[2] for val in relations])
		adj_pvals = statsmodels.stats.multitest.multipletests(pvals, method=method, is_sorted=False, returnsorted=False)
		count = 0
		for adj_pval in adj_pvals:
			relations[count][2] = adj_pval
			count +=1

	def get_kernel(layer2kernel, kernel, normalization=False):
		matrix, node_names = self.adjacency_matrices[layer2kernel]
		matrix_result = Adv_mat_calc.get_kernel(matrix, node_names, kernel, normalization=normalization)
		self.kernels[layer2kernel] = matrix_result

	def write_kernel(layer2kernel, output_file):
		numpy.save(output_file, self.kernels[layer2kernel])

	## AUXILIAR METHODS
	#######################################################################################

	def add_record(self, hash, node1, node2):
	    query = hash.get(node1)
	    if query is None:
	        hash[node1] = [node2]
	    else:
	        query.append(node2)

	def add_nested_record(self, hash, node1, node2, val):
	    query_node1 = hash.get(node1)
	    if query_node1 is None:
	        hash[node1] = {node2: val}
	    else:
	        query_node1[node2] = val
