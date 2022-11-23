import re
import networkx as nx
# https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
# http://mkivela.com/pymnet
class NetAnalyzer:

	def __init__(self, layers):
		self.graph = nx.Graph()
		self.layers = []

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

	def get_nodes_layer(self, layers):
		nodes = []
		for layer in layers:
			nodes.extend([nodeID for nodeID, attr in self.graph.nodes(data=True) if attr['layer']== layer])
		return nodes

	def get_edge_number(self):
		return len(self.graph.edges())

	def try_test(self):
		return 2




# def get_counts_association(layers, base_layer)
# 	relations = get_associations(layers, base_layer) do |associatedIDs_node1, associatedIDs_node2, intersectedIDs, node1, node2|
# 		countValue = intersectedIDs.length	
# 	end
# 	#@association_values[:counts] = relations
# 	return relations
