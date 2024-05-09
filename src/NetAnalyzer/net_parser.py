import numpy
from NetAnalyzer.netanalyzer import NetAnalyzer

class Net_parser:

	def load(options):
		net = None
		if options['input_format'] == 'pair':
		  net = Net_parser.load_network_by_pairs(options['input_file'], options['layers'], options['split_char'])
		elif options['input_format'] == 'bin':
		  net = Net_parser.load_network_by_bin_matrix(options['input_file'], options['node_files'], options['layers'])
		elif options['input_format'] == 'matrix':
		  net = Net_parser.load_network_by_plain_matrix(options['input_file'], options['node_files'], options['layers'], options['split_char'])
		else:
		  raise("ERROR: The format " + options['input_format'] + " is not defined")

		if options.get('load_both'): # TODO: Not tested Yet.
			if not net.graph:
				layerA, layerB = list(net.matrices["adjacency_matrices"].keys())[0]
				net.adjMat2netObj(layerA,layerB)				
			if net.matrices["adjacency_matrices"] == {}:
				net.generate_all_biadjs()

		return net

	def load_network_by_pairs(file, layers, split_character="\t"):
		net = NetAnalyzer([layer[0] for layer in layers])
		f = open(file)
		for line in f:
			pair = line.rstrip().split(split_character)
			node1 = pair[0]
			node2 = pair[1]
			net.add_node(node1, net.set_layer(layers, node1))
			net.add_node(node2, net.set_layer(layers, node2))
			if len(pair) == 3:
				net.add_edge(node1, node2, weight=float(pair[2]))
			else:
				net.add_edge(node1, node2)

			net.add_edge(node1, node2)	
		f.close()
		return net

	def load_network_by_bin_matrix(input_file, node_file, layers):
		tag_layers = tuple([layer[0] for layer in layers])
		net = NetAnalyzer(tag_layers)
		if len(node_file) == 1:
			node_names = Net_parser.load_input_list(node_file[0])
			row_names = col_names = node_names
		else:
			row_names = Net_parser.load_input_list(node_file[0])
			col_names = Net_parser.load_input_list(node_file[1])
		if len(tag_layers) == 1:
			net.matrices["adjacency_matrices"][(tag_layers[0],tag_layers[0])] = [numpy.load(input_file), row_names, col_names]
		else:
			net.matrices["adjacency_matrices"][tag_layers] = [numpy.load(input_file), row_names, col_names]
		return net

	def load_network_by_plain_matrix(input_file, node_file, layers, splitChar="\t"):
		tag_layers = tuple([layer[0] for layer in layers])
		net = NetAnalyzer(tag_layers)
		if len(node_file) == 1:
			node_names = Net_parser.load_input_list(node_file[0])
			row_names = col_names = node_names
		else:
			row_names = Net_parser.load_input_list(node_file[0])
			col_names = Net_parser.load_input_list(node_file[1])
		if len(tag_layers) == 1:
			net.matrices["adjacency_matrices"][(tag_layers[0],tag_layers[0])] = [numpy.genfromtxt(input_file, delimiter=splitChar), row_names, col_names]
		else:
			net.matrices["adjacency_matrices"][tag_layers] = [numpy.genfromtxt(input_file, delimiter=splitChar), row_names, col_names]
		return net

	def load_input_list(input_path):
		file = open(input_path, "r")
		input_data = file.readlines()
		file.close()
		return [line.rstrip() for line in input_data]
