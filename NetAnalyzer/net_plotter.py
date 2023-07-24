import os
import sys
import graphviz
import json
import base64
import igraph as ig
from igraph.layout import Layout
import matplotlib as mpl
import random
import numpy as np
import pickle
from py_report_html import Py_report_html
class Net_plotter:

    TEMPLATES = os.path.join(os.path.dirname(__file__), 'templates')

    def __init__(self, net_data, options={}):

        self.group_nodes = net_data['group_nodes']
        self.reference_nodes = net_data['reference_nodes']
        self.graph = net_data['graph']
        self.layers = net_data['layers']

        if options['method'] == 'graphviz':
            self.plot_dot(options)
        if options['method'] == 'igraph':
            self.plot_igraph(options)            
        elif options['method'] == 'cyt_app':
            self.plot_cyt_app(options)
        else:
            container = {'net_data' : {'group_nodes' : self.group_nodes, 'reference_nodes' : self.reference_nodes, 'graph' : self.graph, 'layers' : self.layers}}
            template = open(os.path.join(Net_plotter.TEMPLATES, 'network.txt')).read()
            report = Py_report_html(container, os.path.basename(options['output_file']), True, True)
            report.build(template, build_options=options)
            report.write(options['output_file'] + '.html')

    def get_node_layer(self, node_id):
        return self.graph.nodes(data=True)[node_id]['layer']

    ## GRAPHVIZ
    ##############################################################
    def plot_dot(self, user_options = {}): # input keys: layout
        # Watch out: Node ids must be with no ":".
        options = {'layout': "sfdp"}
        options.update(user_options)
        graphviz_colors = ['lightsteelblue1', 'lightyellow1', 'lightgray', 'orchid2']
        palette = {}
        for layer in self.layers: palette[layer] = graphviz_colors.pop(0)
        graph = graphviz.Graph('graph')
        graph.attr(overlap = 'false', outputorder='edgesfirst')
        for e in self.graph.edges:
            l0 = self.get_node_layer(e[0])
            graph.node(f'"{e[0]}"', '', style = 'filled', fillcolor = palette[l0])
            l1 = self.get_node_layer(e[1])
            graph.node(f'"{e[1]}"', '', style = 'filled', fillcolor = palette[l1])
            graph.edge(f'"{e[0]}"', f'"{e[1]}"')

        for nodeID in self.reference_nodes: graph.node(f'"{nodeID}"', '', style = 'filled', fillcolor = 'firebrick1')
        graphviz_border_colors = ['blue', 'darkorange', 'red', 'olivedrab4']
        for groupID, gNodes in self.group_nodes.items():
            border_color = graphviz_border_colors.pop(0)
            for nodeID in gNodes: graph.node(f'"{nodeID}"', '', color = border_color, penwidth = '10')
        graph.render(outfile= options['output_file'] + '.png', format='png', engine = options['layout'])

    ## IGRAPH
    ##########################################################################
    def plot_igraph(self, user_options = {}):
        random.seed(1234)
        ig_net = ig.Graph.from_networkx(self.graph)
        net_edge_weight = ig_net.es['weight']
        newMax= np.percentile(net_edge_weight, 90)
        #net_edge_weight = [ f"rgba(0.5,0.5,0.5,{round(w/newMax, 3)})" for w in net_edge_weight ]
        norm_net_edge_weight = [ ]
        for w in net_edge_weight:
            normalized = round(w/newMax, 3)
            if normalized > 1: normalized = 1
            norm_net_edge_weight.append(f"rgba(0.7,0.7,0.7,{normalized})")
        cmap=mpl.colormaps['Pastel1']
        node_ids = ig_net.vs['_nx_name']
        node_base_color = list(cmap(0))
        node_base_color[3] = 0.25
        node_base_color = tuple(node_base_color)
        node_colors = [node_base_color] * len(node_ids)
        # Node color
        count = 1
        for groupID, gNodes in self.group_nodes.items():
            color = cmap(count)
            for n in gNodes:
                idx = node_ids.index(n)
                node_colors[idx] = color
            count += 1

        # Node order: tag each node with a int that says in which order mut be plotted. 0 is the first node to be plotted and N node the last (so the first in the image)
        node_order=[0] * len(node_ids)
        node_count = len(node_ids) -1
        node_dict = {}
        for groupID, gNodes in self.group_nodes.items():
            for n in gNodes:
                node_dict[n] = node_count
                node_count -= 1
        for i,n_id in enumerate(node_ids):
            order = node_dict.get(n_id)
            if order == None: 
                order = node_count
                node_count -= 1
            node_order[i] = order
        opts = {
            'bbox' : (2400, 2400),
            'vertex_size' : 7,
            'layout' : "drl",
            'edge_color' : norm_net_edge_weight,
            'vertex_color': node_colors,
            'vertex_order': node_order
        }
        opts.update(user_options)
        layout = self.get_igraph_layout(node_ids, ig_net, opts.pop('layout'), opts)
        ig.plot(ig_net, layout=layout, target=user_options['output_file'] + '.png', **opts)

    def get_igraph_layout(self, node_ids, igraph_obj, layout_name, opts):
        layout = None
        load_path = opts.get('load')
        if load_path != None:
            sorted_coords = []
            with open(load_path, 'rb') as file: 
                tagged_coordinates = pickle.load(file)
                for nodeID in node_ids: # Reorder file loaded layout with the current node ordering in the igraph object
                    coords = tagged_coordinates.get(nodeID)
                    if coords != None: sorted_coords.append(coords)
            layout = Layout(sorted_coords) # Create layout objet from reordered loaded node coords
        else:
            layout_custom_opts = {}
            custom_opts_string = opts.get('custom_opts')
            if layout_custom_opts != None: layout_custom_opts = eval(custom_opts_string)
            layout = igraph_obj.layout(layout_name, **layout_custom_opts)
        save_path = opts.get('save')
        if save_path != None:
            tagged_coordinates = {}
            for i, n in enumerate(node_ids):
                tagged_coordinates[n] = layout[i]
            with open(save_path, 'wb') as file: pickle.dump(tagged_coordinates, file)
        return layout

    ## CYTOSCAPE APP
    ###########################################################################
    def plot_cyt_app(self, user_options = {}):
        options = {}
        options.update(user_options)

        group_nodes = {}
        for groupID, gNodes in self.group_nodes.items():
            for gNode in gNodes:
                group_nodes[gNode] = groupID

        node_cyt_ids = {}
        nodes = []
        count = 0
        for node in self.graph.nodes: 
            self.cyt_app_add_node(nodes, count, node, group_nodes)
            node_cyt_ids[node] = str(count)
            count += 1
        edges = self.cyt_app_add_edges(node_cyt_ids, count)
        cys_net = {
            'elements' : {
                'nodes' : nodes,
                'edges' : edges
            }
        }
        with open(options['output_file'] + '.cyjs', 'w') as f: f.write(json.dumps(cys_net, indent=4))

    def cyt_app_add_node(self, nodes, count, node, group_nodes):
        cyt_node = {
            'data' : {
                'id' : str(count),
                'name' : node
            }
        }
        cyt_node['data']['type'] = self.get_node_layer(node)
        if len(self.reference_nodes) > 0:
            cyt_node['data']['ref'] = 'y' if node in self.reference_nodes else 'n'
        if len(group_nodes) > 0:
            query = group_nodes[node]
            if query != None: cyt_node['data']['group'] = query
        nodes.append(cyt_node)

    def cyt_app_add_edges(self, node_ids, count):
        edges = []
        for e in self.graph.edges:
            edges.append({
                'data' : {
                    'id' : str(count),
                    'source' : node_ids[e[0]],
                    'target' : node_ids[e[1]],
                    "interaction" : "-",
                    "weight" : 1.0
                }
            })
            count +=1
        return edges
    ###########################################################################
