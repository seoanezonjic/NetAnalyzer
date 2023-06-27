import os
import sys
import graphviz
import json
import base64
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

    def plot_dot(self, user_options = {}): # input keys: layout
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

        for nodeID in self.reference_nodes: graph.node(nodeID, '', style = 'filled', fillcolor = 'firebrick1')
        graphviz_border_colors = ['blue', 'darkorange', 'red', 'olivedrab4']
        for groupID, gNodes in self.group_nodes.items():
            border_color = graphviz_border_colors.pop(0)
            for nodeID in gNodes: graph.node(nodeID, '', color = border_color, penwidth = '10')
        graph.render(outfile= options['output_file'] + '.png', format='png', engine = options['layout'])

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
