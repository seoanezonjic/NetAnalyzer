import graphviz

class Net_plotter:

    # TEMPLATES = File.join(File.dirname(__FILE__), 'templates')

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
            if options['method'] == 'elgrapho':
                template = 'el_grapho'
            elif options['method'] == 'cytoscape':
                template = 'cytoscape'
            elif options['method'] == 'sigma':
                template = 'sigma'
            #renderered_template = ERB.new(File.open(File.join(TEMPLATES, template + '.erb')).read).result(binding)
            #File.open(options[output_file] + '.html', 'w'){|f| f.puts renderered_template}

    def plot_dot(self, user_options = {}): # input keys: layout
        options = {'layout': "sfdp"}
        options.update(user_options)
        graphviz_colors = ['lightsteelblue1', 'lightyellow1', 'lightgray', 'orchid2']
        palette = {}
        for layer in self.layers: palette[layer] = graphviz_colors.pop(0)
        graph = graphviz.Graph('graph')
        graph.attr = {'overlap' : False}
        for e in self.graph.edges:
            l0 = self.graph.nodes(data=True)[e[0]]['layer'] # get node layer
            graph.node(e[0], '', style = 'filled', fillcolor = palette[l0])
            l1 = self.graph.nodes(data=True)[e[1]]['layer'] # get node layer
            graph.node(e[1], '', style = 'filled', fillcolor = palette[l1])
            graph.edge(e[0], e[1])

        for nodeID in self.reference_nodes: graph.node(nodeID, '', style = 'filled', fillcolor = 'firebrick1')
        graphviz_border_colors = ['blue', 'darkorange', 'red', 'olivedrab4']
        for groupID, gNodes in self.group_nodes.items():
            border_color = graphviz_border_colors.pop(0)
            for nodeID in gNodes: graph.node(nodeID, '', color = border_color, penwidth = '10')
        graph.render(outfile= options['output_file'] + '.png', format='png', engine = options['layout'])
