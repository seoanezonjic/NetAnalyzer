import os
    
class SeedParser:

    @staticmethod
    def load_nodes_by_group(node_groups, sep= ','):
        group_nodes = {}
        weights_nodes = {}
        if os.path.exists(node_groups):
            group_nodes, weights_nodes = SeedParser.load_node_groups_from_file(node_groups, sep=sep)
        else:
            group_nodes = {"seed_genes": node_groups.split(sep)}
        return group_nodes, weights_nodes

    @staticmethod
    def load_node_groups_from_file(file, sep=','):
        group_nodes = {}
        weights_nodes = {}
        with open(file) as f:
            for line in f:
                fields = line.rstrip().split("\t")
                set_name, nodes, *weights = fields
                group_nodes[set_name] = nodes.split(sep)
                if weights:
                    weights = weights[0].split(sep)
                    weights_nodes[set_name] = {group_nodes[set_name][i]: float(weight) for i, weight in enumerate(weights)} # TODO: check this section
        return group_nodes, weights_nodes
