#! /usr/bin/env python

import argparse
import sys
import os
ROOT_PATH=os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_PATH, '..'))
from NetAnalyzer import Net_parser, NetAnalyzer

##############################
#OPTPARSE
##############################
def layer_parse(string): return [sublst.split(",") for sublst in string.split(";")]

parser = argparse.ArgumentParser(description='Perform Network analysis from NetAnalyzer package')

parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
          help="Input file to create networks for further analysis")
parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
          help="Output file to save random network")
parser.add_argument("-n", "--node_names_file", dest="node_file", default= None, 
          help="File with node names corresponding to the input matrix, only use when -f is set to bin or matrix.")
parser.add_argument("-f", "--input_format", dest="input_format", default= 'pair', 
          help="Input file format: pair (default), bin, matrix")
parser.add_argument("-s", "--split_char", dest="split_char", default= "\t", 
          help="Character for splitting input file. Default: tab")
parser.add_argument("-l","--layers", dest="layers", default=['layer', '-'], type= layer_parse,
          help="Layer definition on network: layer1name,regexp1;layer2name,regexp2...")
parser.add_argument("-r", "--type_random", dest="type_random", default= None, 
          help="Randomized basis. 'nodes' Node-baseds randomize or 'links' Links-baseds randomize")

options = parser.parse_args()

##########################
#MAIN
##########################

fullNet = Net_parser.load(vars(options)) 
fullNet.randomize_network(options.type_random)

with open(options.output_file, "w") as outfile:
  for e in fullNet.graph.edges:
    outfile.write(f"{e[0]}\t{e[1]}\n")