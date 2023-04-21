#! /usr/bin/env python

import argparse
import sys
import os
import glob
import numpy as np

ROOT_PATH=os.path.dirname(__file__)
sys.path.insert(0,os.path.join(ROOT_PATH, '..'))
from NetAnalyzer import Ranker

########################### METHODS ########################
############################################################

def write_ranking(file, ranking_list):
  with open(file, 'w') as f:
    for seed_name, ranking in ranking_list.items():
      for ranked_gene in ranking:
        f.write('\t'.join(map(str,ranked_gene)) + "\t" + f"{seed_name}" + "\n")     


########################### OPTPARSE ########################
#############################################################

def threads_based_0(trheads): return int(threads) - 1

parser = argparse.ArgumentParser(description='Get the ranks from a matrix similarity score and a list of seeds')
parser.add_argument("-k", "--input_kernels", dest="kernel_file", default=None, 
  help="The roots from each kernel to integrate")
parser.add_argument("-n", "--input_nodes", dest="input_nodes", default=None,
  help="The list of node for each kernel in lst format")
parser.add_argument("-s", "--genes_seed", dest="genes_seed", default=None,
  help="The name of the gene to look for backups")
parser.add_argument("-S", "--seed_sep", dest="seed_sep", default=",",
  help="Separator of seed genes. Only use when -s point to a file")
parser.add_argument("-f", "--filter", dest="filter", default=None,
  help="PATH to file with seed_name and genes to keep in output")
parser.add_argument("-l", "--leave_one_out", dest="leave_one_out", default=False, 
  action='store_true', help="PATH to file with seed_name and genes to keep in output")
parser.add_argument("-t", "--top_n", dest="top_n", default=None, type=int,
  help="Top N genes to print in output")
parser.add_argument("--output_top", dest="output_top", default=None,
  help="File to save Top N genes")
parser.add_argument("-o", "--output_name", dest="output_name", default="ranked_genes",
 help="PATH to file with seed_name and genes to keep in output")
parser.add_argument("--type_of_candidates", dest="type_of_candidates", default=False, action="store_true",
 help="type of candidates to output in ranking list: all, new, seed.")

# TODO: Add Threat section
#parser.add_argument("-T", "--threads", dest="threads", default=0, type=threads_based_0,
# help="Number of threads to use in computation, one thread will be reserved as manager.")
options = parser.parse_args()


########################### MAIN ############################
#############################################################

ranker = Ranker()
ranker.matrix = np.load(options.kernel_file)
ranker.load_nodes_from_file(options.input_nodes)
ranker.load_seeds(options.genes_seed, sep= options.seed_sep)
options.filter is not None and ranker.load_references(options.filter, sep= ",") 
ranker.do_ranking(leave_one_out= options.leave_one_out)
rankings = ranker.ranking

discarded_seeds = [seed_name for seed_name, ranks in rankings.items() if not ranks]

if discarded_seeds:
  with open(options.output_name + "_discarded","w") as f:
    for seed_name in discarded_seeds:
      f.write(f"{seed_name}\t{options.seed_sep.join(ranker.seeds[seed_name])}")

if options.top_n is not None:
  top_n = ranker.get_top(options.top_n)
  if options.output_top is None:
    rankings = top_n
  else:
    write_ranking(options.output_top, top_n)

if options.filter is not None:
  rankings = ranker.get_reference_ranks()

if rankings:
  if options.type_of_candidates:
    for seed_name, rankings_by_seed in rankings.items():
      added_ranking_column = []
      for ranking in rankings_by_seed:
        if ranking[0] in ranker.seeds[seed_name]:
          ranking.insert(5, "seed")
        else:
          ranking.insert(5, "new")
        added_ranking_column.append(ranking)
      rankings[seed_name] = added_ranking_column
  
  write_ranking(f"{options.output_name}_all_candidates", rankings)