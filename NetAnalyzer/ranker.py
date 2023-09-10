import numpy as np
import os
from itertools import combinations
from sklearn.model_selection import KFold, LeaveOneOut


class Ranker:

  def __init__(self):
    self.matrix = None
    self.nodes = []  # kernel_nodes
    self.seeds = {}  # genes_seed
    self.reference_nodes = {}
    self.ranking = {}  # ranked_genes


  def normalize_matrix(self, mode = "by_column"):
    degree_matrix = self.matrix.sum(0)
    inv_degree_matrix = np.diag(1/degree_matrix)
    if mode == "by_column":
      self.matrix = self.matrix @ inv_degree_matrix 
    elif mode == "by_row":
      self.matrix = inv_degree_matrix  @ self.matrix
    elif mode == "by_row_col":
      self.matrix = np.sqrt(inv_degree_matrix) @ self.matrix @ np.sqrt(inv_degree_matrix)

  def load_seeds(self, node_groups, sep=',', uniq= True):
    self.seeds = self.load_nodes_by_group(node_groups, sep=sep)
    if uniq:
      self.seeds = {key: sorted(list(set(seed))) for key, seed in self.seeds.items()}

  def load_references(self, node_groups, sep=','):
    self.reference_nodes = self.load_nodes_by_group(node_groups, sep=sep)

  def load_nodes_by_group(self, node_groups, sep: ','):
    if os.path.exists(node_groups):
      group_nodes = self.load_node_groups_from_file(node_groups, sep=sep)
    else:
      group_nodes = {"seed_genes": node_groups.split(sep)}
    return group_nodes

  def load_node_groups_from_file(self, file, sep=','):
    group_nodes = {}
    with open(file) as f:
      for line in f:
        set_name, nodes = line.rstrip().split("\t")
        group_nodes[set_name] = nodes.split(sep)
    return group_nodes

  def load_nodes_from_file(self, file):
    with open(file) as f:
      for line in f:
        self.nodes.append(line.rstrip())

  def get_seed_cross_validation(self, k_fold = None):
    new_seeds = {}
    genes2predict = {}

    for seed_name, seed in self.seeds.items():
      if k_fold != None:
        cv = KFold(n_splits=k_fold)
      else:
        cv = LeaveOneOut()

      for indx, (train_index, test_index) in enumerate(cv.split(seed)):
          seed_name_one_out = str(seed_name) + "_iteration_" + str(indx)
          new_seeds[seed_name_one_out] = [ seed[i] for i in train_index ]
          genes2predict[seed_name_one_out] = [ seed[i] for i in test_index ]
          if self.reference_nodes.get(seed_name) is not None:
            genes2predict[seed_name_one_out] += self.reference_nodes[seed_name]

          genes2predict[seed_name_one_out] = list(
            set(genes2predict[seed_name_one_out]))

    self.seeds = new_seeds
    self.reference_nodes = genes2predict


  def do_ranking(self, cross_validation = False, k_fold = None, propagate = False, options = {"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}): #TODO: Add thread option
    if cross_validation:
      self.get_seed_cross_validation(k_fold=k_fold)

    seed_indexes = self.get_seed_indexes()
    #seed_groups = list(self.seeds)  # Array conversion needed for parallelization
    ranked_lists = []
    # TODO: Implement parallelization in the process as ruby if needed.
    for seed_name, seed in self.seeds.items():
      # The code in this block CANNOT modify nothing outside
      rank_list = self.rank_by_seed(seed_indexes, seed, propagate=propagate, options=options)  # Production mode
      if cross_validation:
        rank_list = self.delete_seed_from_rank(rank_list, seed)
      # if cross_validation and len(self.reference_nodes[seed_name]) == 1:
      #    rank_list = self.get_individual_rank(seed, self.reference_nodes[seed_name][0], propagate = propagate, options= options)
      # else:
      #    rank_list = self.rank_by_seed(seed_indexes, seed, propagate=propagate, options=options)  # Production mode
      ranked_lists.append([seed_name, rank_list])

    for seed_name, rank_list in ranked_lists:  # Transfer resuls to hash
      self.ranking[seed_name] = rank_list

  def delete_seed_from_rank(self, rank_list, seed):
    rank_list = [row for row in rank_list if row[0] not in seed]
    return rank_list

  def propagate_seed(self, matrix, seed_attr, tol=1e-6, n = 1000, restart_factor = 0):
    seed_attr_old = seed_attr
    error = tol + 1 # Now error is more than tol
    k = 0
    while error > tol and k < n:
      if restart_factor > 0:
        seed_attr_new = (1 - restart_factor)*matrix@seed_attr_old + restart_factor*seed_attr
      else:
        seed_attr_new = matrix @ seed_attr_old
      # Normalization
      seed_attr_new = seed_attr_new / seed_attr_new.sum(0) 
      # Take error
      error = np.linalg.norm(seed_attr_new - seed_attr_old, ord=np.inf)
      seed_attr_old = seed_attr_new
      k += 1

    if k >= n: # TODO, let see the error.
      print("Convergence not achieved")
    
    return seed_attr_old

  def update_seed(self, genes_pos, propagate = False, options = {"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}):
    number_of_seed_genes = len(genes_pos)
    number_of_all_nodes = len(self.nodes)
    if propagate:
      seed_vector = np.zeros((number_of_all_nodes))
      seed_vector[genes_pos] = 1
      updated_seed = self.propagate_seed(self.matrix, seed_attr = seed_vector, tol = options["tolerance"], n = options["iteration_limit"], restart_factor = options["with_restart"])
      gen_list = updated_seed
    else:
      subsets_gen_values = self.matrix[genes_pos, :]
      integrated_gen_values = subsets_gen_values.sum(0)
      gen_list = (1/number_of_seed_genes) * integrated_gen_values

    return gen_list



  def rank_by_seed(self, seed_indexes, seed, propagate = False, options = {"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}):
    ordered_gene_score = []
    genes_pos = [seed_indexes.get(s) for s in seed if seed_indexes.get(s) is not None]
    number_of_seed_genes = len(genes_pos)
    number_of_all_nodes = len(self.nodes)

    if number_of_seed_genes > 0:

      gen_list = self.update_seed(genes_pos, propagate = propagate, options = options)

      ordered_indexes = np.argsort(gen_list)  # from smallest to largest

      last_val = None
      n_elements = ordered_indexes.shape[0]

      for pos in range(n_elements):
        order_index = ordered_indexes[pos]
        val = gen_list[order_index]
        node_name = self.nodes[order_index]

        rank = self.get_position_for_items_with_same_score(
            pos, val, last_val, gen_list, n_elements, ordered_gene_score)  # number of items behind
        rank = n_elements - rank  # number of nodes below or equal
        rank_percentage = rank/number_of_all_nodes

        ordered_gene_score.append([node_name, val, rank_percentage, rank])
        last_val = val

      ordered_gene_score.reverse()  # from largest to smallest
      ordered_gene_score = self.add_absolute_rank_column(ordered_gene_score)

    return ordered_gene_score

  def get_position_for_items_with_same_score(self, pos, val, prev_val, gen_list, n_elements, ordered_gene_score):
      members_behind = 0
      if prev_val is not None:
        if prev_val < val:
          members_behind = pos
        else:
          members_behind = n_elements - ordered_gene_score[-1][3]
      return members_behind

  def add_absolute_rank_column(self, ranking):
    ranking_with_new_column = []
    absolute_rank = 1
    n_rows = len(ranking)
    for row_pos in range(n_rows):
      if row_pos == 0:
        new_row = ranking[row_pos] + [absolute_rank]
        ranking_with_new_column.append(new_row)
      else:
        prev_val = ranking[row_pos-1][2]
        val = ranking[row_pos][2]
        if val > prev_val:
          absolute_rank += 1

        new_row = ranking[row_pos] + [absolute_rank]
        ranking_with_new_column.append(new_row)
    return ranking_with_new_column

  def get_individual_rank(self, seed_genes, node_of_interest, propagate, options = {"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}):
    genes_pos = self.get_nodes_indexes(seed_genes)
    if node_of_interest in self.nodes:
      node_of_interest_pos = self.nodes.index(node_of_interest)
    else:
      node_of_interest_pos = None

    ordered_gene_score = []
    if genes_pos and node_of_interest_pos is not None:
      integrated_gen_values = self.update_seed(genes_pos, propagate = propagate, options = options)

      ref_value = integrated_gen_values[node_of_interest_pos]

      members_below_test = 0
      for gen_value in integrated_gen_values:
        if gen_value >= ref_value:
          members_below_test += 1

      rank_percentage = members_below_test/len(self.nodes)
      rank = members_below_test
      rank_absolute = self.get_individual_absolute_rank(
          list(integrated_gen_values), ref_value)

      ordered_gene_score.append(
          [node_of_interest, ref_value, rank_percentage, rank, rank_absolute])

    return ordered_gene_score

  def get_individual_absolute_rank(self, values_list, ref_value):
    ref_pos = None
    values_list = sorted(list(set(values_list)), reverse=True)

    for pos, value in enumerate(values_list):
      if value == ref_value:
        ref_pos = pos+1
        break

    return ref_pos

  def get_reference_ranks(self):
    filtered_ranked_genes = {}

    for seed_name, ranking in self.ranking.items():
      if self.reference_nodes.get(seed_name) is None or not ranking:
        continue 

      ranking = self.array2hash(ranking, 0, range(0, len(ranking[0])))
      references=self.reference_nodes[seed_name]
      filtered_ranked_genes[seed_name]=[]

      for reference in references:
        rank=ranking.get(reference)
        if rank is not None:
          filtered_ranked_genes[seed_name].append(rank)

      filtered_ranked_genes[seed_name]=sorted(filtered_ranked_genes[seed_name], key=lambda rank: -rank[1])

    return filtered_ranked_genes

  def array2hash(self, arr, key, values):
    h={}
    for els in arr:
      h[els[0]]=[els[value] for value in values]
    return h

  def get_top(self, top_n):
    top_ranked_genes={}
    for seed_name, ranking in self.ranking.items():
      if ranking is not None:
        top_ranked_genes[seed_name] = ranking[0:top_n]
    return top_ranked_genes

  def get_nodes_indexes(self, nodes):
    node_indxs=[]
    for node in nodes:
      if node in self.nodes:
        index_node= self.nodes.index(node)
        node_indxs.append(index_node)

    return node_indxs

  def get_seed_indexes(self):
    indexes={}
    for node in sum(self.seeds.values(), []):
      if indexes.get(node) is None:
        if node in self.nodes:
          indx=self.nodes.index(node)
        else:
          indx=None

        if indx is not None:
          indexes[node]=indx

    return indexes
