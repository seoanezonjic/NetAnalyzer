import numpy as np
import os, re
from itertools import combinations
from sklearn.model_selection import KFold, LeaveOneOut
from NetAnalyzer.adv_mat_calc import Adv_mat_calc
from NetAnalyzer.seed_parser import SeedParser
import scipy.stats as stats
import py_exp_calc.exp_calc as pxc
import networkx as nx
import random 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class Ranker:

    def __init__(self):
        self.matrix = None
        self.nodes = []  # kernel nodes
        self.seeds = {}  # node seeds 
        self.weights = {}
        self.negatives = None # Could be 'all' or a dict with the negative nodes
        self.seed_nodes2idx = {} 
        self.reference_nodes = {} 
        self.ranking = {} 
        self.discarded_seeds = {}
        self.in_names = True
        self.seed_presence = True
        self.training_dataset = {"score":[], "label":[]} # [score, label], 0 negative or 1 positive
        self.attributes = {"header": ["candidates", "score", "normalized_rank", "rank", "uniq_rank"]}
        self.network = None

    # Normalization and filtering matrix
    def normalize_matrix(self, mode="by_column"):
        degree_matrix = self.matrix.sum(0)
        inv_degree_matrix = np.diag(1/degree_matrix)
        if mode == "by_column":
            self.matrix = self.matrix @ inv_degree_matrix
        elif mode == "by_row":
            self.matrix = inv_degree_matrix  @ self.matrix
        elif mode == "by_row_col":
            self.matrix = np.sqrt(
                inv_degree_matrix) @ self.matrix @ np.sqrt(inv_degree_matrix)

    def filter_matrix(self, whitelist):
        self.matrix, self.nodes, _ = Adv_mat_calc.filter_rowcols_by_whitelist(
            self.matrix, self.nodes, self.nodes, whitelist, symmetric=True)
        
    def score2pvalue_matrix(self, mode="znormalization"):
        if mode == "znormalization":
            mean_score = np.mean(self.matrix)
            std_score = np.std(self.matrix)
            z_scores = (self.matrix - mean_score) / std_score
            p_values = 1 - stats.norm.cdf(z_scores)  # convert to p-values
        elif mode == "quantile":
            ranks = rankdata(self.matrix, method="average") 
            p_values = 1 - (ranks / (scores.size + 1)) # Pseudo p-values
        elif mode == "logistic":
            logistic_model = self.train_logistic_model()
            #print(logistic_model.predict_proba(self.matrix[1].reshape(-1, 1))[:,1])
            p_values = np.array([logistic_model.predict_proba(self.matrix[i].reshape(-1, 1))[:,1] for i in range(len(self.matrix))])
            #p_values = 1 / (1 + np.exp(-scores))
        self.matrix = p_values

    def train_logistic_model(self):
        model = LogisticRegression()
        model.fit(np.array(self.training_dataset["score"]).reshape(-1,1), self.training_dataset["label"])
        return model

    def load_training_dataset(training_dataset_path):
        with open(training_dataset_path, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                node1 = line[0]
                node2 = line[1]
                label = 1 if line[2] == "P" else 0
                row = self.nodes.index(node1)
                col = self.nodes.index(node2)
                score = self.matrix[row,col]
                self.training_dataset["score"].append(score)
                self.training_dataset["label"].append(label)

    def generate_training_dataset(self, info_to_use="edges", seed = 42):
        if info_to_use == "edges":
            # Collect positives
            for edge in self.network.graph.edges():
                node1 = self.nodes.index(edge[0])
                node2 = self.nodes.index(edge[1])
                score = self.matrix[node1,node2]
                self.training_dataset["score"].append(score)
                self.training_dataset["label"].append(1)
            # Collect negatives
            self.generate_neg_edges(self.network.graph, len(self.training_dataset["score"]), seed)
        else:
            raise Exception(f"Sorry, the method {info_to_use} doesn't exist for generate training dataset")

    def generate_neg_edges(self, G, testing_edges_num, seed=42):
        # Code from: https://github.com/VHRanger/nodevectors/blob/master/nodevectors/evaluation/link_pred.py
        nnodes = G.number_of_nodes()
        negG = np.ones((nnodes, nnodes))
        np.fill_diagonal(negG, 0.)
        original_graph = nx.to_numpy_array(G)
        negG -= original_graph
        neg_edges = np.where(negG > 0)
        neg_edge_indices = list(zip(neg_edges[0], neg_edges[1]))
        testing_edges_num = min(testing_edges_num, len(neg_edge_indices))
        random.seed(seed)
        rng_edges = random.sample(neg_edge_indices, testing_edges_num)
        # return edges in (src, dst) tuple format
        for src_idx, dst_idx in rng_edges:
            node1 = list(G.nodes)[src_idx]
            node2 = list(G.nodes)[dst_idx]
            self.training_dataset["score"].append(self.score_from_edge(node1, node2))
            self.training_dataset["label"].append(0)

    def score_from_edge(self, node1, node2):
        return self.matrix[self.nodes.index(node1), self.nodes.index(node2)]

    # loading and cleaning list of nodes
    def load_nodes_from_file(self, file):
        with open(file) as f:
            for line in f:
                self.nodes.append(line.rstrip())

    def load_seeds(self, node_groups, sep=',', uniq=True):
        self.seeds, self.weights = SeedParser.load_nodes_by_group(node_groups, sep=sep)
        if uniq:
            self.seeds = {seed: pxc.uniq(nodes) for seed, nodes in self.seeds.items()}
    
    def load_negatives(self, negative_file, sep=','):
        if  os.path.exists(negative_file):
            self.negatives, _ = SeedParser.load_node_groups_from_file(negative_file, sep=sep)
        elif negative_file == "all":
            self.negatives = negative_file

    def load_references(self, node_groups, sep=','):
        self.reference_nodes, _ = SeedParser.load_nodes_by_group(node_groups, sep=sep)

    def clean_seeds(self, minimum_size = 1):
        cleaned_seeds = {}
        cleaned_weights = {}
        for seed_name, seed in self.seeds.items():
            cleaned_seed = [ node for node in seed if node in self.nodes ]
            if len(cleaned_seed) < minimum_size:
                self.discarded_seeds[seed_name] = seed
            else:
                cleaned_seeds[seed_name] = cleaned_seed
            if self.weights.get(seed_name) is not None: 
                cleaned_weights[seed_name] = { node: self.weights[seed_name][node] for node in cleaned_seed }
        self.seeds = cleaned_seeds
        self.weights = cleaned_weights


    # id <-> name translation
    def get_nodes_indexes(self, nodes):
        node_indxs = []
        #if type(nodes) == int: nodes = [nodes]
        for node in nodes:
            index_node = self.seed_nodes2idx[node]
            node_indxs.append(index_node)

        return node_indxs

    def get_seed_indexes(self):
        indexes = {}
        for node in sum(self.seeds.values(), []):
            if indexes.get(node) is None:
                indx = self.nodes.index(node)
                indexes[node] = indx
        self.seed_nodes2idx = indexes

    def translate2idx(self):
        self.in_names = False
        for seed_name, seed in self.seeds.items():
            self.seeds[seed_name] = self.get_nodes_indexes(seed)
        for seed_name, node2weight in self.weights.items():
            node2weight = { self.seed_nodes2idx[node]: weight for node, weight in node2weight.items()}
            self.weights[seed_name] = node2weight

    def translate2names(self):
        self.in_names = True
        for seed_name, seed in self.seeds.items():
            self.seeds[seed_name] = [self.nodes[node] for node in seed]
        for seed_name, node2weight in self.weights.items():
            node2weight = { self.nodes[node]: weight for node, weight in node2weight.items()}
            self.weights[seed_name] = node2weight

    # Ranking
    ## Cross validation

    def get_seed_cross_validation(self, k_fold=None):
        new_seeds = {}
        new_weights = {}
        genes2predict = {}

        for seed_name, seed in self.seeds.items():
            if k_fold != None:
                cv = KFold(n_splits=k_fold)
            else:
                cv = LeaveOneOut()

            for indx, (train_index, test_index) in enumerate(cv.split(seed)):
                seed_name_one_out = str(seed_name) + "_iteration_" + str(indx)
                #print("train index is:", train_index)
                new_seeds[seed_name_one_out] = [seed[i] for i in train_index]

                if self.weights.get(seed_name):
                    new_weights[seed_name_one_out] = {node: self.weights[seed_name][node] for node in new_seeds[seed_name_one_out]}

                if self.in_names:
                    genes2predict[seed_name_one_out] = [seed[i] for i in test_index]
                else:
                    genes2predict[seed_name_one_out] = [seed[i] for i in test_index]
                    genes2predict[seed_name_one_out] = [self.nodes[idx] for idx in genes2predict[seed_name_one_out]]
                if self.reference_nodes.get(seed_name) is not None:
                    genes2predict[seed_name_one_out] += self.reference_nodes[seed_name]

                genes2predict[seed_name_one_out] = list(
                    set(genes2predict[seed_name_one_out]))

        self.seeds = new_seeds
        self.reference_nodes = genes2predict
        self.weights = new_weights

    def get_loo_ranking(self, seed_name, seed, metric = "mean"):
         # Generate new seeds
         # Get matrix values
         cv = LeaveOneOut()
         ncols= len(self.nodes)
         nrows = len(seed)
         W = np.zeros((nrows,ncols))
         nodes2predict_pos = []
         nodes2predict_names = []
         new_seed_names = []
         for indx, (train_index, test_index) in enumerate(cv.split(seed)):
            new_seed_names.append(str(seed_name) + "_iteration_" + str(indx))
            nodes = [seed[i] for i in train_index]
            if self.weights.get(seed_name):
                w = [self.weights[seed_name][node] for node in nodes]
                W[indx, nodes] = w
            else:
                W[indx, nodes] = 1
            #print(W)
            node2predict = seed[test_index[0]]
            nodes2predict_pos.append(node2predict)
            nodes2predict_names.append(self.nodes[node2predict])
         # recoger los valores correspondientes
         # Calcular los score
         if metric == "mean":
            R = W @ self.matrix / (nrows - 1)
         elif metric == "max":
            R = np.zeros(self.matrix.shape)
            for row in range(0,self.matrix.shape[0]):
                R[row] = (W[row,:] * self.matrix).max(0)
         scores = R[range(0,nrows), nodes2predict_pos]
         # Calcular los members_below
         if self.seed_presence:
           members_below = (R >= scores.reshape(nrows,1)).sum(1)
           # Calcular los porcentage
           rank_percentage = members_below/ncols
         else:
            R[W != 0] =  np.min(scores) -1
            members_below = (R >= scores.reshape(nrows, 1)).sum(1)
            rank_percentage = members_below/(ncols-nrows+1)
         # Calcular los absolute rank
         return list(zip(new_seed_names, nodes2predict_names, scores, rank_percentage, members_below))

    ## Ranking by seed
    def do_ranking(self, cross_validation=False, k_fold=None, propagate=False,  metric = "mean", options={"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}):
        self.get_seed_indexes()
        self.translate2idx()

        if cross_validation and k_fold is not None:
            self.get_seed_cross_validation(k_fold=k_fold)

        ranked_lists = []
        for seed_name, seed in self.seeds.items():
            # The code in this block CANNOT modify nothing outside
            if cross_validation and k_fold is None:
                self.attributes["header"] = ["candidates", "score", "normalized_rank", "rank"]
                rank_list = self.get_loo_ranking(seed_name, seed, metric=metric)
                #print(rank_list)
                for row in rank_list:
                    ranked_lists.append([row[0], [list(row[1:])]])
            else:
                rank_list = self.rank_by_seed(seed, weights=self.weights.get(seed_name), propagate=propagate, metric = metric, options=options)  # Production mode
                if cross_validation and k_fold is not None:
                    rank_list = self.delete_seed_from_rank(rank_list, [self.nodes[pos] for pos in self.seeds[seed_name]])
                ranked_lists.append([seed_name, rank_list])

        for seed_name, rank_list in ranked_lists:  # Transfer resuls to hash
            self.ranking[seed_name] = rank_list

        self.translate2names()

    def rank_by_seed(self, seed, weights=None, propagate=False, metric = "mean", options={"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}):
        ordered_gene_score = []
        genes_pos = seed
        number_of_all_nodes = len(self.nodes)
        if self.seed_presence:
            genes_of_interest = list(range(0,number_of_all_nodes))
            nodes = self.nodes
        else:
            genes_of_interest = list(set(range(0,number_of_all_nodes))-set(genes_pos))
            nodes = [self.nodes[idx] for idx in genes_of_interest]
        if weights: weights = np.array([weights[s] for s in seed])
        number_of_seed_genes = len(genes_pos)

        if number_of_seed_genes > 0:
            gen_list = self.update_seed(
                genes_pos, genes_of_interest, weights=weights, propagate=propagate, metric = metric, options=options)
            ordered_gene_score = pxc.get_rank_metrics(gen_list, ids=nodes)

        return ordered_gene_score

    def update_seed(self, genes_pos, genes_of_interest, weights=None, propagate=False, metric = "mean", options={"tolerance": 1e-9, "iteration_limit": 100, "with_restart": 0}):
        number_of_seed_genes = len(genes_pos)
        number_of_all_nodes = len(self.nodes)

        if propagate:
            # TODO: Weight extension on this area
            # TODO: This option soe not accept remove the seeds.
            seed_vector = np.zeros((number_of_all_nodes))
            seed_vector[genes_pos] = 1
            updated_seed = self.propagate_seed(
                self.matrix, seed_attr=seed_vector, tol=options["tolerance"], n=options["iteration_limit"], restart_factor=options["with_restart"])
            gen_list = updated_seed[genes_of_interest]
        else:
            subsets_gen_values = self.matrix[genes_pos,:][:,genes_of_interest]

            if metric == "max":
                gen_list = subsets_gen_values.max(0)
            elif metric == "mean": 
                if weights is not None:         
                    integrated_gen_values = weights @ subsets_gen_values
                    gen_list = (1/weights.sum()) * integrated_gen_values
                else:
                    integrated_gen_values = subsets_gen_values.sum(0)
                    gen_list = (1/number_of_seed_genes) * integrated_gen_values
            elif metric == "bayesian":
                #p_values = np.clip(subsets_gen_values, 1e-100, 1 - 1e-100)
                p_values = subsets_gen_values
                gen_list = 1 - np.prod(1 - p_values, axis=0)
            elif metric == "fisher":
                p_values = np.clip(subsets_gen_values, 1e-10, 1)
                chi2_stat = -2 * np.sum(np.log(p_values), axis=0)
                gen_list = 1 - stats.chi2.cdf(chi2_stat, df=2 * p_values.shape[0])
            elif metric == "stouffer":
                def stouffer_combination(p_values, weights=None):
                    if weights is None:
                        weights = np.ones(len(p_values))  
                    z_scores = stats.norm.ppf(1 - np.clip(p_values, 1e-10, 1))
                    z_combined = np.sum(weights[:, np.newaxis] * z_scores, axis=0) / np.sum(weights)
                    p_combined = 1 - stats.norm.cdf(z_combined)
                    
                    return p_combined
                gen_list = stouffer_combination(subsets_gen_values, weights)
                gen_list = 1 - gen_list
        return gen_list

    def propagate_seed(self, matrix, seed_attr, tol=1e-6, n=1000, restart_factor=0):
        seed_attr_old = seed_attr
        error = tol + 1  # Now error is more than tol
        k = 0
        while error > tol and k < n:
            if restart_factor > 0:
                seed_attr_new = (1 - restart_factor) * \
                    matrix@seed_attr_old + restart_factor*seed_attr
            else:
                seed_attr_new = matrix @ seed_attr_old
            # Normalization
            seed_attr_new = seed_attr_new / seed_attr_new.sum(0)
            # Take error
            error = np.linalg.norm(seed_attr_new - seed_attr_old, ord=np.inf)
            seed_attr_old = seed_attr_new
            k += 1

        if k >= n:  # TODO, let see the error.
            print("Convergence not achieved")

        return seed_attr_old

    # Output and parsing ranking

    def get_filtered_ranks_by_reference(self, cross_validation=False):
        filtered_ranked_genes = {}

        if cross_validation:
            iterate_filter_name = lambda seed_name: re.sub(r"_iteration_.*", "", seed_name)
        else:
            iterate_filter_name = lambda seed_name: seed_name

        for seed_name, ranking in self.ranking.items():
            if self.reference_nodes.get(iterate_filter_name(seed_name)) is None or not ranking:
                continue

            ranking = self.array2hash(ranking, 0, range(0, len(ranking[0])))
            references = self.reference_nodes[iterate_filter_name(seed_name)]
            filtered_ranked_genes[seed_name] = []

            for reference in references:
                rank = ranking.get(reference)
                if rank is not None:
                    filtered_ranked_genes[seed_name].append(rank)

            filtered_ranked_genes[seed_name] = sorted(
                filtered_ranked_genes[seed_name], key=lambda rank: -rank[1])

        return filtered_ranked_genes

    def write_ranking(self, output_name, add_header=True, top_n=None): 
        if add_header:
            header = self.attributes["header"]
            header.append("seed_group")
        if top_n is not None:
            rankings = self.get_top(top_n)
        else:
            rankings = self.ranking
        with open(output_name, 'w') as f:
            if add_header: f.write('\t'.join(header)+"\n")
            for seed_name, ranking in rankings.items():
                for row in ranking:
                    f.write('\t'.join(map(str,row)) + "\t" + f"{seed_name}" + "\n")  

    def add_candidate_tag_types(self):
        rankings = {}
        for seed_name, rankings_by_seed in self.ranking.items():
            added_ranking_column = []
            for ranking in rankings_by_seed:
                if ranking[0] in self.seeds[seed_name]:
                    ranking.insert(5, "seed")
                else:
                    ranking.insert(5, "new")
                added_ranking_column.append(ranking)
            rankings[seed_name] = added_ranking_column
        self.ranking = rankings
        self.attributes["header"].insert(5,"type")

    def get_top(self, top_n):
        top_ranked_genes = {}
        for seed_name, ranking in self.ranking.items():
            if ranking is not None:
                top_ranked_genes[seed_name] = ranking[0:top_n]
        return top_ranked_genes

    # Auxiliar methods
    def array2hash(self, arr, key, values):  # 2exp?
        h = {}
        for els in arr:
            h[els[0]] = [els[value] for value in values]
        return h

    def delete_seed_from_rank(self, rank_list, seed):
        rank_list = [row for row in rank_list if row[0] not in seed]
        return rank_list