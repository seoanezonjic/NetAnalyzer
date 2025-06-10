import sys 
import numpy as np
from scipy import linalg
from warnings import warn
#from NetAnalyzer.adv_mat_calc import Adv_mat_calc
import py_exp_calc.exp_calc as pxc
from pecanpy import pecanpy
import numba #To control pecanpy threading
from gensim.models import Word2Vec
import nodevectors
import random # for the random walker

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx


class LINE(nn.Module):
    def __init__(self, num_nodes, embedding_dim=128, order=2, negative_ratio=5, learning_rate=0.01):
        super(LINE, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order
        self.negative_ratio = negative_ratio
        
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        if order == 2:
            self.context_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if order == 2:
            nn.init.xavier_uniform_(self.context_embedding.weight)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def negative_sampling(self, pos_edges, num_neg):
        neg_edges = []
        for src, _ in pos_edges:
            for _ in range(num_neg):
                dst = random.randint(0, self.num_nodes - 1)
                neg_edges.append((src, dst))
        return neg_edges

    def forward(self, src_nodes, dst_nodes, is_positive=True):
        src_embed = self.embedding(src_nodes)
        
        if self.order == 2:
            dst_embed = self.context_embedding(dst_nodes)
        else:
            dst_embed = self.embedding(dst_nodes)
        
        dot_product = torch.sum(src_embed * dst_embed, dim=1)
        
        if is_positive:
            return -F.logsigmoid(dot_product).mean()
        else:
            return -F.logsigmoid(-dot_product).mean()

    def train_model(self, edges, epochs=50):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            pos_src = torch.tensor([e[0] for e in edges], dtype=torch.long)
            pos_dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
            
            neg_edges = self.negative_sampling(edges, self.negative_ratio)
            neg_src = torch.tensor([e[0] for e in neg_edges], dtype=torch.long)
            neg_dst = torch.tensor([e[1] for e in neg_edges], dtype=torch.long)
            
            pos_loss = self.forward(pos_src, pos_dst, is_positive=True)
            neg_loss = self.forward(neg_src, neg_dst, is_positive=False)
            
            loss = pos_loss + neg_loss
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    def get_embeddings(self):
        return self.embedding.weight.detach().cpu().numpy()
                
class RandomWalker:

    def __init__(self, G, neigh_w, comm_w, community_nodes, num_walks, walk_length):
        
        self.G = G
        self.neigh_w = neigh_w
        self.comm_w = comm_w
        self.community_nodes = community_nodes # {# {community: [nodes]}}
        self.community_features = pxc.invert_hash(community_nodes) # {node: [communities]}
        self.num_walks = num_walks
        self.walk_length = walk_length

    def simulate_walks(self):
        random_walks = []
        for _ in range(self.num_walks):
            random_walks.extend(self.random_walk())
        return random_walks

    def random_walk(self):
        walks = []
        for node in self.G.nodes():
            walk = [node]
            nbs = list(self.G.neighbors(node))
            if len(nbs) > 0:
                walk.append(random.choice(nbs))
                for _ in range(2, self.walk_length):
                    v = self.walk_step(walk[-1])
                    if not v:
                        break
                    walk.append(v)
            walk = [str(x) for x in walk]
            walks.append(walk)
        return walks
    
    def walk_step(self,v):
        nbs = list(self.G.neighbors(v))
        weight_neigh = [1]*len(nbs)
        for i,_ in enumerate(nbs):
            weight_neigh[i] = self.neigh_w/len(nbs)

        if self.community_features.get(v):
            c_list = []
            for comm in self.community_features[v]:
                c_list.extend(self.community_nodes[comm])
            weight_comm = [1]*len(c_list)
            for i,_ in enumerate(c_list):
                weight_comm[i] = self.comm_w/len(c_list)
            all_nbs = nbs + c_list
            weights = weight_neigh+weight_comm
        else: 
            all_nbs = nbs
            weights = weight_neigh

        next_node = None
        if sum(weights) != 0:
            next_node = random.choices(all_nbs, weights=weights, k=1)[0]
        return next_node

class Graph2sim:

    allowed_embeddings = ['node2vec', 'deepwalk', 'prone', 'comm_aware', 'ggvec', 'grarep', 'glove','line']
    allowed_kernels = ['el', 'ct', 'rf', 'me', 'vn', 'rl', 'ka', 'md']

    def get_embedding(adj_mat, embedding_nodes, embedding = "node2vec", quiet=True, seed = None, clusters=None, embedding_kwargs={}):
        default_options = {"dimensions":64}
        comm_aware_options={"clusters": None, "neigh_w":1, "comm_w":1, "hs" : 0}
        random_walker_options={"walk_length":30, "num_walks": 200}
        w2v_options={"hs" : 0,"sg" : 0, "negative":5, "p" : 1, "q" : 1, "workers" : 16, "window" : 10, "min_count":1, 
                    "seed": None, "batch_words":4}
        prone_options={"step":10, "mu":0.2, "theta":0.5, "exponent":0.75}
        glove_options={"tol":"auto", "max_epoch":350,"max_count":50, "learning_rate":0.1,
                        "max_loss":10., "exponent": 0.33,"threads":0}
        ggvec_optinos={"tol_samples":30, "negative_ratio":0.15}
        grape_options={"order":1}
        default_options = {**default_options, **grape_options, **glove_options, **prone_options, 
                           **comm_aware_options, **random_walker_options, **w2v_options, **ggvec_optinos}
        default_options.update(embedding_kwargs)

        emb_coords = None
        verbose = not quiet
        if embedding in ['node2vec', 'deepwalk', 'comm_aware']: # TODO 'metapath2vec',
            if embedding == 'node2vec' or embedding == "deepwalk":
                if default_options["workers"] == 0:
                    default_options["workers"] = numba.config.NUMBA_DEFAULT_NUM_THREADS
                numba.set_num_threads(default_options["workers"]) # To config real threading in pecanpy
                workers = default_options["workers"]
                if embedding == "deepwalk":
                    default_options["p"] = 1
                    default_options["q"] = 1 
                g = pecanpy.DenseOTF.from_mat(adj_mat=adj_mat, node_ids=embedding_nodes,p=default_options["p"], q=default_options["q"], workers=default_options["workers"], random_state=123456, verbose= verbose)
                walks = g.simulate_walks(num_walks=default_options["num_walks"], walk_length=default_options["walk_length"])
            elif embedding == "comm_aware": # Based on CRARE paper and modified in HLC paper
                # Following paper CRARE: params 
                # --window_size 10
                # --num_walks 10
                # --walk_length 100
                # --num_iters 2
                # --dimensions 128
                # --min-count 0 
                # --epochs 10
                # --hs 1
                # --sg 1
                random_walker = RandomWalker(adj_mat, default_options["neigh_w"], default_options["comm_w"], clusters, default_options["num_walks"], default_options["walk_length"])
                walks = random_walker.simulate_walks()
            # use random walks to train embeddings
            model = Word2Vec(walks, vector_size=default_options["dimensions"], 
                             window=default_options["window"], min_count=default_options["min_count"],
                             workers=default_options["workers"], hs= default_options["hs"], sg = default_options["sg"],
                             negative=default_options["negative"], seed=123456) # point to extend
            list_arrays=[model.wv.get_vector(str(n)) for n in embedding_nodes]
        elif embedding == "line":
            g = nx.from_numpy_array(adj_mat)
            num_nodes = len(embedding_nodes)
            dim_each = default_options["dimensions"] // 2  
            line_model1 = LINE(num_nodes, embedding_dim=dim_each, order=1)
            line_model2 = LINE(num_nodes, embedding_dim=dim_each, order=2)
            edges = [(int(u), int(v)) for u, v in zip(*np.where(adj_mat))]
            line_model1.train_model(edges, epochs=50)
            line_model2.train_model(edges, epochs=50)
            emb1 = line_model1.get_embeddings()
            emb2 = line_model2.get_embeddings()
            list_arrays = [np.concatenate([emb1[i], emb2[i]]) for i in range(num_nodes)] # Concatenate embeddings for each node
        elif embedding in ["prone", "ggvec", "grarep", "glove"]:
            # embeddings from nodevectos repository: https://github.com/VHRanger/nodevectors.git
            if embedding == "prone":
                # best performance than node2vec: https://www.ijcai.org/proceedings/2019/0594.pdf
                g2v = nodevectors.ProNE(n_components=default_options["dimensions"])#, step=default_options["step"], mu=default_options["mu"], 
                                        #theta=default_options["theta"], exponent=default_options["exponent"], verbose=verbose)
            elif embedding == "ggvec":
                # "Best on large graphs and for visualization" from nodevectors repository
                g2v = nodevectors.GGVec(n_components=default_options["dimensions"])#, order=default_options["order"], learning_rate=default_options["learning_rate"], 
                                        #max_loss=default_options["learning_rate"], tol=default_options["tol"], tol_samples=default_options["tol_samples"], 
                                        #exponent=default_options["exponent"], threads=default_options["threads"], negative_ratio=default_options["negative_ratio"], 
                                        #max_epoch=default_options["max_epoch"], verbose=verbose)
            elif embedding == "grarep":
                # https://dl.acm.org/doi/abs/10.1145/2806416.2806512
                g2v = nodevectors.GraRep(n_components=default_options["dimensions"])#,order=default_options["order"],verbose=verbose)
            elif embedding == "glove":
                # Useful to embed sparse matrices of positive count, like pathway, phenotype, word co-occurence, etc.
                g2v = nodevectors.Glove(n_components=default_options["dimensions"])#,tol=default_options["tol"], max_epoch=default_options["max_epoch"],
                                         #max_count=default_options["max_count"], learning_rate=default_options["learning_rate"], 
                                         #max_loss=default_options["max_loss"], exponent=default_options["exponent"],threads=default_options["threads"], verbose=verbose)
            g2v.fit(adj_mat)
            list_arrays=[nodevec for nodevec in g2v.model.values()]
        else:
            print('Warning: The embedding method was not specified or not exists.')                                                                                                      
            sys.exit(0)
        n_cols=list_arrays[0].shape[0] # Number of col
        n_rows=len(list_arrays)# Number of rows
        emb_coords = np.concatenate(list_arrays).reshape([n_rows,n_cols]) # Concat all the arrays at one.
        return emb_coords

    def emb_coords2kernel(emb_coords, normalization = False, sim_type= "dotProduct"):
        kernel = pxc.coords2sim(emb_coords, sim = sim_type)
        if normalization: kernel = pxc.cosine_normalization(kernel)
        return kernel
    
    def kernel2emb_coords(kernel):
        U, S, _ = np.linalg.svd(kernel)
        emb_coords = U @ np.diag(np.sqrt(S))
        return emb_coords

    def get_kernel(matrix, kernel, normalization=False): #2exp?
        #I = identity matrix
        #D = Diagonal matrix
        #A = adjacency matrix
        #L = laplacian matrix = D − A
        matrix_result = None
        dimension_elements = np.shape(matrix)[1]
        # In scuba code, the diagonal values of A is set to 0. In weighted matrix the kernel result is the same with or without this operation. Maybe increases the computing performance?
        # In the md kernel this operation affects the values of the final kernel
        #dimension_elements.times do |n|
        #	matrix[n,n] = 0.0
        #end
        if kernel in ['el', 'ct', 'rf', 'me'] or 'vn' in kernel or 'rl' in kernel:
            diagonal_matrix = np.zeros((dimension_elements,dimension_elements))
            np.fill_diagonal(diagonal_matrix,matrix.sum(axis=1))	# get the total sum for each row, for this reason the sum method takes the 1 value. If sum colums is desired, use 0
                                                    # Make a matrix whose diagonal is row_sum
            matrix_L = diagonal_matrix - matrix
            if kernel == 'el': #Exponential Laplacian diffusion kernel(active). F Fouss 2012 | doi: 10.1016/j.neunet.2012.03.001
                beta = 0.02
                beta_product = matrix_L * -beta
                #matrix_result = beta_product.expm
                matrix_result = linalg.expm(beta_product)
            elif kernel == 'ct': # Commute time kernel (active). J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
                matrix_result = np.linalg.pinv(matrix_L, hermitian=True) # Hermitian parameter added to ensure convergence, just for real symmetric matrixes.
                # Anibal saids that this kernel was normalized. Why?. Paper do not seem to describe this operation for ct, it describes for Kvn or for all kernels, it is not clear.
            elif kernel == 'rf': # Random forest kernel. J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
                matrix_result = np.linalg.inv(np.eye(dimension_elements) + matrix_L) #Krf = (I +L ) ^ −1
            elif 'vn' in kernel: # von Neumann diffusion kernel. J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
                alpha = float(kernel.replace('vn', '')) * max(np.linalg.eigvals(matrix)) ** -1  # alpha = impact_of_penalization (1, 0.5 or 0.1) * spectral radius of A. spectral radius of A = absolute value of max eigenvalue of A 
                # TODO: The expresion max(np.linalg.eigvals(matrix)) obtain the max eigen computing all but in ruby was used a power series to compute directly this value. Implement this
                matrix_result = np.linalg.inv(np.eye(dimension_elements) - matrix * alpha ) #  (I -alphaA ) ^ −1
            elif 'rl' in kernel: # Regularized Laplacian kernel matrix (active)
                alpha = float(kernel.replace('rl', '')) * max(np.linalg.eigvals(matrix)) ** -1  # alpha = impact_of_penalization (1, 0.5 or 0.1) * spectral radius of A. spectral radius of A = absolute value of max eigenvalue of A
                matrix_result = np.linalg.inv(np.eye(dimension_elements) + matrix_L * alpha ) #  (I + alphaL ) ^ −1
            elif kernel == 'me': # Markov exponential diffusion kernel (active). G Zampieri 2018 | doi.org/10.1186/s12859-018-2025-5 . Taken from compute_kernel script
                beta=0.04
                #(beta/N)*(N*I - D + A)
                id_mat = np.eye(dimension_elements)
                m_matrix = (id_mat * dimension_elements - diagonal_matrix + matrix ) * (beta/dimension_elements)
                #matrix_result = m_matrix.expm
                matrix_result = linalg.expm(m_matrix)
        elif kernel == 'ka': # Kernelized adjacency matrix (active). J.-K. Heriche 2014 | doi: 10.1091/mbc.E13-04-0221
            lambda_value = min(np.linalg.eigvals(matrix)) # TODO implent as power series as shown in ruby equivalent
            matrix_result = matrix + np.eye(dimension_elements) * abs(lambda_value) # Ka = A + lambda*I # lambda = the absolute value of the smallest eigenvalue of A
        elif 'md' in kernel: # Markov diffusion kernel matrix. G Zampieri 2018 | doi.org/10.1186/s12859-018-2025-5 . Taken from compute_kernel script
            t = int(kernel.replace('md', ''))
            #TODO: check implementation
            col_sum = matrix.sum(axis=1)
            p_mat = np.divide(matrix.T,col_sum).T
            p_temp_mat = p_mat.copy()
            zt_mat = p_mat.copy()
            for i in range(0, t-1):
                p_temp_mat = np.dot(p_temp_mat,p_mat)
                zt_mat = zt_mat + p_temp_mat
            zt_mat = zt_mat * (1.0/t)
            matrix_result = np.dot(zt_mat, zt_mat.T)
        else:
            matrix_result = matrix
            warn('Warning: The kernel method was not specified or not exists. The adjacency matrix will be given as result')
            # This allows process a previous kernel and perform the normalization in a separated step.
        if normalization: matrix_result = pxc.cosine_normalization(matrix_result)  #TODO: check implementation with Numo::array
        return matrix_result
    
