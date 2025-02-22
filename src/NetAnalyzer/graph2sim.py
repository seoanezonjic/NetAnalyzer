import sys 
import numpy as np
import scipy
from scipy import sparse, linalg
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from warnings import warn
import networkx as nx
#from NetAnalyzer.adv_mat_calc import Adv_mat_calc
import py_exp_calc.exp_calc as pxc
from pecanpy import pecanpy
from gensim.models import Word2Vec
import random # for the random walker

class ProNE:
        def __init__(self, n_components=32, step=10, mu=0.2, theta=0.5, 
                exponent=0.75, verbose=True):
            self.model = None
            self.n_components = n_components
            self.step = step
            self.mu = mu
            self.theta = theta
            self.exponent = exponent
            self.verbose = verbose
        
        def fit(self, adj_matrix, nodes):
            adj_matrix = csr_matrix(adj_matrix)
            features_matrix = self.pre_factorization(adj_matrix,
                                                     self.n_components, 
                                                     self.exponent)
            vectors = self.chebyshev_gaussian(
                adj_matrix, features_matrix, self.n_components,
                step=self.step, mu=self.mu, theta=self.theta)
            self.model = dict(zip(nodes, vectors))

        def tsvd_rand(self, matrix, n_components):
            """
            Sparse randomized tSVD for fast embedding
            """
            l = matrix.shape[0]
            smat = sparse.csc_matrix(matrix)
            U, Sigma, VT = randomized_svd(smat, 
                n_components=n_components, 
                n_iter=5, random_state=None)
            U = U * np.sqrt(Sigma)
            U = preprocessing.normalize(U, "l2")
            return U

        def pre_factorization(self, G, n_components, exponent):
            """
            Network Embedding as Sparse Matrix Factorization
            """
            C1 = preprocessing.normalize(G, "l1")
            # Prepare negative samples
            neg = np.array(C1.sum(axis=0))[0] ** exponent
            neg = neg / neg.sum()
            neg = sparse.diags(neg, format="csr")
            neg = G.dot(neg)
            # Set negative elements to 1 -> 0 when log
            C1.data[C1.data <= 0] = 1
            neg.data[neg.data <= 0] = 1
            C1.data = np.log(C1.data)
            neg.data = np.log(neg.data)
            C1 -= neg
            features_matrix = self.tsvd_rand(C1, n_components=n_components)
            return features_matrix

        def chebyshev_gaussian(self, G, a, n_components=32, step=10, 
                           mu=0.5, theta=0.5):
            """
            NE Enhancement via Spectral Propagation
            G : Graph (csr graph matrix)
            a : features matrix from tSVD
            mu : damping factor
            theta : bessel function parameter
            """
            nnodes = G.shape[0]
            if step == 1:
                return a
            A = sparse.eye(nnodes) + G
            DA = preprocessing.normalize(A, norm='l1')
            # L is graph laplacian
            L = sparse.eye(nnodes) - DA
            M = L - mu * sparse.eye(nnodes)
            Lx0 = a
            Lx1 = M.dot(a)
            Lx1 = 0.5 * M.dot(Lx1) - a
            conv = scipy.special.iv(0, theta) * Lx0
            conv -= 2 * scipy.special.iv(1, theta) * Lx1
            # Use Bessel function to get Chebyshev polynomials
            for i in range(2, step):
                Lx2 = M.dot(Lx1)
                Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
                if i % 2 == 0:
                    conv += 2 * scipy.special.iv(i, theta) * Lx2
                else:
                    conv -= 2 * scipy.special.iv(i, theta) * Lx2
                Lx0 = Lx1
                Lx1 = Lx2
                del Lx2
            mm = A.dot(a - conv)
            emb = self.svd_dense(mm, n_components)
            return emb
        
        def svd_dense(self, matrix, dimension):
            """
            dense embedding via linalg SVD
            """
            U, s, Vh = linalg.svd(matrix, full_matrices=False, 
                                  check_finite=False, 
                                  overwrite_a=True)
            U = np.array(U)
            U = U[:, :dimension]
            s = s[:dimension]
            s = np.sqrt(s)
            U = U * s
            U = preprocessing.normalize(U, "l2")
            return U
                
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

    allowed_embeddings = ['node2vec', 'deepwalk', 'prone', 'comm_aware']
    allowed_kernels = ['el', 'ct', 'rf', 'me', 'vn', 'rl', 'ka', 'md']

    def get_embedding(adj_mat, embedding_nodes, embedding = "node2vec", clusters=None, neigh_w=1, comm_w=1,
                    dimensions = 64, walk_length=30, num_walks = 200, hs = 0, sg = 0, negative=5,
                    p = 1, q = 1, workers = 16, window = 10, min_count=1, 
                    seed = None, quiet=False, batch_words=4):

        emb_coords = None
        if embedding in ['node2vec', 'deepwalk', 'comm_aware']: # TODO 'metapath2vec',
            if embedding == 'node2vec' or embedding == "deepwalk":
                if embedding == "deepwalk":
                    p = 1
                    q = 1 
                verbose = not quiet
                g = pecanpy.DenseOTF(p=p, q=q, workers=workers, verbose= verbose)
                g = g.from_mat(adj_mat=adj_mat, node_ids=embedding_nodes)
                walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)
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
                random_walker = RandomWalker(adj_mat, neigh_w, comm_w, clusters, num_walks, walk_length)
                walks = random_walker.simulate_walks()
            # use random walks to train embeddings
            model = Word2Vec(walks, vector_size=dimensions, 
                             window=window, min_count=min_count,
                             workers=workers, hs= hs, sg = sg,
                             negative=negative) # point to extend
            list_arrays=[model.wv.get_vector(str(n)) for n in embedding_nodes]
        elif embedding == "prone":
                g2v = ProNE()
                g2v.fit(adj_mat, embedding_nodes)
                list_arrays=[g2v.model[node] for node in embedding_nodes]
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