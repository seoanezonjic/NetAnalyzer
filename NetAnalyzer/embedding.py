import sys 
import numpy as np
from scipy import linalg
from warnings import warn
import networkx as nx
from node2vec import Node2Vec
#from gensim.models import Word2Vec
#model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=2, iter=1)

# TODO, check stellargraph implemmentation.

class Embedding:

	def get_embedding(graph, node_names, embedding, p = 1, q = 1):
		emb_coords = None
		if embedding in ['node2vec', 'deepwalk']: # TODO 'metapath2vec',
			if embedding == 'node2vec' or "deepwalk":
				if embedding == "deepwalk":
					p = 1
					q = 1 
					
				node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=10, p = p, q = q, workers = 1)
				model = node2vec.fit(window=5, min_count= 0) # batch_words=10000
				# min_count: is the minimun number of counts a word must have.
				# batch_words: are Target size (in words) for batches of examples
				list_arrays=[model.wv.get_vector(str(n)) for n in graph.nodes()]
				n_cols=list_arrays[0].shape[0] # Number of col
				n_rows=len(list_arrays)# Number of rows
				emb_coords = np.concatenate(list_arrays).reshape([n_rows,n_cols]) # Concat all the arrays at one.
		else:
			print('Warning: The embedding method was not specified or not exists.')                                                                                                      
            sys.exit(0)

            return emb_coords

	def emb_coords2kernel(self, emb_coords, cosine_norm = False):
		return emb_coords.dot(emb_coords.T) 


	# Section of customized embeddings #
	# RandomWalker -> (Corpus) -> Word2Vec -> (Embedding)
	def multigraph2vec():
		pass