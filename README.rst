.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/NetAnalyzer.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/NetAnalyzer
    .. image:: https://readthedocs.org/projects/NetAnalyzer/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://NetAnalyzer.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/NetAnalyzer/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/NetAnalyzer
    .. image:: https://img.shields.io/pypi/v/NetAnalyzer.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/NetAnalyzer/
    .. image:: https://img.shields.io/conda/vn/conda-forge/NetAnalyzer.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/NetAnalyzer
    .. image:: https://pepy.tech/badge/NetAnalyzer/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/NetAnalyzer
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/NetAnalyzer

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
NetAnalyzer
===========


    Python package for network analysis, operations and priorization.

This package is prepared to perform different steps in network analysis and processing following a modular design. Key features of this library include:

* Randomization: Both clustered and per-node or per-edge networks can be randomized. 
* Projections: Decrease the number of layers of a network based on the connections of an excluded layer. E.g.: Moving from a Phenotype-Patient-Mutation network to a Patient-Mutation network, connecting the two layers based on the number of common nodes between patient and mutation.
* Topological Analysis: Obtain different topological graph metrics by node (degree, betweness,etc.) or summarizing for a given network. 
* Cluster analysis: Able to perform certain metrics on predefined clusters and clusterize based on cdlib library.
* Embedding of networks (Kernels and node2vec). Define similarity between nodes by different methods of processing context information in networks. The main building blocks are the classical Kernels approach and the additional node2vec approach. With the additional capacity to integrate different layers.
* Prioritization: Given a node similarity matrix (the adjacency matrix, for example) and a set of nodes (seed), it is possible to apply propagation algorithm to prioritize by evarage similarity.
* Net plotting: Provides several tools for graphing networks from different net plotter packages (igraph, cytoscape, graphviz).
