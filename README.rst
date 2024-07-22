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


    Python library for network analysis, operations and priorization.

This package is designed to perform various steps in network analysis and processing through a modular design. Key features include:

* Randomization: Enables randomization of both clustered and individual nodes or edges within networks.
* Projections: Simplifies network complexity by reducing the number of layers based on connections from an excluded layer. For example, it can transition from a Phenotype-Patient-Mutation network to a Patient-Mutation network, connecting layers based on common nodes between patients and mutations.
* Topological Analysis: Computes various topological metrics for nodes (e.g., degree, betweenness) and provides summary statistics for entire networks.
* Cluster analysis: Performs metrics on predefined clusters and applies clustering algorithms based on the cdlib library.
* Embedding of networks (Kernels and node2vec): Defines node similarity using methods for processing context information in networks, including classical Kernel approaches and node2vec. It also supports integration of multiple layers.
* Prioritization: Applies propagation algorithms to prioritize nodes based on similarity metrics, such as the adjacency matrix, and a set of seed nodes.
* Net plotting: Provides several tools for graphing networks from different net plotter packages (igraph, cytoscape, graphviz).

Please, cite this library as: Rojano E., Seoane-Zonjic P., Bueno-Amorós A., Perkins JR., and Ranea JAG. Revealing the Relationship Between Human Genome Regions and Pathological Phenotypes Through Network Analysis. Lecture Notes in Computer Science, DOI: 10.1007/978-3-319-56148-6_17.