#!/usr/bin/env bash
source ~soft_bio_267/initializes/init_python
#python -m venv ~/tests/lucia/net --system-site-packages
source ~/tests/lucia/net/bin/activate


net_explorer -i "net,mock_net" --seed_nodes target_genes -N --graph_options 'method=sigma2,width=1200px,height=900px,iterations=200' --embedding_proj "umap" -G group_nodes -g network_umap