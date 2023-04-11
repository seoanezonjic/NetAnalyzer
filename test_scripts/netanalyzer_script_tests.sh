#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
data_kernel=../test/data/data_kernel
data_to_test=../test/data

out=output_test_scripts/netanalyzer
data_test_scripts=data_test_scripts/netanalyzer

# Deleting netanalyzer output files folder if already exists
if [ -d $out ]; then
	rm -r $out
fi

mkdir -p $out
ls -R $out | wc -l
mkdir -p $out/random
mkdir -p $out/plots
mkdir -p $out/projections
mkdir -p $out/kernels
mkdir -p $out/clustering



####### NETANALYZER TESTS #######

# ---------------------------------- Projections  ---------------------------------------------
# Perform a projection layer with jaccard
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -a $out/projections/jaccard_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -m "jaccard" -u 'gen;pathway'
# Perform a projection layer with counts
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -a $out/projections/counts_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -m "counts" -u 'gen;pathway'
# Perform of transference method
netanalyzer.py -i $data_to_test/tripartite_network_for_validating.txt -a $out/projections/transference_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+;salient,S[0-9]+' -m "transference" -u 'gen,salient;pathway'

#  ---------------------------------- Obtaining kernels ----------------------------------------
# Non normalized kernels.
netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' -K $out/kernels/ct -n $data_kernel/adj_mat.lst -u 'genes' -k 'ct' 
# Cosine Normalized kernels.
netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' -K $out/kernels/ka_normalized -n $data_kernel/adj_mat.lst -u 'genes' -k 'ka' -z
# kernels with embedding Node2Vec
netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' --embedding_add_options "'window':10,'p':0.7,'seed':123" --both_repre_formats -K $out/kernels/node2vec -n $data_kernel/adj_mat.lst -u 'genes' -k 'node2vec'

for file_to_test in `ls $out/projections`; do
 	echo $file_to_test
 	diff $out/projections/$file_to_test $data_to_test/$file_to_test
done

for file_to_test in `ls $out/kernels`; do
	echo $file_to_test
	diff $out/kernels/$file_to_test $data_kernel/$file_to_test
done

# ---------------------------------- Plotting ----------------------------------------
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=graphviz,layout=dot' -g $out/plots/test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=cyt_app' -g $out/plots/test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -g $out/plots/test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=cytoscape' -g $out/plots/test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=sigma' -g $out/plots/test

# ---------------------------------- Randoms ----------------------------------------
randomize_clustering.py -i $data_to_test/bipartite_network_for_validating.txt -o $out/random/random_clusters.txt -r 'fixed:10:3'
randomize_network.py -i $data_to_test/monopartite_network_for_validating.txt -o $out/random/random_net.txt -f pair -l 'nodes,[A-Z]' -r links
randomize_network.py -i $data_to_test/monopartite_network_for_validating.txt -o $out/random/random_net_same_seed1.txt -f pair -l 'nodes,[A-Z]' -r links --seed 1
randomize_network.py -i $data_to_test/monopartite_network_for_validating.txt -o $out/random/random_net_same_seed2.txt -f pair -l 'nodes,[A-Z]' -r links --seed 1
diff $out/random/random_net_same_seed1.txt $out/random/random_net_same_seed2.txt #We should expect no difference is the seed is the same

# ---------------------------------- Comunities ----------------------------------------
# Create Communities
netanalyzer.py -i $data_to_test/counts_results.txt -f pair -o ./$out/clustering/ -l 'genes' -b "rber_pots" #Deternimistic algorithm
netanalyzer.py -i $data_to_test/counts_results.txt -f pair -o ./$out/clustering/ -l 'genes' -b "der" --seed 1 #Non-deterministic algorithm
mv ./$out/clustering/der_discovered_clusters.txt ./$out/clustering/der_discovered_clusters2.txt
netanalyzer.py -i $data_to_test/counts_results.txt -f pair -o ./$out/clustering/ -l 'genes' -b "der" --seed 1
diff ./$out/clustering/der_discovered_clusters.txt ./$out/clustering/der_discovered_clusters2.txt #We should expect no difference if the seed is the same
rm ./$out/clustering/der_discovered_clusters.txt ./$out/clustering/der_discovered_clusters2.txt #We remove the files to avoid unexpected errors that could not overwrite the file
# Community Metrics
## Summ
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -o ./$out/clustering/  -f pair -l 'genes' -M 'max_odf;avg_transitivity;conductance' -S
## Not Summ
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -o ./$out/clustering/ -f pair -l 'genes' -M 'comparative_degree;max_odf'
# Comparing group families
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -R $data_test_scripts/clustering/rber_pots_discovered_clusters.txt -f pair -l 'genes' | tail -n 1 > ./$out/clustering/comparing_clusters.txt
# Group expansion
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -o ./$out/clustering/ -f pair -l 'genes' -x 'sht_path'
sort -k1 ./$out/clustering/expand_clusters.txt | sort -k2 > ./$out/clustering/tmp_expand_clusters.txt
mv ./$out/clustering/tmp_expand_clusters.txt ./$out/clustering/expand_clusters.txt

for file_to_test in `ls ./$out/clustering`; do
	echo $file_to_test
	diff ./$out/clustering/$file_to_test $data_test_scripts/clustering/$file_to_test
done

