#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
data_kernel=../test/data/data_kernel
out=output_test_scripts/netanalyzer
data_to_test=../test/data
data_test_scripts=data_test_scripts
mkdir -p $out


# Projections  -----------------------------------------------------------------------------------------------------------------------------------------------
# Perform a projection layer with jaccard
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -a $out/projections/jaccard_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -m "jaccard" -u 'gen;pathway'
# Perform a projection layer with counts
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -a $out/projections/counts_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -m "counts" -u 'gen;pathway'
# Perform of transference method
netanalyzer.py -i $data_to_test/tripartite_network_for_validating.txt -a $out/projections/transference_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+;salient,S[0-9]+' -m "transference" -u 'gen,salient;pathway'

# Obtaining kernels ---------------------------------------------------------------------------------------------------------------------------------------------
# Non normalized kernels.
netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' -K $out/kernels/ct -n $data_kernel/adj_mat.lst -u 'genes' -k 'ct' 
# Cosine Normalized kernels.
netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' -K $out/kernels/ka_normalized -n $data_kernel/adj_mat.lst -u 'genes' -k 'ka' -z


for file_to_test in `ls $out/projections`; do
	echo $file_to_test
	diff $out/projections/$file_to_test $data_to_test/$file_to_test
done

for file_to_test in `ls $out/kernels`; do
	echo $file_to_test
	diff $out/kernels/$file_to_test $data_kernel/$file_to_test
done

#PLotting
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=graphviz,layout=dot' -g ./test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=cyt_app' -g ./test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -g ./test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=cytoscape' -g ./test
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_options 'method=sigma' -g ./test

# Randoms
randomize_clustering.py -i $data_to_test/bipartite_network_for_validating.txt -o ./random_clusters.txt -r 'fixed:10:3'
randomize_network.py -i $data_to_test/monopartite_network_for_validating.txt -o ./random_net.txt -f pair -l 'nodes,[A-Z]' -r links

# Communities
# Create Communities
echo "Community discovery\n"
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -o ./output_test_scripts/clustering/ -l 'genes' -b "der" 
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -o ./output_test_scripts/clustering/ -l 'genes' -b "label_propagation"
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -f pair -o ./output_test_scripts/clustering/ -l 'genes' -b "gdmp2"
netanalyzer.py -i $data_to_test/counts_results.txt -f pair -o ./output_test_scripts/clustering/ -l 'genes' -b "rber_pots"
# Community Metrics
echo "Community metrics \n"
## Summ
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -o ./output_test_scripts/clustering/  -f pair -l 'genes' -M 'max_odf;avg_transitivity;conductance' -S
## Not Summ
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -o ./output_test_scripts/clustering/ -f pair -l 'genes' -M 'comparative_degree;max_odf'
# Comparing group families
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -R $data_test_scripts/clustering/der_discovered_clusters.txt -f pair -l 'genes' | tail -n 1 > ./output_test_scripts/clustering/comparing_clusters.txt
# Group expansion
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -G $data_test_scripts/clustering/clusters_toy.txt -o ./output_test_scripts/clustering/ -f pair -l 'genes' -x 'sht_path'

for file_to_test in `ls ./output_test_scripts/clustering`; do
	echo $file_to_test
	diff ./output_test_scripts/clustering/$file_to_test $data_test_scripts/clustering/$file_to_test
done