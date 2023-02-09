#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
data_kernel=../test/data/data_kernel
out=output_test_scripts/netanalyzer
data_to_test=../test/data
mkdir $out


# Projections  -----------------------------------------------------------------------------------------------------------------------------------------------
# Perform a projection layer with jaccard
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -a $out/jaccard_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -m "jaccard" -u 'gen;pathway'
# Perform a projection layer with counts
netanalyzer.py -i $data_to_test/bipartite_network_for_validating.txt -a $out/counts_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' -m "counts" -u 'gen;pathway'
# Perform of transference method
netanalyzer.py -i $data_to_test/tripartite_network_for_validating.txt -a $out/transference_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+;salient,S[0-9]+' -m "transference" -u 'gen,salient;pathway'


# Obtaining kernels ---------------------------------------------------------------------------------------------------------------------------------------------
# Non normalized kernels.
netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' -K $out/ct_test -n $data_kernel/adj_mat.lst -u 'genes' -k 'ct' 
# Cosine Normalized kernels.
#netanalyzer.py -i $data_kernel/adj_mat.npy -f bin -l 'genes' -K $out/ka_test -n $data_kernel/adj_mat.npy -u 'genes' -k 'ka' -z


for file_to_test in `ls $out`; do
	echo $file_to_test
	diff $out/$file_to_test $data_to_test/$file_to_test
done
