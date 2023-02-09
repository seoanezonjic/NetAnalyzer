#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
data_kernel=../test/data/data_kernel
out=output_test_scripts/netanalyzer
data_to_test=../test/data
mkdir $out


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
