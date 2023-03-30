#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
test_data=../test/data/data_ranker
out=output_test_scripts/ranker
data_to_test=data_test_scripts/ranker

# Deleting output files folder if already exists
if [ -d $out ]; then
	rm -r $out
fi
mkdir $out


# ranker by seeds -----------------------------------------------------------------------------------------------------------------------------------------------
##string
echo "$out/ranker_by_seed_string_results"
ranker.py -k $test_data/kernel_for_validating -n $test_data/kernel_for_validating.lst -s 'A,B' -o $out/ranker_by_seed_string_results
## file
ranker.py -k $test_data/kernel_for_validating -n $test_data/kernel_for_validating.lst -s $test_data/seed_genes_for_validating -o $out/ranker_by_seed_file_results

# ranker leaving one out ----------------------------------------------------------------------------------------------------------------------------------------
ranker.py -k $test_data/kernel_for_validating -n $test_data/kernel_for_validating.lst -s $test_data/seed_genes_for_validating -l -o $out/ranker_leave_one_out_by_seed_results

# ranker with filter --------------------------------------------------------------------------------------------------------------------------------------------
ranker.py -k $test_data/kernel_for_validating -n $test_data/kernel_for_validating.lst -s $test_data/seed_genes_for_validating -f $test_data/genes2filter_for_validating -o $out/ranker_filter_results

# ranker with top=2 ---------------------------------------------------------------------------------------------------------------------------------------------
ranker.py -k $test_data/kernel_for_validating -n $test_data/kernel_for_validating.lst -s $test_data/seed_genes_for_validating -t '2' --output_top $out/ranker_top_results
rm ./ranked_genes_all_candidates #TODO: this option is also generating "ranked_genes_all_candidates" file in the current working directory, even if only the option --output_top is used

for file_to_test in `ls $out`; do
	echo $file_to_test
	diff $out/$file_to_test $data_to_test/$file_to_test"_to_test"
done

