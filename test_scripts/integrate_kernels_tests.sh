#!/usr/bin/env bash

#source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
out=output_test_scripts/integrate_kernels
data_to_test=data_test_scripts/integrate_kernels
mkdir -p $out

integrate_kernels.py -i "mean" -t "$data_to_test/kernel1.npy\t$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst\t$data_to_test/kernel2.lst " -o $out/int_mean
integrate_kernels.py -i "integration_mean_by_presence" -t "$data_to_test/kernel1.npy\t$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst\t$data_to_test/kernel2.lst " -o $out/int_mean_by_presence

for file_to_test in `ls $out`; do
	echo $file_to_test
	diff $out/$file_to_test $data_to_test/$file_to_test
done