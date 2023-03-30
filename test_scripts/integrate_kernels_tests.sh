#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
out=output_test_scripts/integrate_kernels
data_to_test=data_test_scripts/integrate_kernels

#Delete integrate kernels output folder if already exist
if [ -d $out ]; then
	rm -r $out
fi
mkdir -p $out

integrate_kernels.py -i "mean" -t "$data_to_test/kernel1.npy;$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_mean
integrate_kernels.py -i "integration_mean_by_presence" -t "$data_to_test/kernel1.npy;$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_mean_by_presence
integrate_kernels.py -i "median" -t "$data_to_test/kernel1.npy;$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_median
integrate_kernels.py -i "max" -t "$data_to_test/kernel1.npy;$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_max
integrate_kernels.py -i "geometric_mean" -t "$data_to_test/kernel1.npy;$data_to_test/kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_geometric_mean

# Testing also assymetric kernels
integrate_kernels.py --asym -i "mean" -t "$data_to_test/asym_kernel1.npy;$data_to_test/asym_kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_mean_asym
integrate_kernels.py --asym -i "integration_mean_by_presence" -t "$data_to_test/asym_kernel1.npy;$data_to_test/asym_kernel2.npy" -n " $data_to_test/kernel1.lst;$data_to_test/kernel2.lst " -o $out/int_mean_by_presence_asym

for file_to_test in `ls $out`; do
	echo $file_to_test
	diff $out/$file_to_test $data_to_test/$file_to_test
done

# Testing integration speed with 1 and 12 CPUs
python $data_to_test/create_temporal_big_matrices.py
mv ./bigkernel* $data_to_test/

START_TIME_1CPU=$(date +%s%3N)
integrate_kernels.py -i "mean" -t "$data_to_test/bigkernel1.npy;$data_to_test/bigkernel2.npy;$data_to_test/bigkernel3.npy" -n " $data_to_test/bigkernel1.lst;$data_to_test/bigkernel2.lst;$data_to_test/bigkernel3.lst " -o $out/int_mean_big --cpu 1
ELAPSED_TIME_1CPU=$(expr $(date +%s%3N) - $START_TIME_1CPU)

START_TIME_12CPU=$(date +%s%3N)
integrate_kernels.py -i "mean" -t "$data_to_test/bigkernel1.npy;$data_to_test/bigkernel2.npy;$data_to_test/bigkernel3.npy" -n " $data_to_test/bigkernel1.lst;$data_to_test/bigkernel2.lst;$data_to_test/bigkernel3.lst " -o $out/int_mean_big --cpu 12
ELAPSED_TIME_12CPU=$(expr $(date +%s%3N) - $START_TIME_12CPU)

if [ $ELAPSED_TIME_1CPU -lt $ELAPSED_TIME_12CPU ]; then
	echo "Integration with 12 CPUs is being slower than with 1 CPU"
	echo "1 CPU: $ELAPSED_TIME_1CPU milliseconds"
	echo "12 CPU: $ELAPSED_TIME_12CPU milliseconds"
fi

rm $data_to_test/bigkernel*
rm $out/int_mean_big*