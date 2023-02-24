#!/usr/bin/env bash

source ~soft_bio_267/initializes/init_python
export PATH=../bin/:$PATH
out=output_test_scripts/text2binary_matrix
data_to_test=data_test_scripts/text2binary_matrix
mkdir -p $out

source ~soft_bio_267/initializes/init_ruby

text2binary_matrix.py -i $data_to_test/test_matrix_bin.npy -t "bin" -s -o $out/output_file > $out/statistics_from_text2bin
rm $out/output_file.npy

text2binary_matrix.py -i $data_to_test/test_matrix_bin.npy -t "bin" -o $out/cutoff_no_binarizado -c 0.5

text2binary_matrix.py -i $data_to_test/test_matrix_bin.npy -t "bin" -o $out/cutoff_binarizado -B 0.5

text2binary_matrix.py -i $data_to_test/test_matrix_bin.npy -t "bin" -o $out/set_diagonal_matrix -d

text2binary_matrix.py -i $data_to_test/test_matrix_bin.npy -t "bin" -O "bin" -o $out/test_matrix_bin.npy

text2binary_matrix.py -i $data_to_test/test_matrix -t "matrix" -O "mat" -o $out/test_matrixfrommatrix

text2binary_matrix.py -i $data_to_test/test_pairs -t "pair" -O "mat" -o $out/matrix_from_pairs

for file_to_test in `ls $out`; do
	echo $file_to_test
	diff $out/$file_to_test $data_to_test/$file_to_test
done