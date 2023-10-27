import pytest
import sys
import os 
from io import StringIO
import shutil
import NetAnalyzer
import numpy as np
from py_cmdtabs import CmdTabs
ROOT_PATH=os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'data')

INPUT_PATH = os.path.join(DATA_PATH, 'input_scripts')
INTEGRATE_KERNELS = os.path.join(INPUT_PATH, 'integrate_kernels')
RANDOMIZE_CLUSTERING = os.path.join(INPUT_PATH, 'randomize_clustering')
RANDOMIZE_NETWORK = os.path.join(INPUT_PATH, 'randomize_network')
NETANALYZER = os.path.join(INPUT_PATH, 'netanalyzer')
RANKER = os.path.join(INPUT_PATH, 'ranker')
TEXT2BINARY_MATRIX = os.path.join(INPUT_PATH, 'text2binary_matrix')



@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("./tmp_output")
    return fn

def capture_stdout(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        tmpfile = StringIO()
        sys.stdout = tmpfile
        returned = func(*args, **kwargs)
        printed = sys.stdout.getvalue()
        sys.stdout = original_stdout
        return returned, printed
    return wrapper


def strng2table(strng, fs="\t", rs="\n"):
	table = [row.split(fs) for row in strng.split(rs)][0:-1]
	return table

def sort_table(table, sort_by, transposed=False):
	if transposed: 
		table = list(map(list, zip(*table)))
		table = sorted(table, key= lambda row: row[sort_by])
		table = list(map(list, zip(*table)))
	else:
		table = sorted(table, key= lambda row: row[sort_by])
	return table

@capture_stdout
def randomize_clustering(lsargs):
     return NetAnalyzer.randomize_clustering(lsargs)

@capture_stdout
def randomize_network(lsargs):
     return NetAnalyzer.randomize_network(lsargs)

@capture_stdout
def netanalyzer(lsargs):
    return NetAnalyzer.netanalyzer(lsargs)

@capture_stdout
def netanalyzer_dsl(lsargs):
    with pytest.raises(SystemExit):
        return NetAnalyzer.netanalyzer(lsargs)

def compare_exec(ref_file, output_file, args, dsl = False, matrix = False, roundTo=3):
    if dsl:
        _, printed = netanalyzer_dsl(args)
    else:
        _, printed = netanalyzer(args)
    #print(printed)
    diff(ref_file, output_file, matrix, roundTo)

def diff(file1, file2, matrix=False, roundTo=3):
    if matrix:
        expected_result = np.round(np.load(file1),roundTo)
        test_result = np.round(np.load(file2),roundTo)
        assert (expected_result == test_result).all()
    else:
        expected_result = CmdTabs.load_input_data(file1)
        test_result = CmdTabs.load_input_data(file2)
        print(expected_result)
        print(test_result)
        assert expected_result == test_result

# Netanalyzer bin.

def test_projections(tmp_dir):
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(DATA_PATH, "jaccard_results.txt")
    output_file = os.path.join(tmp_dir, "jaccard_results.txt")
    args = f"-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -m jaccard -u gen;pathway".split(" ")
    compare_exec(ref_file, output_file, args)

    ref_file = os.path.join(DATA_PATH, "counts_results.txt")
    output_file = os.path.join(tmp_dir, "counts_results.txt")
    args = f"-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -m counts -u gen;pathway".split(" ")
    compare_exec(ref_file, output_file, args)

    input_file = os.path.join(DATA_PATH, "tripartite_network_for_validating.txt")
    ref_file = os.path.join(DATA_PATH, "transference_results.txt")
    output_file = os.path.join(tmp_dir, "transference_results.txt")
    args = f"-i {input_file} -a {output_file} -f pair -lgen,M[0-9]+;pathway,P[0-9]+;salient,S[0-9]+ -m transference -u gen,salient;pathway".split(" ")
    compare_exec(ref_file, output_file, args)

    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "dsl", "jaccard_results.txt")
    output_file = os.path.join(NETANALYZER, "dsl","output", "jaccard_results.txt")
    dsl = os.path.join(NETANALYZER, "dsl","jaccard_dsl")
    args = f"-i {input_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ --dsl_script {dsl}".split(" ")
    compare_exec(ref_file, output_file, args, dsl=True)

def test_get_node_graph_attrs(tmp_dir):

    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "attributes", "node_attributes_summ.txt")
    output_file = os.path.join(tmp_dir, "jaccard_results.txt")
    args = f"-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -A get_degree,get_degreeZ --attributes_summarize".split(" ")
    output_file = os.path.join(tmp_dir, "node_attributes.txt")
    netanalyzer(args)
    shutil.move("node_attributes.txt", output_file)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = CmdTabs.load_input_data(ref_file)
    assert expected_result == test_result

    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "attributes", "node_attributes_nonsumm.txt")
    output_file = os.path.join(tmp_dir, "jaccard_results.txt")
    args = f"-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -A get_degree,get_degreeZ".split(" ")
    output_file = os.path.join(tmp_dir, "node_attributes.txt")
    netanalyzer(args)
    shutil.move("node_attributes.txt", output_file)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = CmdTabs.load_input_data(ref_file)
    assert expected_result == test_result

    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "attributes", "graph_attributes.txt")
    output_file = os.path.join(tmp_dir, "jaccard_results.txt")
    #netanalyzer -i $data_to_test/bipartite_network_for_validating.txt -a $out/projections/jaccard_results.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --graph_attributes "size,edge_density"
    args = f"-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ --graph_attributes size,edge_density".split(" ")
    output_file = os.path.join(tmp_dir, "graph_attributes.txt")
    netanalyzer(args)
    shutil.move("graph_attributes.txt", output_file)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = CmdTabs.load_input_data(ref_file)
    assert expected_result == test_result


def test_get_kernels(tmp_dir):
    input_file = os.path.join(DATA_PATH, "data_kernel", "adj_mat")
    ref_file = os.path.join(DATA_PATH, "data_kernel", "ct.npy")
    output_file = os.path.join(tmp_dir, "ct.npy")
    args = f"-i {input_file}.npy -f bin -l genes -K {output_file} -n {input_file}.lst -u genes -k ct".split(" ")
    netanalyzer(args)
    diff(output_file, ref_file, matrix=True)

    ref_file = os.path.join(DATA_PATH, "data_kernel", "ka_normalized.npy")
    output_file = os.path.join(tmp_dir, "ka_normalized.npy")
    args = f"-i {input_file}.npy -f bin -l genes -K {output_file} -n {input_file}.lst -u genes -k ka -z".split(" ")
    netanalyzer(args)
    diff(output_file, ref_file, matrix=True)

    ref_file = os.path.join(NETANALYZER, "dsl", "ct.npy")
    output_file = os.path.join(NETANALYZER, "dsl","output", "ct.npy")
    dsl = os.path.join(NETANALYZER, "dsl","kernel_dsl")
    args = f"-i {input_file}.npy -f bin -l genes -n {input_file}.lst --dsl_script {dsl}".split(" ")
    netanalyzer_dsl(args)
    diff(output_file, ref_file, matrix=True)

def test_filtering(tmp_dir):
    # netanalyzer -i {input_file} -f pair -l main --dsl_script {dsl}
    input_file = os.path.join(DATA_PATH, "monopartite_network_weights_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "dsl", "filter_cutoff")
    output_file = os.path.join(NETANALYZER, "dsl","output", "filter_cutoff")
    dsl = os.path.join(NETANALYZER, "dsl", "filter_dsl")
    args = f"-i {input_file} -f pair -l main --dsl_script {dsl}".split(" ")
    netanalyzer_dsl(args)
    diff(ref_file, output_file)

    #netanalyzer -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --dsl_script $data_test_scripts/dsl/jaccard_count_filter_dsl
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "dsl", "filter_with_count")
    output_file = os.path.join(NETANALYZER, "dsl","output", "filter_with_count")
    dsl = os.path.join(NETANALYZER, "dsl", "jaccard_count_filter_dsl")
    args = f"-i {input_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ --dsl_script {dsl}".split(" ")
    _, printed = netanalyzer_dsl(args)
    print(printed)
    for tag in ["_colIds", "_rowIds"]:
        diff(ref_file+tag, output_file+tag, matrix=False)
    tag = ".npy"
    diff(ref_file+".npy", output_file+".npy", matrix=True, roundTo=4)

def test_communities(tmp_dir):
    # Create communities
    input_file = os.path.join(DATA_PATH, "counts_results.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "rber_pots_discovered_clusters.txt")
    output_dir = os.path.join(tmp_dir)
    args=f"-i {input_file} -f pair -o {output_dir} -l genes -b rber_pots --output_build_clusters {output_dir}/rber_pots_discovered_clusters.txt".split(" ")
    netanalyzer(args)
    diff(ref_file, f"{output_dir}/rber_pots_discovered_clusters.txt")

    ref_file = os.path.join(NETANALYZER, "clustering", "der_discovered_clusters.txt")
    output_dir = os.path.join(tmp_dir)
    args=f"-i {input_file} -f pair -o {output_dir} -l genes -b der --seed 1 --output_build_clusters {output_dir}/der_discovered_clusters.txt".split(" ")
    netanalyzer(args)
    diff(ref_file, f"{output_dir}/der_discovered_clusters.txt")

    # Metrics
    ## Summ
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    group_file = os.path.join(NETANALYZER, "clustering", "clusters_toy.txt")
    output_file = os.path.join(tmp_dir, "group_metrics_summarized.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "group_metrics_summarized.txt")
    args=f"-i {input_file} -G {group_file} --output_summarized_metrics {output_file} -f pair -l genes -S max_odf;avg_transitivity;conductance".split(" ")
    netanalyzer(args)
    diff(ref_file, output_file)
    ## Not Summ
    output_file = os.path.join(tmp_dir, "group_metrics.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "group_metrics.txt")
    args=f"-i {input_file} -G {group_file} --output_metrics_by_cluster {output_file} -f pair -l genes -M comparative_degree;max_odf".split(" ")
    netanalyzer(args)
    diff(ref_file, output_file)
    ## Sum and Not summ
    output_file_summ = os.path.join(tmp_dir, "group_metrics_summarized.txt")
    output_file_not_summ = os.path.join(tmp_dir, "group_metrics.txt")
    ref_file_summ = os.path.join(NETANALYZER, "clustering", "group_metrics.txt")
    ref_file_not_summ = os.path.join(NETANALYZER, "clustering", "group_metrics_summarized.txt")
    args=f"-i {input_file} -G {group_file} -f pair -l genes --output_metrics_by_cluster {output_file_not_summ} -M comparative_degree;max_odf --output_summarized_metrics {output_file_summ} -S max_odf;avg_transitivity;conductance".split(" ")
    netanalyzer(args)
    diff(ref_file, output_file)
    ### Not Summ Not connected
    #netanalyzer -i $data_test_scripts/clustering/non_connected_network.txt -G $data_test_scripts/clustering/clusters_toy.txt --output_metrics_by_cluster ./$out/clustering/group_metrics_non_connected.txt -f pair -l 'genes' -M 'comparative_degree;max_odf;avg_sht_path'
    input_file = os.path.join(DATA_PATH, "non_connected_network.txt")
    output_file = os.path.join(tmp_dir, "group_metrics.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "group_metrics_non_connected.txt")
    args=f"-i {input_file} -G {group_file} --output_metrics_by_cluster {output_file} -f pair -l genes -M comparative_degree;max_odf;avg_sht_path".split(" ")
    netanalyzer(args)
    diff(ref_file, output_file)

    # Comparing group families
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    group_reference_file = os.path.join(NETANALYZER, "clustering","rber_pots_discovered_clusters.txt")
    args=f"-i {input_file} -G {group_file} -R {group_reference_file} -f pair -l genes -M comparative_degree;max_odf;avg_sht_path".split(" ")
    _, printed = netanalyzer(args)
    assert 0.348 == round(float(printed.split("\n")[-2]),3)

    # Group expansion
    output_file = os.path.join(tmp_dir, "expand_clusters.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "expand_clusters.txt")
    args=f"-i {input_file} -G {group_file} --output_expand_clusters {output_file} -f pair -l genes -x sht_path".split(" ")
    netanalyzer(args)   
    expected_result = CmdTabs.load_input_data(ref_file)
    test_result = CmdTabs.load_input_data(output_file)
    test_result = sort_table(sort_table(test_result,0),1)
    expected_result = sort_table(sort_table(expected_result,0),1)
    assert expected_result == test_result

def test_randomize(tmp_dir):
    # Randomize clustering
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    output_file = os.path.join(tmp_dir, "random_clusters.txt")
    ref_file = os.path.join(RANDOMIZE_CLUSTERING, "random_clusters.txt")
    args=f"-i {input_file} -o {output_file} -r fixed:10:3".split(" ")
    _, printed = randomize_clustering(args)
    diff(ref_file, output_file)

    # Randomize network
    input_file = os.path.join(DATA_PATH, "monopartite_network_for_validating.txt")
    output_file = os.path.join(tmp_dir, "random_net_same_seed.txt")
    ref_file = os.path.join(RANDOMIZE_NETWORK, "random_net_same_seed.txt")
    args=f"-i {input_file} -o {output_file} -f pair -l nodes,[A-Z] -r links --seed 1".split(" ")
    randomize_network(args)
    diff(ref_file, output_file)


@capture_stdout
def text2binary_matrix(lsargs):
     return NetAnalyzer.text2binary_matrix(lsargs)

# def test_text2binary_matrix(tmp_dir):
#     input_file = os.path.join(DATA_PATH, "test_matrix_bin.npy")
#     output_file = os.path.join(tmp_dir, "output_text2binary")

#     ref_name = ["statistics_from_text2bin", "cutoff_no_binarizado", "cutoff_binarizado", "set_diagonal_matrix", "test_matrix_bin.npy"]
#     args_basic="-i {input_file} -t bin -o {output_file}".split(" ")
#     args_specific = ["-c 0.5".split(" "), "-B 0.5".split(" "), ["-d"], "-O bin".split(" ")]
#     for i in range(0, len(ref_name)):
#         args= args_basic + args_specific[i]
#         ref_file = os.path.join(TEXT2BINARY_MATRIX, ref_name[i])
#         text2binary_matrix(args)
#         diff(output_file, ref_file)
         


# -i $data_to_test/test_matrix -t "matrix" -O "mat" -o $out/test_matrixfrommatrix
# -i $data_to_test/test_pairs -t "pair" -O "mat" -o $out/matrix_from_pairs

