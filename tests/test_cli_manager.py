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
NET_EXPLORER = os.path.join(INPUT_PATH, 'net_explorer')


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory, request):
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

@capture_stdout
def text2binary_matrix(lsargs):
    return NetAnalyzer.text2binary_matrix(lsargs)

@capture_stdout
def integrate_kernels(lsargs):
    return NetAnalyzer.integrate_kernels(lsargs)

@capture_stdout
def ranker(lsargs):
    return NetAnalyzer.ranker(lsargs)

@capture_stdout
def randomize_clustering(lsargs):
    return NetAnalyzer.randomize_clustering(lsargs)

@capture_stdout
def randomize_network(lsargs):
    return NetAnalyzer.randomize_network(lsargs)

@capture_stdout
def net_explorer(lsargs):
    return NetAnalyzer.net_explorer(lsargs, True)

@capture_stdout
def netanalyzer(lsargs):
    return NetAnalyzer.netanalyzer(lsargs)

@capture_stdout
def netanalyzer_dsl(lsargs):
    with pytest.raises(SystemExit):
        return NetAnalyzer.netanalyzer(lsargs)

def diff(file1, file2, matrix=False, roundTo=3, sort = None):
    if matrix:
        expected_result = np.round(np.load(file1),roundTo)
        test_result = np.round(np.load(file2),roundTo)
        assert (expected_result == test_result).all()
    else:
        expected_result = CmdTabs.load_input_data(file1)
        if sort:
            for c in sort:
                expected_result.sort(key=lambda x: x[c])
        test_result = CmdTabs.load_input_data(file2)
        if sort:
            for c in sort:
                test_result.sort(key=lambda x: x[c])
        assert expected_result == test_result

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

def dsloutput2tmp(outputdsl, outputtmp):
    shutil.move(outputdsl,outputtmp)

# netanalyzer #
###############

@pytest.mark.parametrize("input_file, ref_file, output_file, args", [
    (   # Get projection from jaccard
        "bipartite_network_for_validating.txt", 
        "jaccard_results.txt", 
        "jaccard_results.txt", 
        "-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -m jaccard -u gen;pathway"
    ),
    (   # Get projection from counts
        "bipartite_network_for_validating.txt", 
        "counts_results.txt", "counts_results.txt", 
        "-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -m counts -u gen;pathway"
    ),
    (   # Select nodes to remove in network
        "bipartite_network_for_validating.txt", 
        "counts_results_with_deleted.txt", 
        "counts_results.txt", 
        "-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ -d {filter_node};d -m counts -u gen;pathway"
    ),
    (   # Get tripartite projection in network
        "tripartite_network_for_validating.txt", 
        "transference_results.txt", 
        "transference_results.txt", 
        "-i {input_file} -a {output_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+;salient,S[0-9]+ -m transference -u gen,salient;pathway"
    )
])
def test_projections(tmp_dir, input_file, ref_file, output_file, args):
    input_file = os.path.join(DATA_PATH, input_file)
    ref_file = os.path.join(DATA_PATH, ref_file)
    output_file = os.path.join(tmp_dir, output_file)
    filter_node = os.path.join(tmp_dir, "filter")
    CmdTabs.write_output_data(["M1","M2"], output_path=filter_node, sep="\t")

    args = args.format(input_file=input_file, output_file=output_file, filter_node=filter_node).split(" ")
    _, printed = netanalyzer(args)
    diff(ref_file, output_file)

def test_projections_dsl(tmp_dir):
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "dsl", "jaccard_results.txt")
    output_file = os.path.join(tmp_dir, "jaccard_results.txt")
    dsl = os.path.join(NETANALYZER, "dsl","jaccard_dsl")
    args = f"-i {input_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ --dsl_script {dsl}".split(" ")
    _, printed = netanalyzer_dsl(args)
    dsloutput2tmp("outpathfile",output_file)
    diff(ref_file, output_file)
    

@pytest.mark.parametrize("input_file, ref_file, output_file1, output_file2check, args", [
    (   # Node attribute summarized
        "bipartite_network_for_validating.txt",
        "node_attributes_summ.txt",
        "jaccard_results.txt",
        "node_attributes.txt",
        "-f pair -l gen,M[0-9]+;pathway,P[0-9]+ -A get_degree,get_degreeZ --attributes_summarize".split(" ")
    ),
    (   # Node attribute not summarized
        "bipartite_network_for_validating.txt",
        "node_attributes_nonsumm.txt",
        "jaccard_results.txt",
        "node_attributes.txt",
        "-f pair -l gen,M[0-9]+;pathway,P[0-9]+ -A get_degree,get_degreeZ".split(" ")
    ),
    (   # Graph attributes
        "bipartite_network_for_validating.txt",
        "graph_attributes.txt",
        "jaccard_results.txt",
        "graph_attributes.txt",
        "-f pair -l gen,M[0-9]+;pathway,P[0-9]+ --graph_attributes size,edge_density".split(" ")
    ),
])
def test_get_node_graph_attrs(tmp_dir, input_file, ref_file, output_file1, output_file2check, args):
    input_file = os.path.join(DATA_PATH, input_file)
    ref_file = os.path.join(NETANALYZER, "attributes", ref_file)
    output_file1 = os.path.join(tmp_dir,  output_file1)
    args_base = f"-i {input_file} -a {output_file1}".split(" ")
    args = args_base + args
    output_file_check = os.path.join(tmp_dir,  output_file2check)
    netanalyzer(args)
    shutil.move(output_file2check, output_file_check)
    test_result = CmdTabs.load_input_data(output_file_check)
    expected_result = CmdTabs.load_input_data(ref_file)
    assert expected_result == test_result

@pytest.mark.parametrize("ref_file, output_file2check,args, with_dsl", [
    (os.path.join(DATA_PATH, "data_kernel", "ct"), "ct", "-i {input_file}.npy -f bin -l genes -K {output_file} -n {input_file}.lst -u genes -k ct", False), # Get ct kernel
    (os.path.join(DATA_PATH, "data_kernel", "ka_normalized"), "ka_normalized", "-i {input_file}.npy -f bin -l genes -K {output_file} -n {input_file}.lst -u genes -k ka -z", False), # Cosine Normalized kernel
    (os.path.join(NETANALYZER, "dsl", "ct"), "ct", "-i {input_file}.npy -f bin -l genes -n {input_file}.lst --dsl_script {dsl}", True) # Using dsl
])
def test_get_kernels(tmp_dir, ref_file, output_file2check, args, with_dsl):
    input_file = os.path.join(DATA_PATH, "data_kernel", "adj_mat")
    dsl = os.path.join(NETANALYZER, "dsl","kernel_dsl")
    output_file = os.path.join(tmp_dir, output_file2check)
    args = args.format(input_file=input_file, output_file=output_file, dsl=dsl).split(" ")
    if with_dsl:
        netanalyzer_dsl(args)
        for tag in [".npy","_colIds","_rowIds"]:
            dsloutput2tmp("outpathfile"+tag,output_file+tag)
    else:
        netanalyzer(args)
    diff(output_file + ".npy", ref_file + ".npy", matrix=True)
    diff(output_file + "_colIds", ref_file + "_colIds")
    diff(output_file + "_rowIds", ref_file + "_rowIds")

def test_filtering(tmp_dir):
    # filter bycomponent
    input_file = os.path.join(DATA_PATH, "monopartite_network_weights_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "filter", "filter_by_ccomponent")
    output_file = os.path.join(tmp_dir, "filter_by_ccomponent")
    args = f"-i {input_file} -f pair -l main --filter_connected_components 4 --output_network {output_file}".split(" ")
    netanalyzer(args)
    diff(ref_file, output_file)

    # netanalyzer -i {input_file} -f pair -l main --dsl_script {dsl}
    input_file = os.path.join(DATA_PATH, "monopartite_network_weights_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "dsl", "filter_cutoff")
    output_file = os.path.join(tmp_dir, "filter_cutoff")
    dsl = os.path.join(NETANALYZER, "dsl", "filter_dsl")
    args = f"-i {input_file} -f pair -l main --dsl_script {dsl}".split(" ")
    netanalyzer_dsl(args)
    dsloutput2tmp("outpathfile",output_file)
    diff(ref_file, output_file)

    #netanalyzer -i $data_to_test/bipartite_network_for_validating.txt -f pair -l 'gen,M[0-9]+;pathway,P[0-9]+' --dsl_script $data_test_scripts/dsl/jaccard_count_filter_dsl
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    ref_file = os.path.join(NETANALYZER, "dsl", "filter_with_count")
    output_file = os.path.join(tmp_dir, "filter_with_count")
    dsl = os.path.join(NETANALYZER, "dsl", "jaccard_count_filter_dsl")
    args = f"-i {input_file} -f pair -l gen,M[0-9]+;pathway,P[0-9]+ --dsl_script {dsl}".split(" ")
    _, printed = netanalyzer_dsl(args)
    for tag in ["_colIds", "_rowIds"]:
        dsloutput2tmp("outpathfile"+tag, output_file+tag)
        diff(ref_file+tag, output_file+tag, matrix=False)
    tag = ".npy"
    dsloutput2tmp("outpathfile"+tag, output_file+tag)
    diff(ref_file+tag, output_file+tag, matrix=True, roundTo=4)

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

    ref_file = os.path.join(NETANALYZER, "clustering", "der_discovered_clusters_by_subgroup.txt")
    output_dir = os.path.join(tmp_dir)
    group_file = os.path.join(NETANALYZER, "clustering", "clusters_toy_subgroup.txt")
    args=f"-i {input_file} -f pair -o {output_dir} -l genes -b der -G {group_file} --seed 1 --output_build_clusters {output_dir}/der_discovered_clusters.txt".split(" ")
    netanalyzer(args)
    diff(ref_file, f"{output_dir}/der_discovered_clusters.txt")

    # Metrics
    ## Summ
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    group_file = os.path.join(NETANALYZER, "clustering", "clusters_toy.txt")
    output_file = os.path.join(tmp_dir, "group_metrics_summarized.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "group_metrics_summarized.txt")
    #--output_metrics_by_cluster
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
    input_file = os.path.join(DATA_PATH, "non_connected_network.txt")
    output_file = os.path.join(tmp_dir, "group_metrics.txt")
    ref_file = os.path.join(NETANALYZER, "clustering", "group_metrics_non_connected.txt")
    args=f"-i {input_file} -G {group_file} --output_metrics_by_cluster {output_file} -f pair -l genes -M comparative_degree;max_odf;avg_sht_path".split(" ")
    netanalyzer(args)
    diff(ref_file, output_file)
    # Comparing group families
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    group_reference_file = os.path.join(NETANALYZER, "clustering","rber_pots_discovered_clusters.txt")
    args=f"-i {input_file} -G {group_file} -R {group_reference_file} -f pair -l genes".split(" ")
    _, printed = netanalyzer(args)
    assert 0.647 == round(float(printed.split("\n")[-2].split("\t")[1]),3)
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

# randomize_clustering #
# randomize_network    #
########################

def test_randomize(tmp_dir):
    # Randomize clustering
    input_file = os.path.join(DATA_PATH, "bipartite_network_for_validating.txt")
    output_file = os.path.join(tmp_dir, "random_clusters.txt")
    ref_file = os.path.join(RANDOMIZE_CLUSTERING, "random_clusters.txt")
    args=f"-G {input_file} -o {output_file} -r custom:3:3:nr".split(" ")
    _, printed = randomize_clustering(args)
    diff(ref_file, output_file)

    input_file = os.path.join(DATA_PATH, "minicluster")
    output_file = os.path.join(tmp_dir, "random_clusters.txt")
    ref_file = os.path.join(RANDOMIZE_CLUSTERING, "random_minicluster.txt")
    args=f"-G {input_file} -o {output_file} -r hard_fixed --seed 111".split(" ")
    _, printed = randomize_clustering(args)
    print(printed)
    diff(ref_file, output_file)

    # Randomize network
    input_file = os.path.join(DATA_PATH, "monopartite_network_for_validating.txt")
    output_file = os.path.join(tmp_dir, "random_net_same_seed.txt")
    ref_file = os.path.join(RANDOMIZE_NETWORK, "random_net_same_seed.txt")
    args=f"-i {input_file} -o {output_file} -f pair -l nodes,[A-Z] -r links --seed 1".split(" ")
    randomize_network(args)
    diff(ref_file, output_file)

# text2binary_matrix #
######################

@pytest.mark.parametrize("ref_name, ref_output, args, matrix", [
    ('cutoff_no_binarizado', '{output_file}', '-i {input_file1} -t bin -o {output_file} -c 0.5', True), # cutoff no binarizado
    ('cutoff_binarizado', '{output_file}', '-i {input_file1} -t bin -o {output_file} -B 0.5', True), # cutoff binarizado
    ('set_diagonal_matrix', '{output_file}', '-i {input_file1} -t bin -o {output_file} -d', True), # setting diagonal to 1
    ('test_matrix_bin', '{output_file}', '-i {input_file1} -t bin -o {output_file} -O bin', True), # checking reading
    ('statistics_from_text2bin', '{statistics_file}', '-i {input_file1} -t bin -s {statistics_file} -o {output_file}', False), # extract stats from matrix
    ('test_matrixfrommatrix', '{output_file}', '-i {input_file2} -t matrix -O matrix -o {output_file}', False), # matrix with input in text matrix file
    ('matrix_from_pairs', '{output_file}', '-i {input_file3} -t pair -O matrix -o {output_file}', False) # Transform matrix from pairs
])
def test_text2binary_matrix(tmp_dir, ref_name, ref_output, args, matrix):
    input_file1 = os.path.join(TEXT2BINARY_MATRIX, "test_matrix_bin.npy")
    input_file2 = os.path.join(TEXT2BINARY_MATRIX, "test_matrix")
    input_file3 = os.path.join(TEXT2BINARY_MATRIX, "test_pairs")
    output_file = os.path.join(tmp_dir, "output_text2binary")
    statistics_file= os.path.join(tmp_dir, "statistics_from_text2bin")
    
    ref_file = os.path.join(TEXT2BINARY_MATRIX, ref_name)
    output_file = ref_output.format(output_file=output_file, statistics_file=statistics_file)
    args = args.format(output_file=output_file, statistics_file=statistics_file, input_file1=input_file1, input_file2=input_file2, input_file3=input_file3).split(" ")
    _, printed = text2binary_matrix(args)
    if matrix:
        tag = ".npy"
    else:
        tag = ""
    diff(output_file + tag, ref_file + tag, matrix=matrix)

# ranker #
##########

@pytest.mark.parametrize("ref_file, args, output2check, tag", [
       ('ranker_by_seed_string_results', '--seed_nodes A,B -k {kernel_file} -n {kernel_file}.lst -o {output_file}', '{output_file}', '_all_candidates'), # set seed from terminal
       ('ranker_by_seed_file_results', '--seed_nodes {seeds_file} -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # seed from file,
       ('ranker_by_seed_file_results_tagged', '--seed_nodes {seeds_file} -k {kernel_file} -n {kernel_file}.lst -o {output_file} --add_tags {tagged_file}','{output_file}', '_all_candidates'), # seed from file
       ('ranker_by_seed_file_nonseeded_results', '--seed_nodes {seeds_file} --seed_presence remove -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # seed from file
       ('ranker_by_seed_file_results_header', '--seed_nodes {seeds_file} -k {kernel_file} -n {kernel_file}.lst -o {output_file} --header','{output_file}', '_all_candidates'), # seed from file
       ('ranker_by_seed_file_weighted_results', '--seed_nodes {seeds_file_weighted} -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # probando
       ('ranker_by_seed_file_results_type_added', '--seed_nodes {seeds_file} --seed_presence annotate -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # if candidates are in seeds or not
       ('ranker_by_seed_file_results_type_added_header', '--seed_nodes {seeds_file} --seed_presence annotate -k {kernel_file} -n {kernel_file}.lst -o {output_file} --header','{output_file}', '_all_candidates'), # seed from file
       ('ranker_leave_one_out_by_seed_results_bigseed', '--seed_nodes {bigseed} -l -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # leave one out option
       ('ranker_leave_one_out_by_seed_results', '--seed_nodes {seeds_file} -l -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # leave one out option
       ('ranker_leave_one_out_by_seed_filter_results', '--seed_nodes {seeds_file} -l -k {kernel_file} -f {filter_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # leave one out option with filter
       ('ranker_leave_one_out_by_seed_nonseed_results', '--seed_nodes {seeds_file} -l --seed_presence remove -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # leave one out option
       ('ranker_leave_one_out_by_seed_results_header', '--seed_nodes {seeds_file} -l -k {kernel_file} -n {kernel_file}.lst -o {output_file} --header','{output_file}', '_all_candidates'), # leave one out option
       ('ranker_cross_validation_by_seed_results', '--seed_nodes {seeds_file} -l -K 2 -k {kernel_file} -f {filter_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Cross validation option
       ('ranker_cross_validation_all_by_seed_results', '--seed_nodes {seeds_file} -l -K 2 -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Cross validation option
       ('ranker_cross_validation_by_seed_results_bigseed', '--seed_nodes {bigseed} -l -K 3 -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Cross validation option
       ('ranker_cross_validation_by_seed_results_remove_seed', '--seed_nodes {bigseed} -l -K 3 -k {kernel_file} --seed_presence remove -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Cross validation option
       ('ranker_filter_results', '--seed_nodes {seeds_file} -f {filter_file} -o {output_file} -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Filter ranking to select genes to keep in output
       ('ranker_whitelist_results', '--seed_nodes {seeds_file} --whitelist {whitelist} -o {output_file} -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Whitelist to filter in matrix
       ('ranker_propagate_normalized', '--seed_nodes {seeds_file} -p -N by_column -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Propagate with normalization by column
       ('ranker_propagate_not_normalized', '--seed_nodes {seeds_file} -p -k {kernel_file} -n {kernel_file}.lst -o {output_file}','{output_file}', '_all_candidates'), # Propagate with no normalization
       ('ranker_propagate_with_restart', "--seed_nodes {seeds_file} -p --propagate_options 'tolerance':1e-5,'iteration_limit':100,'with_restart':0.7 -k {kernel_file} -n {kernel_file}.lst -o {output_file}", '{output_file}', '_all_candidates'), # Propagate with restart
       ('ranker_top_results', "--seed_nodes {seeds_file} --output_top {top_output} -t 2 -o {output_file} -k {kernel_file} -n {kernel_file}.lst", '{top_output}', ''), # Select top results
       ('output_ranker', "--seed_nodes JK,LL -o {output_file} -k {kernel_file} -n {kernel_file}.lst", '{output_file}', '_discarded') # Check the discarded seeds
    ])
def test_ranker(tmp_dir, ref_file, args, output2check, tag):
    kernel_file = os.path.join(DATA_PATH, 'data_ranker', 'kernel_for_validating')
    output_file = os.path.join(tmp_dir, "output_ranker")
    seeds_file= os.path.join(DATA_PATH, 'data_ranker', 'seed_genes_for_validating_withNotInkernels')
    tagged_file = os.path.join(DATA_PATH, 'data_ranker', 'tagged_file')
    bigseed = os.path.join(DATA_PATH, 'data_ranker', "bigseed")
    seeds_file_weighted = os.path.join(DATA_PATH, 'data_ranker', 'seed_weighted_for_validating')
    filter_file = os.path.join(DATA_PATH, 'data_ranker', 'genes2filter_for_validating')
    whitelist = os.path.join(RANKER, "whitelist")
    top_output = os.path.join(tmp_dir, "ranker_top_results")
    args = args.format(kernel_file=kernel_file, output_file=output_file, seeds_file=seeds_file, filter_file=filter_file, bigseed=bigseed,whitelist=whitelist, top_output=top_output,seeds_file_weighted=seeds_file_weighted, tagged_file=tagged_file).split(" ")
    ref_file = os.path.join(RANKER, ref_file)
    _, printed = ranker(args)
    print(printed)
    diff(output2check.format(output_file=output_file,top_output=top_output)+tag, ref_file+tag, sort=[0,1])


# integration #
###############

@pytest.mark.parametrize("ref_file, output_file, args", [
    ("int_mean", "outputfile_integration", "-i mean -k {kernel1}.npy;{kernel2}.npy -n {kernel1}.lst;{kernel2}.lst -o {output_file}"), # Integration for symmetric matrixes
    ("int_mean_asym", "outputfile_integration", "--asym -i mean -k {asym_kernel1}.npy;{asym_kernel2}.npy -n {kernel1}.lst;{kernel2}.lst -o {output_file}") # INtegration for non symmetric matrixes
])
def test_integration(tmp_dir, ref_file, output_file, args):
    output_file = os.path.join(tmp_dir, output_file)
    ref_file = os.path.join(INTEGRATE_KERNELS, ref_file)
    kernel1 = os.path.join(INTEGRATE_KERNELS, "kernel1")
    kernel2 = os.path.join(INTEGRATE_KERNELS, "kernel2")
    asym_kernel1 = os.path.join(INTEGRATE_KERNELS, "asym_kernel1")
    asym_kernel2 = os.path.join(INTEGRATE_KERNELS, "asym_kernel2")
    args = args.format(ref_file=ref_file, output_file=output_file,  kernel1=kernel1,  kernel2=kernel2, asym_kernel1= asym_kernel1, asym_kernel2 = asym_kernel2).split(" ")
    _, printed = integrate_kernels(args)
    diff(ref_file + ".npy", output_file + ".npy", matrix=True)
    diff(ref_file + ".lst", output_file + ".lst")


# Net explorer #
################

@pytest.mark.parametrize("args", [
    ( "-i graphA,{net1}.npy;graphB,{net2}.npy -n graphA,{net1}.lst;graphB,{net2}.lst -N --neigh_level layerA,1;layerB,2 --seed_nodes {seeds} -o {output_file} --embedding_proj umap --plot_network_method pyvis --compare_nets")
])
def test_net_explorer(tmp_dir, args):
    output_file = os.path.join(tmp_dir, "output_file")
    net1 = os.path.join(NET_EXPLORER, "net1")
    net2 = os.path.join(NET_EXPLORER, "net2")
    seeds = os.path.join(NET_EXPLORER, "seeds")
    args = args.format(output_file=output_file, net1=net1, net2=net2, seeds=seeds).split(" ")
    returned, printed = net_explorer(args)
    assert {'seed': ['A', 'B', 'C']} == returned["seeds2explore"] 
    returned_subgraph_A = returned["seeds2subgraph"]["seed"]["graphA"]
    returned_subgraph_B = returned["seeds2subgraph"]["seed"]["graphB"]
    assert list(returned_subgraph_B.nodes) == ['C', 'A']
    assert list(returned_subgraph_A.nodes) == ['A', 'B', 'C']
    assert list(returned_subgraph_B.edges) == []
    assert list(returned_subgraph_A.edges) == [('A', 'C')]
    returned_embedding_proj_A = returned["net2embedding_proj"]["graphA"]
    returned_embedding_proj_B = returned["net2embedding_proj"]["graphB"]
    assert returned_embedding_proj_A[0].shape == (5, 2)
    assert returned_embedding_proj_A[1] == ['A', 'B', 'C', 'D', 'E']
    assert returned_embedding_proj_B[0].shape == (4, 2)
    assert returned_embedding_proj_B[1] == ['C', 'E', 'D', 'A']
