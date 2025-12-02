import itertools

#Global configs for experiments
test_size = 1000
val_size = 1000

target_instances_list = [100, 300, 500]
source_instances = 1000  #we use a fixed number of source instances

d_list = [3, 6, 9]
seed_list = list(range(85,90))

#LSTransferTreeBoost configs
v_list = [0.05, 0.1]
source_tree_size_list = [1, 2, 3, 4, 5]
target_tree_size_list = [1, 2, 3, 4, 5]
k_list = [0.0, 0.01, 0.05]
m_0_list = [0.1, 0.5, 0.9]
epoch_list = [400]

# Create full parameter grid ---
param_grid_LSTransferTreeBoost = list(
    itertools.product(v_list, source_tree_size_list, target_tree_size_list,
                      k_list, m_0_list, epoch_list))

