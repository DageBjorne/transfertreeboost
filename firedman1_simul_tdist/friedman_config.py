import itertools

#Global configs for experiments
test_size = 1000
val_size = 1000

target_instances_list = [100, 200, 300]
source_instances = 1000  #we use a fixed number of source instances

d_list = [6]
seed_list = list(range(50,70))

#LSTransferTreeBoost configs
v_list = [0.05]
source_tree_size_list = [1]
target_tree_size_list = [1]
k_list = [0.01]
m_0_list = [0.9]
epoch_list = [200]

# Create full parameter grid ---
param_grid_LSTransferTreeBoost = list(
    itertools.product(v_list, source_tree_size_list, target_tree_size_list,
                      k_list, m_0_list, epoch_list))

