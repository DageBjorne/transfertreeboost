import itertools


#Global configs for experiments
test_size = 1000
val_size = 1000

target_instances_list = [100,200,300]
source_instances = 1000 #we use a fixed number of source instances

d_list = [1,2,3,4,5,6,7,8,9,10]
seed_list = [1,2,3,4,5]


#LSTransferTreeBoost configs
v_list = [0.05, 0.1]
source_tree_size_list = [1, 2]
target_tree_size_list = [1, 2]
k_list = [0.01, 0.05]
m_0_list = [0.5, 0.9]

# Create full parameter grid ---
param_grid_LSTransferTreeBoost = list(
    itertools.product(v_list, source_tree_size_list, target_tree_size_list,
                      k_list, m_0_list))

#XGBoost and Naïve XGBoost configs
v_list = [0.01, 0.02, 0.05, 0.1, 0.15]
target_tree_size_list = [1, 2, 3, 4]

# Create full parameter grid ---
param_grid_XGBoost = list(itertools.product(v_list, target_tree_size_list))

#MLP configs
fine_tuning_lrs = [1e-4, 5e-5]
base_lrs = [5e-4, 1e-4]
dropout_list = [0.0, 0.1]
include_batch_norm = [True, False]

param_grid_MLP = list(itertools.product(fine_tuning_lrs, base_lrs,
                                        dropout_list, include_batch_norm))

#TradaBoost.R2 configs
n_estimators_list = [10,20,30,40,50]
lr_list = [0.1, 0.5, 1.0]
tree_size_list = [1,2,3,4,5]


# --- Step 2: Create full parameter grid ---
param_grid_tradaboost = list(itertools.product(
    n_estimators_list,
    lr_list,
    tree_size_list
))

