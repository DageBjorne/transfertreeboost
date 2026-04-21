import itertools

seed_list = list(range(860,890)) #10 seeds for uci

#LSTransferTreeBoost configs
v_list = [0.05, 0.1, 0.2, 0.3]
source_tree_size_list = [1,2,3]
target_tree_size_list = [1,2,3]
k_list = [0.01, 0.0]
m_0_list = [0.9, 0.5]
epoch_list = [400]

# Create full parameter grid ---
param_grid_TransferTreeBoost = [
    params for params in itertools.product(
        v_list,
        source_tree_size_list,
        target_tree_size_list,
        k_list,
        m_0_list,
        epoch_list
    )
    if (params[3] == 0.0 and params[4] == 0.5) or #filter out here
       (params[3] == 0.01 and params[4] == 0.9)
]

#XGBoost configs
v_list = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
target_tree_size_list = [1, 2, 3, 4, 5, 6, 7]
param_grid_XGBoost = list(itertools.product(v_list, target_tree_size_list))

#Two-stage TrAdaBoost.R2 configs
lr_list = [0.1, 0.5, 1.0]
n_estimators_list = [10, 20, 30, 50, 100, 150]
tree_size_list = [1,2,3,4]
# n_estimators_fs_list = [30]
# cv_list = [10]

param_grid_TwoTrada = list(itertools.product(lr_list, 
                                             n_estimators_list,
                                             tree_size_list))

learning_rate_list = [1e-3, 5e-4, 1e-4]# , 5e-5, 1e-5] #5
dropout_list = [0.0, 0.1] #[0.0, 0.1, 0.2] #15
d_main_list = [64, 256]#, 128]  #45
num_blocks_list = [2, 4]#, 3] #135

param_grid_ResNet = list(itertools.product(learning_rate_list, 
                                           dropout_list,
                                           d_main_list,
                                           num_blocks_list))


predictor_columns = [str(i) for i in range(3, 26)]
target_column = 'estimated_volume'
