import itertools

seed_list = list(range(860,890)) #10 seeds for uci

#LSTransferTreeBoost configs
v_list = [0.1]
source_tree_size_list = [1,2,3]
target_tree_size_list = [1,2,3]
k_list = [0, 0.01, 0.05]
m_0_list = [0.1, 0.5, 0.9]
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
    if not (params[3] == 0.0 and params[4] == 0.9) #filter out these bad ones
]

#XGBoost configs
v_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
target_tree_size_list = [1, 2, 3, 4, 5, 6]

param_grid_XGBoost = list(itertools.product(v_list, target_tree_size_list))

#Two-stage TrAdaBoost.R2 configs
lr_list = [0.1, 0.5, 1.0]
n_estimators_list = [10, 20, 30]
n_estimators_fs_list = [10, 20, 30]
cv_list = [5, 10]

param_grid_TwoTrada = list(itertools.product(lr_list, 
                                             n_estimators_list, 
                                             n_estimators_fs_list,
                                             cv_list))

predictor_columns = [str(i) for i in range(3, 26)]
target_column = 'estimated_volume'
