import itertools

### RS Data

#Global configs

train_size_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
target_columns = ['Hgv']
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
d_list = [1, 2, 3, 4]

predictor_columns = [
    'pzabovezmean', 'pzabove2', 'zq5', 'zq10', 'zq15', 'zq20', 'zq25', 'zq30',
    'zq35', 'zq40', 'zq45', 'zq50', 'zq55', 'zq60', 'zq65', 'zq70', 'zq75',
    'zq80', 'zq85', 'zq90', 'zq95', 'zpcum1', 'zpcum2', 'zpcum3', 'zpcum4',
    'zpcum5', 'zpcum6', 'zpcum7', 'zpcum8', 'zpcum9'
]

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

v_list = [0.01, 0.02, 0.05, 0.1, 0.15]
target_tree_size_list = [1, 2, 3, 4]

# --- Step 2: Create full parameter grid ---
param_grid_XGBoost = list(itertools.product(v_list, target_tree_size_list))
