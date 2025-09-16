import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from adapt.instance_based import TrAdaBoostR2, TwoStageTrAdaBoostR2
from sklearn.metrics import mean_squared_error, mean_absolute_error

import itertools
import os
from sklearn.preprocessing import StandardScaler

from adapt.instance_based import TrAdaBoostR2

from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings("ignore")

target_columns = ['Volume', 'Dgv']
test_size_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  #0.8or 0.93

predictor_columns = [
    'pzabovezmean', 'pzabove2', 'zq5', 'zq10', 'zq15', 'zq20', 'zq25', 'zq30',
    'zq35', 'zq40', 'zq45', 'zq50', 'zq55', 'zq60', 'zq65', 'zq70', 'zq75',
    'zq80', 'zq85', 'zq90', 'zq95', 'zpcum1', 'zpcum2', 'zpcum3', 'zpcum4',
    'zpcum5', 'zpcum6', 'zpcum7', 'zpcum8', 'zpcum9'
]

#ablation study for TradaBoostR2, Gaussian errors, with gaussian source domain errors
ablation_transfer_tradaboost_normal_normal = pd.DataFrame(columns=[
    'seed', 'target_column', 'target_instances', 'method', 'n_estimators',
    'lr', 'tree_size', 'val_rmse', 'val_mae', 'rmse', 'mae'
])

n_estimators_list = [10, 20, 30, 40, 50]
lr_list = [0.1, 0.5, 1.0]
tree_size_list = [1, 2, 3, 4, 5]

# --- Step 2: Create full parameter grid ---
param_grid = list(itertools.product(n_estimators_list, lr_list,
                                    tree_size_list))


## starta en screen för varje seed in [0,1,2,3,4] och en för varje test_size_index i [0,1,2,3,4,5] och kör!!
def train_run_tester_trada(s_idx, t_idx):
    seed_list = [1, 2, 3, 4, 5]
    test_size_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  #0.8or 0.93
    seed_list = [seed_list[int(s_idx)]]
    test_size_list = [test_size_list[int(t_idx)]]
    for seed in seed_list:
        for test_size in test_size_list:
            for target_column in target_columns:

                #data from Svedala
                data_sweden = pd.read_csv(r'datasets/rs_sweden.csv',
                                          index_col=[0])

                #evaluate and rain on latvia instead (keep naming for simplicity)
                #data from latvia target
                data_latvia = pd.read_csv(r'datasets/rs_lettland.csv',
                                          index_col=[0])
                train_size = int((1 - test_size) * len(data_latvia))
                data_latvia = data_latvia.rename(columns={
                    'H_AVERAGE': 'Hgv',
                    'D_AVERAGE': 'Dgv',
                    'VOLUME': 'Volume'
                })
                data_train, data_temp = train_test_split(data_latvia,
                                                         test_size=test_size,
                                                         random_state=seed)
                data_val, data_test = train_test_split(data_temp,
                                                       test_size=0.5,
                                                       random_state=seed)

                #"General" base dataset (to use for transfer)
                X_source_train = np.array(data_sweden[predictor_columns])
                y_source_train = np.array(data_sweden[target_column])

                #Specific train and test set
                X_target_train = np.array(data_train[predictor_columns])
                y_target_train = np.array(data_train[target_column])

                X_target_val = np.array(data_val[predictor_columns])
                y_target_val = np.array(data_val[target_column])

                X_target_test = np.array(data_test[predictor_columns])
                y_target_test = np.array(data_test[target_column])

                #print(len(X_target_train), len(X_target_val),
                #      len(X_target_test))
                for config in param_grid:
                    n_estimators, lr, tree_size = config

                    method = f'TradaBoostR2'
                    base_estimator = LinearTreeRegressor(
                        base_estimator=LinearRegression(),
                        max_depth=tree_size,
                        min_samples_leaf=4)
                    model = TrAdaBoostR2(base_estimator,
                                         n_estimators=n_estimators,
                                         lr=lr)
                    model.fit(X_source_train, y_source_train, X_target_train,
                              y_target_train)
                    preds = model.predict(X_target_test)
                    val_preds = model.predict(X_target_val)
                    val_rmse = np.sqrt(
                        mean_squared_error(val_preds, y_target_val))
                    val_mae = mean_absolute_error(val_preds, y_target_val)
                    rmse = np.sqrt(mean_squared_error(preds, y_target_test))
                    mae = mean_absolute_error(preds, y_target_test)
                    ablation_transfer_tradaboost_normal_normal = pd.DataFrame(
                        columns=[
                            'seed', 'target_column', 'target_instances',
                            'method', 'n_estimators', 'lr', 'tree_size',
                            'val_rmse', 'val_mae', 'rmse', 'mae'
                        ])
                    ablation_transfer_tradaboost_normal_normal.loc[len(
                        ablation_transfer_tradaboost_normal_normal)] = [
                            seed, target_column, train_size, method,
                            n_estimators, lr, tree_size, val_rmse, val_mae,
                            rmse, mae
                        ]
                    ablation_file = f'results/tradaboost_ablation_rs.csv'
                    file_exists = os.path.isfile(ablation_file)
                    ablation_transfer_tradaboost_normal_normal.to_csv(
                        ablation_file, mode='a', header=not file_exists)


#train_run_tester_trada(1)
