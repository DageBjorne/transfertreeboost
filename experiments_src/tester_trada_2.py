import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import experiments.config as c
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

target_columns = ['Hgv']
#test_size_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  #0.8or 0.93

predictor_columns = c.predictor_columns

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


## starta en screen för varje seed in [0,1,2,3,4] och en för varje test_size_index i [0,1,2,3,4,5,6,7,8,9] och kör!!
def train_run_tester_trada(s_idx):
    seed_list = c.seed_list  #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for d in c.d_list:
        seed_list = [seed_list[int(s_idx)]]
        for seed in seed_list:
            for train_size in c.train_size_list:
                for target_column in target_columns:

                    #data from Svedala
                    #data from Svedala
                    data_sweden = pd.read_csv(r'datasets/rs_sweden.csv',
                                              index_col=[0])
                    data_sweden = data_sweden[data_sweden['area_code'] == d]
                    print(len(data_sweden))
                    data_sweden = data_sweden.sample(1000, random_state=seed)
                    #evaluate and rain on latvia instead (keep naming for simplicity)
                    #data from latvia target
                    data_latvia = pd.read_csv(r'datasets/rs_lettland.csv',
                                              index_col=[0])
                    data_latvia = data_latvia.rename(columns={
                        'H_AVERAGE': 'Hgv',
                        'D_AVERAGE': 'Dgv',
                        'VOLUME': 'Volume'
                    })
                    data_temp, data_test = train_test_split(data_latvia,
                                                            test_size=0.25,
                                                            random_state=seed)
                    data_train, data_val = train_test_split(data_temp,
                                                            test_size=0.333,
                                                            random_state=seed)
                    train_size_ = int(len(data_train) * train_size)
                    data_train = data_train[0:train_size_]

                    #"General" base dataset (to use for transfer)
                    X_source_train = np.array(data_sweden[c.predictor_columns])
                    #y_source_train = np.array(data_sweden[target_column])
                    y_source_train = np.array(data_sweden[target_column])

                    #Specific train and test set
                    X_target_train = np.array(data_train[c.predictor_columns])
                    y_target_train = np.array(data_train[target_column])

                    X_target_val = np.array(data_val[c.predictor_columns])
                    y_target_val = np.array(data_val[target_column])

                    X_target_test = np.array(data_test[c.predictor_columns])
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
                        model.fit(X_source_train, y_source_train,
                                  X_target_train, y_target_train)
                        preds = model.predict(X_target_test)
                        val_preds = model.predict(X_target_val)
                        val_rmse = np.sqrt(
                            mean_squared_error(val_preds, y_target_val))
                        val_mae = mean_absolute_error(val_preds, y_target_val)
                        rmse = np.sqrt(mean_squared_error(
                            preds, y_target_test))
                        mae = mean_absolute_error(preds, y_target_test)
                        ablation_transfer_tradaboost_normal_normal = pd.DataFrame(
                            columns=[
                                'seed', 'target_column', 'target_instances',
                                'method', 'n_estimators', 'lr', 'tree_size',
                                'val_rmse', 'val_mae', 'rmse', 'mae'
                            ])
                        ablation_transfer_tradaboost_normal_normal.loc[len(
                            ablation_transfer_tradaboost_normal_normal)] = [
                                seed, target_column, train_size_, method,
                                n_estimators, lr, tree_size, val_rmse, val_mae,
                                rmse, mae
                            ]
                        ablation_file = f'results/tradaboost_ablation_rs_{d}__250930.csv'
                        file_exists = os.path.isfile(ablation_file)
                        ablation_transfer_tradaboost_normal_normal.to_csv(
                            ablation_file, mode='a', header=not file_exists)


#train_run_tester_trada(1)
