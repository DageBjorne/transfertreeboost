import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from adapt.instance_based import TrAdaBoostR2, TwoStageTrAdaBoostR2
from sklearn.metrics import mean_squared_error, mean_absolute_error

import itertools
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append('../')
from friedman1 import *

test_size = 1000
val_size = 1000

target_instances_list = [100, 200, 300]
source_instances = 1000  #we use a fixed number of source instances

#d_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
seed_list = [1, 2, 3, 4, 5]

from sklearn.tree import DecisionTreeRegressor
from adapt.instance_based import TrAdaBoostR2

from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings("ignore")

#ablation study for TradaBoostR2, Gaussian errors, with gaussian source domain errors
ablation_transfer_tradaboost_normal_normal = pd.DataFrame(columns=[
    'seed', 'd', 'target_instances', 'method', 'n_estimators', 'lr',
    'tree_size', 'val_rmse', 'val_mae', 'rmse', 'mae'
])

n_estimators_list = [10, 20, 30, 40, 50]
lr_list = [0.1, 0.5, 1.0]
tree_size_list = [1, 2, 3, 4, 5]


def train_run_trada(d_idx):
    d_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d_list = [d_list[int(d_idx)]]
    # --- Step 2: Create full parameter grid ---
    param_grid = list(
        itertools.product(n_estimators_list, lr_list, tree_size_list))

    # --- Step 3: Sample random combinations ---
    #sampled_configs = random.sample(param_grid, n_samples)

    for seed in seed_list:
        for target_instances in target_instances_list:

            X_target_test, y_target_test = friedman1(
                n_samples=test_size,
                add_noise=False,
                noise_distribution='gaussian',
                n_features=10,
                random_seed=seed)  #do NOT add noise to test set!!!!
            X_target_val, y_target_val = friedman1(
                n_samples=val_size,
                add_noise=False,
                noise_distribution='gaussian',
                n_features=10,
                random_seed=seed + 10)  #do NOT add noise to test set!!!!
            X_target_train, y_target_train = friedman1(
                n_samples=target_instances,
                add_noise=True,
                noise_distribution='gaussian',
                n_features=10,
                random_seed=seed)  #add noise to train set
            for d in d_list:
                X_source_train, y_source_train = friedman1_altered(
                    n_samples=1000,
                    add_noise=True,
                    noise_distribution='gaussian',
                    n_features=10,
                    d=d,
                    shift_seed=seed,
                    random_seed=seed
                )  #also add noise to source (only train here)
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
                    ablation_transfer_tradaboost_normal_normal.loc[len(
                        ablation_transfer_tradaboost_normal_normal)] = [
                            seed, d, target_instances, method, n_estimators,
                            lr, tree_size, val_rmse, val_mae, rmse, mae
                        ]

                    import os

                    ablation_file = f'results/trada_ablation_friedman.csv'
                    file_exists = os.path.isfile(ablation_file)

                    ablation_transfer_tradaboost_normal_normal.to_csv(
                        ablation_file, mode='a', header=not file_exists)

                    #ablation_transfer_tradaboost_normal_normal.to_csv(
                    #    f'results/tradaboost_ablation_friedman.csv')
