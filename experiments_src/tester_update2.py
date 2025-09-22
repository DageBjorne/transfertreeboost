#import sys

#sys.path.append('../')

from core import LADTransferTreeBoost, LSTransferTreeBoost, MTransferTreeBoost
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from utils import *
import itertools
import os
import matplotlib.pyplot as plt
from baselines import *


def train_run_tester(s_idx):

    train_size_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    target_columns = ['Dgv']
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d_list = [1, 2, 3, 4]

    len_of_smallest_regional_dataset = 1506  # detta är för att få exakt samma train sizes på alla 4 delmängder

    predictor_columns = [
        'pzabovezmean', 'pzabove2', 'zq5', 'zq10', 'zq15', 'zq20', 'zq25',
        'zq30', 'zq35', 'zq40', 'zq45', 'zq50', 'zq55', 'zq60', 'zq65', 'zq70',
        'zq75', 'zq80', 'zq85', 'zq90', 'zq95', 'zpcum1', 'zpcum2', 'zpcum3',
        'zpcum4', 'zpcum5', 'zpcum6', 'zpcum7', 'zpcum8', 'zpcum9'
    ]

    #ablation study for transfertreeboost Gaussian errors, with gaussian source domain errors
    ablation_transfer_real = pd.DataFrame(columns=[
        'seed', 'target_column', 'target_instances', 'method', 'v',
        'source_tree_size', 'target_tree_size', 'k', 'm_0', 'val_rmse',
        'val_mae', 'rmse', 'mae'
    ])
    #ablation_transfer_real = pd.read_csv(f'results/LSTransferTreeBoost_ablation_rs_norrland.csv', index_col = [0])

    v_list = [0.02]
    source_tree_size_list = [1, 2]
    target_tree_size_list = [1, 2]
    k_list = [0.01]
    m_0_list = [0.9]

    # --- Step 2: Create full parameter grid ---
    param_grid = list(
        itertools.product(v_list, source_tree_size_list, target_tree_size_list,
                          k_list, m_0_list))

    # --- Step 3: Sample random combinations ---
    #sampled_configs = random.sample(param_grid, n_samples)
    for d in d_list:
        for seed in seed_list:
            for train_size in train_size_list:
                for target_column in target_columns:

                    #data from Svedala
                    data_sweden = pd.read_csv(r'datasets/rs_sweden.csv',
                                              index_col=[0])
                    data_sweden = data_sweden[data_sweden['area_code'] == d]
                    print(len(data_sweden))

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
                    train_size_ = int(len_of_smallest_regional_dataset *
                                      train_size)
                    data_train = data_train[0:train_size_]

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

                    print(len(X_target_train), len(X_target_val),
                          len(X_target_test))
                    for config in param_grid:
                        v, source_tree_size, target_tree_size, k, m_0 = config

                        #Test for all methods!!!!

                        method = f'LSTransferTreeBoost'
                        fiter = LSTransferTreeBoost(
                            epochs=1000,
                            v=v,
                            source_tree_size=source_tree_size,
                            target_tree_size=target_tree_size,
                            k=k,
                            m_0=m_0)
                        fiter.fit(X_target_train,
                                  y_target_train,
                                  X_source_train,
                                  y_source_train,
                                  val_x=X_target_val,
                                  val_y=y_target_val,
                                  early_stopping_rounds=8,
                                  show_curves=True)
                        rmse = fiter.evaluate(X_target_test,
                                              y_target_test,
                                              metric='rmse')
                        val_rmse = fiter.evaluate(X_target_val,
                                                  y_target_val,
                                                  metric='rmse')
                        mae = fiter.evaluate(X_target_test,
                                             y_target_test,
                                             metric='mae')
                        val_mae = fiter.evaluate(X_target_val,
                                                 y_target_val,
                                                 metric='mae')

                        ablation_transfer_real.loc[len(
                            ablation_transfer_real)] = [
                                seed, target_column, train_size_, method, v,
                                source_tree_size, target_tree_size, k, m_0,
                                val_rmse, val_mae, rmse, mae
                            ]
                        ablation_transfer_real.to_csv(
                            f'results/LSTransferTreeBoost_ablation_rs_{d}.csv')
                        ablation_transfer_real = pd.DataFrame(columns=[
                            'seed', 'target_column', 'target_instances',
                            'method', 'v', 'source_tree_size',
                            'target_tree_size', 'k', 'm_0', 'val_rmse',
                            'val_mae', 'rmse', 'mae'
                        ])
                        ablation_transfer_real.loc[len(
                            ablation_transfer_real)] = [
                                seed, target_column, train_size_, method, v,
                                source_tree_size, target_tree_size, k, m_0,
                                val_rmse, val_mae, rmse, mae
                            ]
                        ablation_file = f'results/LSTransferTreeBoost_ablation_rs_{d}.csv'
                        file_exists = os.path.isfile(ablation_file)
                        ablation_transfer_real.to_csv(ablation_file,
                                                      mode='a',
                                                      header=not file_exists)
