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

    test_size_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    target_columns = ['Volume', 'Dgv']
    seed_list = [1, 2, 3, 4, 5]

    predictor_columns = [
        'pzabovezmean', 'pzabove2', 'zq5', 'zq10', 'zq15', 'zq20', 'zq25',
        'zq30', 'zq35', 'zq40', 'zq45', 'zq50', 'zq55', 'zq60', 'zq65', 'zq70',
        'zq75', 'zq80', 'zq85', 'zq90', 'zq95', 'zpcum1', 'zpcum2', 'zpcum3',
        'zpcum4', 'zpcum5', 'zpcum6', 'zpcum7', 'zpcum8', 'zpcum9'
    ]

    v_list = [0.05, 0.1]
    source_tree_size_list = [1, 2]
    target_tree_size_list = [1, 2]
    k_list = [0.01, 0.05]
    m_0_list = [0.5, 0.9]

    param_grid = list(
        itertools.product(v_list, source_tree_size_list, target_tree_size_list,
                          k_list, m_0_list))

    data_sweden = pd.read_csv(r'datasets/rs_sweden.csv', index_col=[0])
    data_latvia = pd.read_csv(r'datasets/rs_lettland.csv', index_col=[0])

    seed_list = [1, 2, 3, 4, 5]
    seed_list = [seed_list[int(s_idx)]]
    for seed in seed_list:
        for test_size in test_size_list:
            for target_column in target_columns:

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
                              show_curves=False)
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
                    ablation_transfer_real = pd.DataFrame(columns=[
                        'seed', 'target_column', 'target_instances', 'method',
                        'v', 'source_tree_size', 'target_tree_size', 'k',
                        'm_0', 'val_rmse', 'val_mae', 'rmse', 'mae'
                    ])
                    ablation_transfer_real.loc[len(ablation_transfer_real)] = [
                        seed, target_column, train_size, method, v,
                        source_tree_size, target_tree_size, k, m_0, val_rmse,
                        val_mae, rmse, mae
                    ]
                    ablation_file = f'results/LSTransferTreeBoost_ablation_rs.csv'
                    file_exists = os.path.isfile(ablation_file)
                    ablation_transfer_real.to_csv(ablation_file,
                                                  mode='a',
                                                  header=not file_exists)

    fine_tuning_lrs = [1e-4, 5e-5]
    base_lrs = [5e-4, 1e-4]
    dropout_list = [0.0, 0.1]
    include_batch_norm = [True, False]

    for seed in seed_list:
        for test_size in test_size_list:
            for target_column in target_columns:

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

                print(len(X_target_train), len(X_target_val),
                      len(X_target_test))
                #Test for all methods!!!!

                for base_lr in base_lrs:
                    for finetuning_lr in fine_tuning_lrs:
                        for dropout_rate in dropout_list:
                            for batch_norm in include_batch_norm:

                                method = f'MLP'
                                mlp = MLP(30,
                                          100,
                                          100,
                                          100,
                                          1,
                                          dropout_rate=dropout_rate,
                                          include_batch_norm=batch_norm)
                                dataloader_train = process_dataset_for_base_network(
                                    X_source_train, y_source_train)
                                mlp, train_loss, val_loss = train_mlp_on_source(
                                    dataloader_train, mlp, epochs=1000)
                                dataloader_train, dataloader_val, dataloader_test = process_datasets_for_finetuning(
                                    X_target_train,
                                    y_target_train,
                                    X_target_val,
                                    y_target_val,
                                    X_target_test,
                                    y_target_test,
                                    batch_size=32)

                                mlp, train_loss, val_loss = finetune_mlp_on_target(
                                    dataloader_train,
                                    dataloader_val,
                                    mlp,
                                    epochs=1000,
                                    freeze_layers=None)
                                rmse, mae = test_final_mlp(
                                    dataloader_test, mlp)
                                val_rmse, val_mae = test_final_mlp(
                                    dataloader_val, mlp)
                                ablation_transfer_real = pd.DataFrame(columns=[
                                    'seed', 'target_column',
                                    'target_instances', 'method', 'base_lr',
                                    'fine_tuning_lr', 'dropout_rate',
                                    'batch_norm', 'val_rmse', 'val_mae',
                                    'rmse', 'mae'
                                ])
                                ablation_transfer_real.loc[len(
                                    ablation_transfer_real)] = [
                                        seed, target_column, train_size,
                                        method, base_lr, finetuning_lr,
                                        dropout_rate, batch_norm, val_rmse,
                                        val_mae, rmse, mae
                                    ]
                                ablation_file = f'results/MLP_ablation_rs.csv'
                                file_exists = os.path.isfile(ablation_file)
                                ablation_transfer_real.to_csv(
                                    ablation_file,
                                    mode='a',
                                    header=not file_exists)
                                #ablation_transfer_real.to_csv(
                                #    f'results/MLP_ablation_rs.csv')
