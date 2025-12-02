import sys

sys.path.append('../')
from core import LADTransferTreeBoost, LSTransferTreeBoost, MTransferTreeBoost
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost

from friedman1 import *
import friedman_config as c

#Run experiments for gaussian distributed errors (target_instances = 200, d = 6)
#We already have the experiments for LS in from before so only LAD is needed

df = pd.DataFrame(columns=[
    'seed', 'target_instances', 'd', 'method', 'v', 'source_tree_size', 'target_tree_size', 'k',
    'm_0', 'epochs', 'val_rmse', 'val_mae', 'rmse', 'mae'
])

for seed in c.seed_list:

    for d in c.d_list:
        
        for target_instances in c.target_instances_list:

            X_target_test, y_target_test = friedman1(
                n_samples=1000,
                add_noise=False,
                noise_distribution='gaussian',
                n_features=10,
                random_seed=seed)  #do NOT add noise to test set!!!!
            X_target_val, y_target_val = friedman1(
                n_samples=1000,
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
            X_source_train, y_source_train = friedman1_altered(
                n_samples=1000,
                add_noise=True,
                noise_distribution='gaussian',
                n_features=10,
                d=d,
                shift_seed=seed,
                random_seed=seed)  #also add noise to source (only train here)
            for config in c.param_grid_LSTransferTreeBoost:
                v, source_tree_size, target_tree_size, k, m_0, epochs = config

                #LS already done before

                method = f'LSTransferTreeBoost'
                fiter = LSTransferTreeBoost(epochs=epochs,
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
                          early_stopping_rounds=5,
                          show_curves=False,
                          eval_metric='rmse')
                rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
                val_rmse = fiter.evaluate(X_target_val,
                                          y_target_val,
                                          metric='rmse')
                mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
                val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
                df.loc[len(df)] = [
                    seed, target_instances, d, method, v, source_tree_size, target_tree_size, k, m_0,
                    epochs, val_rmse, val_mae, rmse, mae
                ]

                method = f'LADTransferTreeBoost'
                fiter = LADTransferTreeBoost(epochs=epochs,
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
                          early_stopping_rounds=5,
                          show_curves=False,
                          eval_metric='rmse')
                rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
                val_rmse = fiter.evaluate(X_target_val,
                                          y_target_val,
                                          metric='rmse')
                mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
                val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
                df.loc[len(df)] = [
                    seed, target_instances, d, method, v, source_tree_size, target_tree_size, k, m_0,
                    epochs, val_rmse, val_mae, rmse, mae
                ]

                method = f'MTransferTreeBoost'
                fiter = MTransferTreeBoost(epochs=epochs,
                                        v=v,
                                        source_tree_size=source_tree_size,
                                        target_tree_size=target_tree_size,
                                        k=k,
                                        m_0=m_0,
                                        quantile=0.9)
                fiter.fit(X_target_train,
                        y_target_train,
                        X_source_train,
                        y_source_train,
                        val_x=X_target_val,
                        val_y=y_target_val,
                        early_stopping_rounds=5,
                        show_curves=False)
                rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
                val_rmse = fiter.evaluate(X_target_val,
                                        y_target_val,
                                        metric='rmse')
                mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
                val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
                df.loc[len(df)] = [
                    seed, target_instances, d, method, v, source_tree_size, target_tree_size, k, m_0,
                    epochs, val_rmse, val_mae, rmse, mae
                ]
                df.to_csv('results/gaussian.csv')

# df = pd.DataFrame(columns=[
#     'seed', 'd', 'method', 'v', 'source_tree_size', 'target_tree_size', 'k',
#     'm_0', 'epochs', 'val_rmse', 'val_mae', 'rmse', 'mae'
# ])

# for seed in c.seed_list:

#     for d in c.d_list:

#         X_target_test, y_target_test = friedman1(
#             n_samples=1000,
#             add_noise=False,
#             noise_distribution='slash',
#             n_features=10,
#             random_seed=seed)  #do NOT add noise to test set!!!!
#         X_target_val, y_target_val = friedman1(
#             n_samples=1000,
#             add_noise=False,
#             noise_distribution='slash',
#             n_features=10,
#             random_seed=seed + 10)  #do NOT add noise to test set!!!!
#         X_target_train, y_target_train = friedman1(
#             n_samples=200,
#             add_noise=True,
#             noise_distribution='slash',
#             n_features=10,
#             random_seed=seed)  #add noise to train set
#         X_source_train, y_source_train = friedman1_altered(
#             n_samples=1000,
#             add_noise=True,
#             noise_distribution='slash',
#             n_features=10,
#             d=d,
#             shift_seed=seed,
#             random_seed=seed)  #also add noise to source (only train here)
#         for config in c.param_grid_LSTransferTreeBoost:
#             v, source_tree_size, target_tree_size, k, m_0, epochs = config

#             #LS already done before

#             method = f'LSTransferTreeBoost'
#             fiter = LSTransferTreeBoost(epochs=epochs,
#                                         v=v,
#                                         source_tree_size=source_tree_size,
#                                         target_tree_size=target_tree_size,
#                                         k=k,
#                                         m_0=m_0)
#             fiter.fit(X_target_train,
#                       y_target_train,
#                       X_source_train,
#                       y_source_train,
#                       val_x=X_target_val,
#                       val_y=y_target_val,
#                       early_stopping_rounds=5,
#                       show_curves=False,
#                       eval_metric='rmse')
#             rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
#             val_rmse = fiter.evaluate(X_target_val,
#                                       y_target_val,
#                                       metric='rmse')
#             mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
#             val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
#             df.loc[len(df)] = [
#                 seed, d, method, v, source_tree_size, target_tree_size, k, m_0,
#                 epochs, val_rmse, val_mae, rmse, mae
#             ]

#             method = f'LADTransferTreeBoost'
#             fiter = LADTransferTreeBoost(epochs=epochs,
#                                          v=v,
#                                          source_tree_size=source_tree_size,
#                                          target_tree_size=target_tree_size,
#                                          k=k,
#                                          m_0=m_0)
#             fiter.fit(X_target_train,
#                       y_target_train,
#                       X_source_train,
#                       y_source_train,
#                       val_x=X_target_val,
#                       val_y=y_target_val,
#                       early_stopping_rounds=5,
#                       show_curves=False,
#                       eval_metric='rmse')
#             rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
#             val_rmse = fiter.evaluate(X_target_val,
#                                       y_target_val,
#                                       metric='rmse')
#             mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
#             val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
#             df.loc[len(df)] = [
#                 seed, d, method, v, source_tree_size, target_tree_size, k, m_0,
#                 epochs, val_rmse, val_mae, rmse, mae
#             ]

#             method = f'MTransferTreeBoost'
#             fiter = MTransferTreeBoost(epochs=epochs,
#                                        v=v,
#                                        source_tree_size=source_tree_size,
#                                        target_tree_size=target_tree_size,
#                                        k=k,
#                                        m_0=m_0)
#             fiter.fit(X_target_train,
#                       y_target_train,
#                       X_source_train,
#                       y_source_train,
#                       val_x=X_target_val,
#                       val_y=y_target_val,
#                       early_stopping_rounds=5,
#                       show_curves=False)
#             rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
#             val_rmse = fiter.evaluate(X_target_val,
#                                       y_target_val,
#                                       metric='rmse')
#             mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
#             val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
#             df.loc[len(df)] = [
#                 seed, d, method, v, source_tree_size, target_tree_size, k, m_0,
#                 epochs, val_rmse, val_mae, rmse, mae
#             ]
#             df.to_csv('results/200_slash.csv')

# df = pd.DataFrame(columns=[
#     'seed', 'd', 'method', 'v', 'source_tree_size', 'target_tree_size', 'k',
#     'm_0', 'epochs', 'val_rmse', 'val_mae', 'rmse', 'mae'
# ])

# for seed in c.seed_list:

#     for d in c.d_list:

#         X_target_test, y_target_test = friedman1(
#             n_samples=1000,
#             add_noise=False,
#             noise_distribution='t',
#             n_features=10,
#             random_seed=seed)  #do NOT add noise to test set!!!!
#         X_target_val, y_target_val = friedman1(
#             n_samples=1000,
#             add_noise=False,
#             noise_distribution='t',
#             n_features=10,
#             random_seed=seed + 10)  #do NOT add noise to test set!!!!
#         X_target_train, y_target_train = friedman1(
#             n_samples=200,
#             add_noise=True,
#             noise_distribution='t',
#             n_features=10,
#             random_seed=seed)  #add noise to train set
#         X_source_train, y_source_train = friedman1_altered(
#             n_samples=1000,
#             add_noise=True,
#             noise_distribution='t',
#             n_features=10,
#             d=d,
#             shift_seed=seed,
#             random_seed=seed)  #also add noise to source (only train here)
#         for config in c.param_grid_LSTransferTreeBoost:
#             v, source_tree_size, target_tree_size, k, m_0, epochs = config

#             #LS already done before

#             method = f'LSTransferTreeBoost'
#             fiter = LSTransferTreeBoost(epochs=epochs,
#                                         v=v,
#                                         source_tree_size=source_tree_size,
#                                         target_tree_size=target_tree_size,
#                                         k=k,
#                                         m_0=m_0)
#             fiter.fit(X_target_train,
#                       y_target_train,
#                       X_source_train,
#                       y_source_train,
#                       val_x=X_target_val,
#                       val_y=y_target_val,
#                       early_stopping_rounds=5,
#                       show_curves=False,
#                       eval_metric='rmse')
#             rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
#             val_rmse = fiter.evaluate(X_target_val,
#                                       y_target_val,
#                                       metric='rmse')
#             mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
#             val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
#             df.loc[len(df)] = [
#                 seed, d, method, v, source_tree_size, target_tree_size, k, m_0,
#                 epochs, val_rmse, val_mae, rmse, mae
#             ]

#             method = f'LADTransferTreeBoost'
#             fiter = LADTransferTreeBoost(epochs=epochs,
#                                          v=v,
#                                          source_tree_size=source_tree_size,
#                                          target_tree_size=target_tree_size,
#                                          k=k,
#                                          m_0=m_0)
#             fiter.fit(X_target_train,
#                       y_target_train,
#                       X_source_train,
#                       y_source_train,
#                       val_x=X_target_val,
#                       val_y=y_target_val,
#                       early_stopping_rounds=5,
#                       show_curves=False,
#                       eval_metric='rmse')
#             rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
#             val_rmse = fiter.evaluate(X_target_val,
#                                       y_target_val,
#                                       metric='rmse')
#             mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
#             val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
#             df.loc[len(df)] = [
#                 seed, d, method, v, source_tree_size, target_tree_size, k, m_0,
#                 epochs, val_rmse, val_mae, rmse, mae
#             ]

#             method = f'MTransferTreeBoost'
#             fiter = MTransferTreeBoost(epochs=epochs,
#                                        v=v,
#                                        source_tree_size=source_tree_size,
#                                        target_tree_size=target_tree_size,
#                                        k=k,
#                                        m_0=m_0)
#             fiter.fit(X_target_train,
#                       y_target_train,
#                       X_source_train,
#                       y_source_train,
#                       val_x=X_target_val,
#                       val_y=y_target_val,
#                       early_stopping_rounds=5,
#                       show_curves=False,
#                       eval_metric='rmse',
#                       quantile = 0.5)
#             rmse = fiter.evaluate(X_target_test, y_target_test, metric='rmse')
#             val_rmse = fiter.evaluate(X_target_val,
#                                       y_target_val,
#                                       metric='rmse')
#             mae = fiter.evaluate(X_target_test, y_target_test, metric='mae')
#             val_mae = fiter.evaluate(X_target_val, y_target_val, metric='mae')
#             df.loc[len(df)] = [
#                 seed, d, method, v, source_tree_size, target_tree_size, k, m_0,
#                 epochs, val_rmse, val_mae, rmse, mae
#             ]
#             df.to_csv('results/200_t.csv')
