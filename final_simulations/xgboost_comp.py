import sys

sys.path.append('../')

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
from baselines import *

from friedman1 import *
import final_simulations_config as c

#Run experiments for gaussian distributed errors (target_instances = 200, d = 6)
#We already have the experiments for LS in from before so only LAD is needed

df = pd.DataFrame(columns=[
    'seed', 'd', 'v',  'target_tree_size', 'val_rmse', 'val_mae', 'rmse', 'mae'
])

for seed in c.seed_list:

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
        n_samples=200,
        add_noise=False,
        noise_distribution='gaussian',
        n_features=10,
        random_seed=seed)  #add noise to train set

    for d in c.d_list:
        
    
        X_source_train, y_source_train = friedman1_altered(
            n_samples=1000,
            add_noise=False,
            noise_distribution='gaussian',
            n_features=10,
            d=d,
            shift_seed=seed,
            random_seed=seed)  #also add noise to source (only train here)


        method = f'XGBoost'
        for config in c.param_grid_XGBoost:
            v, target_tree_size = config
            params = {
            'objective': 'reg:squarederror',  # Regression with squared error
            'max_depth': target_tree_size,                   # Maximum depth of a tree
            'eta': v,                       # Learning rate
            'eval_metric': 'rmse', 
        # RMSE as evaluation metric
            }
                
            bst = train_xgboost(X_target_train, y_target_train, X_target_val, y_target_val, boosting_rounds=1000, 
                                params=params, early_stopping_rounds=5, show_curve=False)
            preds = test_xgboost(X_target_test, bst)
            val_preds = test_xgboost(X_target_val, bst)
            rmse = compute_rmse(preds, y_target_test)
            mae = compute_mae(preds, y_target_test)
            val_rmse = compute_rmse(val_preds, y_target_val)
            val_mae = compute_mae(val_preds, y_target_val)
            df.loc[len(df)] = [seed, d, v, target_tree_size, 
                                        val_rmse, val_mae, rmse, mae]
            df.to_csv(f'results_200/xgb.csv')