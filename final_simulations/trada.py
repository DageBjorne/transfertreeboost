import sys
sys.path.append('../')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from friedman1 import *
from utils import *

import final_simulations_config as c
from sklearn.metrics import mean_squared_error, mean_absolute_error


import warnings
warnings.filterwarnings('ignore')
from adapt.instance_based import TrAdaBoostR2
from m5py import M5Prime



  


df = pd.DataFrame(columns = ['seed', 'd', 'lr', 'n_estimators', 'tree_size', 'rmse', 'mae'])

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
            n_samples=200,
            add_noise=False,
            noise_distribution='gaussian',
            n_features=10,
            d=d,
            shift_seed=seed,
            random_seed=seed)


        for config in c.param_grid_TwoTrada:
            lr, n_estimators, tree_size = config
            #base_estimator = M5Prime()
            # base_estimator = LinearTreeRegressor(
            #     base_estimator=LinearRegression(),
            #     max_depth=tree_size
            # )

            base_estimator = M5Prime(max_depth=tree_size)
            model = TrAdaBoostR2(base_estimator,
                                    n_estimators=n_estimators,
                                    lr=lr,
                                    #n_estimators_fs=10,
                                    #cv=5)
            )

            model.fit(X_source_train, y_source_train,
                        X_target_train, y_target_train)
            preds = model.predict(X_target_test).ravel()
            print(preds.shape, y_target_test.shape)
            
           # rmse = np.sqrt(mean_squared_error(y_target_test, preds))
            rmse = np.sqrt(np.mean((preds - y_target_test)**2))
            #rmse = compute_rmse(preds, y_target_test)
            mae = mean_absolute_error(y_target_test, preds)
            df.loc[len(df)] = [seed, d, lr, n_estimators, tree_size, rmse, mae]
            df.to_csv(f'results_200/trada.csv')



    