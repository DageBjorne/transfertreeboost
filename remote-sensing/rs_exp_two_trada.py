import sys
sys.path.append('../')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
import rs_config as c

import warnings
warnings.filterwarnings('ignore')
from adapt.instance_based import TrAdaBoostR2
from m5py import M5Prime

from baselines import *
  
df_exp = pd.DataFrame(columns = ['seed', 'lr', 'n_estimators', 'tree_size', 'rmse', 'mae'])

for seed in c.seed_list:
    
    # Sweden/Latvia split
    # data_target = pd.read_csv('../datasets/rs_lettland.csv')[0:300]
    # data_source = pd.read_csv('../datasets/rs_sweden.csv')[0:2000]

    #split according to latitude
    data_target = pd.read_csv('../datasets/rs_lettland.csv')[0:300]
    data_source = pd.read_csv('../datasets/rs_sweden.csv')
    q3 = np.percentile(data_source.copy()['north_processed'], 75)
    data_source = data_source[data_source['north_processed'] >= q3][0:2000]

    ###########################################################

    ### Split data into train/validation/test
    data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
    data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state = 3)

    X_source_train = np.array(data_source[c.predictor_columns])
    y_source_train = np.array(data_source[c.target_column]) #change this to "Volume" to use volume as source label!


    #Specific train and test set
    X_target_train = np.array(data_train[c.predictor_columns])
    y_target_train = np.array(data_train[c.target_column])

    X_target_test = np.array(data_test[c.predictor_columns])
    y_target_test = np.array(data_test[c.target_column])

    ############################################################



    ### Loop over possible hyperparameter settings


    for config in c.param_grid_TwoTrada:
        lr, n_estimators, tree_size = config
        base_estimator = M5Prime(max_depth=tree_size)
        model = TrAdaBoostR2(base_estimator,
                                n_estimators=n_estimators,
                                lr=lr)
        model.fit(X_source_train, y_source_train,
                    X_target_train, y_target_train)
        preds = model.predict(X_target_test)

        rmse = compute_rmse(preds, y_target_test)
        mae = compute_mae(preds, y_target_test)
        df_exp.loc[len(df_exp)] = [seed, lr, n_estimators, tree_size, rmse, mae]
        df_exp.to_csv(f'results_location/two_trada.csv')



    