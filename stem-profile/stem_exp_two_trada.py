import sys
sys.path.append('../')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
import stem_config as c

import warnings
warnings.filterwarnings('ignore')
from adapt.instance_based import TrAdaBoostR2
from m5py import M5Prime

from baselines import *
  
df_exp = pd.DataFrame(columns = ['seed', 'lr', 'n_estimators', 'rmse', 'mae'])

for seed in c.seed_list:
    
    # split according to species 
    data = pd.read_csv('../datasets/stem_data.csv')
    data_target = data[data['Species'] == 'Spruce']
    data_source = data[data['Species'] == 'Pine']

    # #split according to latitude
    # q3 = np.percentile(data.copy()['Lat'], 25)
    # data_source = data[data['Lat'] >= q3]
    # data_target = data[data['Lat'] < q3]




    ###########################################################
    
    ### Split data into train/validation/test
    data_train, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)

    X_source_train = np.array(data_source[c.predictor_columns])
    y_source_train = np.array(data_source[c.target_column]) #change this to "Height" to use Height as source label!

    #Specific train and test set
    X_target_train = np.array(data_train[c.predictor_columns])
    y_target_train = np.array(data_train[c.target_column])

    X_target_test = np.array(data_test[c.predictor_columns])
    y_target_test = np.array(data_test[c.target_column])



    ### Loop over possible hyperparameter settings


    for config in c.param_grid_TwoTrada:
        lr, n_estimators= config
        base_estimator = M5Prime()
        model = TrAdaBoostR2(base_estimator,
                                n_estimators=n_estimators,
                                lr=lr)
        model.fit(X_source_train, y_source_train,
                    X_target_train, y_target_train)
        preds = model.predict(X_target_test)

        rmse = compute_rmse(preds, y_target_test)
        mae = compute_mae(preds, y_target_test)
        df_exp.loc[len(df_exp)] = [seed, lr, n_estimators, rmse, mae]
        df_exp.to_csv(f'results_species/two_trada.csv')



    