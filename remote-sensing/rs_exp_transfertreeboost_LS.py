import sys

sys.path.append('../')
from core import  LSTransferTreeBoost # for simplicity we will work only with M
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
import rs_config as c
  


df_exp = pd.DataFrame(columns = ['seed', 'v', 'source_tree_size', 'target_tree_size', 'k', 'm_0', 'epochs',
                                                        'val_rmse', 'val_mae', 'rmse', 'mae'])

for seed in c.seed_list:
    
    # data (as pandas dataframes) 
    data_target = pd.read_csv('../datasets/rs_lettland.csv')[0:300]
    data_source = pd.read_csv('../datasets/rs_sweden.csv')[0:2000]




    ###########################################################

    ### Split data into train/validation/test
    data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
    data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state = 3)

    X_source_train = np.array(data_source[c.predictor_columns])
    y_source_train = np.array(data_source[c.target_column]) #change this to "Dgv" to use diameter as source label!

    #Specific train and test set
    X_target_train = np.array(data_train[c.predictor_columns])
    y_target_train = np.array(data_train[c.target_column])

    X_target_val = np.array(data_val[c.predictor_columns])
    y_target_val = np.array(data_val[c.target_column])

    X_target_test = np.array(data_test[c.predictor_columns])
    y_target_test = np.array(data_test[c.target_column])

    ############################################################

    ### Loop over possible hyperparameter settings

    for config in c.param_grid_TransferTreeBoost:
        v, source_tree_size, target_tree_size, k, m_0, epochs = config
        fiter = LSTransferTreeBoost(epochs=epochs, 
                                v = v,
                                source_tree_size = source_tree_size,  
                                target_tree_size = target_tree_size,
                                k = k,
                                m_0 = m_0)
        fiter.fit(X_target_train, y_target_train, X_source_train, y_source_train, val_x = X_target_val, val_y = y_target_val, show_curves = False)
        rmse = fiter.evaluate(X_target_test, y_target_test, metric = 'rmse')
        val_rmse = fiter.evaluate(X_target_val, y_target_val, metric = 'rmse')
        mae = fiter.evaluate(X_target_test, y_target_test, metric = 'mae')
        val_mae = fiter.evaluate(X_target_val, y_target_val, metric = 'mae')
        df_exp.loc[len(df_exp)] = [seed, v, source_tree_size, target_tree_size, k, m_0, epochs, 
                                                        val_rmse, val_mae, rmse, mae]
        df_exp.to_csv(f'results/ttb_LS.csv')



    