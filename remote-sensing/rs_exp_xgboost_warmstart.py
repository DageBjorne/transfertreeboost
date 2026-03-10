import sys

sys.path.append('../')
from baselines import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
import rs_config as c
  


df_exp = pd.DataFrame(columns = ['seed', 'v', 'target_tree_size',
                                                        'val_rmse', 'val_mae', 'rmse', 'mae'])

for seed in c.seed_list:
    
    # data (as pandas dataframes) 
    data_target = pd.read_csv('../datasets/rs_lettland.csv').sample(n=200, random_state=1)
    data_source = pd.read_csv('../datasets/rs_sweden.csv').sample(n=2000, random_state=1)

    # # #split according to latitude
    # data_target = pd.read_csv('../datasets/rs_lettland.csv').sample(n=200, random_state=1)
    # data_source = pd.read_csv('../datasets/rs_sweden.csv')
    # q3 = np.percentile(data_source.copy()['north_processed'], 75)
    # data_source = data_source[data_source['north_processed'] >= q3]

    ###########################################################

    ### Split data into train/validation/test
    data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
    data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state = 3)

    X_source_train = np.array(data_source[c.predictor_columns])
    y_source_train = np.array(data_source[c.predictor_columns]) #change this to "Volume" to use diameter as source label!




### Split data into train/validation/test
    data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
    data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state = 3)

    #Split source data into train/val for this approach
    data_source_train, data_source_val = train_test_split(data_source, test_size=0.2, random_state=seed)

    X_source_train = np.array(data_source_train[c.predictor_columns])
    y_source_train = np.array(data_source_train[c.target_column]) #set to "Volume to change source label!"

    X_source_val = np.array(data_source_val[c.predictor_columns])
    y_source_val = np.array(data_source_val[c.target_column]) 
    #y_source_train = y_source_train**1.5
    #Specific train and test set
    X_target_train = np.array(data_train[c.predictor_columns])
    y_target_train = np.array(data_train[c.target_column])

    X_target_val = np.array(data_val[c.predictor_columns])
    y_target_val = np.array(data_val[c.target_column])

    X_target_test = np.array(data_test[c.predictor_columns])
    y_target_test = np.array(data_test[c.target_column])

    ### Loop over possible hyperparameter settings

    ### Loop over possible hyperparameter settings

    for config in c.param_grid_XGBoost:
        v, target_tree_size = config
        params = {
        'objective': 'reg:squarederror',  # Regression with squared error
        'max_depth': target_tree_size,                   # Maximum depth of a tree
        'eta': v,                       # Learning rate
        'eval_metric': 'rmse',           # RMSE as evaluation metric
        }
            
        bst_base = train_xgboost(X_source_train, y_source_train, X_source_val, y_source_val, boosting_rounds=400, 
                            params=params, early_stopping_rounds=5, show_curve=False)
            
        bst = train_xgboost(X_target_train, y_target_train, X_target_val, y_target_val, boosting_rounds=400, 
                            params=params, early_stopping_rounds=5, show_curve=False, xgb_model = bst_base)
        preds = test_xgboost(X_target_test, bst)
        val_preds = test_xgboost(X_target_val, bst)
        rmse = compute_rmse(preds, y_target_test)
        mae = compute_mae(preds, y_target_test)
        val_rmse = compute_rmse(val_preds, y_target_val)
        val_mae = compute_mae(val_preds, y_target_val)
        df_exp.loc[len(df_exp)] = [seed, v, target_tree_size, 
                                    val_rmse, val_mae, rmse, mae]
        df_exp.to_csv(f'results_normal/xgb_warmstart.csv')



    