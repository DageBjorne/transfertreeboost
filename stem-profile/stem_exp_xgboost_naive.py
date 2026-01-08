import sys

sys.path.append('../')
from baselines import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
import stem_config as c
  


df_exp = pd.DataFrame(columns = ['seed', 'v', 'target_tree_size',
                                                        'val_rmse', 'val_mae', 'rmse', 'mae'])

for seed in c.seed_list:
    
    # data (as pandas dataframes) 
    data = pd.read_csv('../datasets/stem_data.csv')
    data_target = data[data['Species'] == 'Spruce']
    data_source = data[data['Species'] == 'Pine']




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

    # Add domain indicator column
    # Source = 0
    source_indicator = np.zeros((X_source_train.shape[0], 1))
    X_source_train = np.hstack((X_source_train, source_indicator))

    # Target = 1
    target_indicator = np.ones((X_target_train.shape[0], 1))
    X_target_train = np.hstack((X_target_train, target_indicator))

    X_target_comb = np.vstack((X_target_train, X_source_train))
    y_target_comb = np.concatenate((y_target_train, y_source_train))
    # Validation set (target → 1)
    val_indicator = np.ones((X_target_val.shape[0], 1))
    X_target_val = np.hstack((X_target_val, val_indicator))

    # Test set (target → 1)
    test_indicator = np.ones((X_target_test.shape[0], 1))
    X_target_test = np.hstack((X_target_test, test_indicator))

    ############################################################

    ### Loop over possible hyperparameter settings

    for config in c.param_grid_XGBoost:
        v, target_tree_size = config
        params = {
        'objective': 'reg:squarederror',  # Regression with squared error
        'max_depth': target_tree_size,                   # Maximum depth of a tree
        'eta': v,                       # Learning rate
        'eval_metric': 'rmse',           # RMSE as evaluation metric
        }
            
        bst = train_xgboost(X_target_comb, y_target_comb, X_target_val, y_target_val, boosting_rounds=400, 
                            params=params, early_stopping_rounds=5, show_curve=False)
        preds = test_xgboost(X_target_test, bst)
        val_preds = test_xgboost(X_target_val, bst)
        rmse = compute_rmse(preds, y_target_test)
        mae = compute_mae(preds, y_target_test)
        val_rmse = compute_rmse(val_preds, y_target_val)
        val_mae = compute_mae(val_preds, y_target_val)
        df_exp.loc[len(df_exp)] = [seed, v, target_tree_size, 
                                    val_rmse, val_mae, rmse, mae]
        df_exp.to_csv(f'results/xgb_naive.csv')



