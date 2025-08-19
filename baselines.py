import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from adapt.instance_based import TwoStageTrAdaBoostR2
from utils import *

#xgboost (can also be used for naive transfer xgboost)
def train_xgboost(data_train, labels_train, data_val, labels_val, boosting_rounds, params, show_curve = False, early_stopping_rounds=8):
    dtrain = xgb.DMatrix(data_train, label=labels_train)
    dval = xgb.DMatrix(data_val, label=labels_val)

    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=boosting_rounds,
        evals=evallist,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False  # You can still turn this on if you want
    )

    if show_curve:
        # Plot training and validation metrics
        metric_name = list(evals_result['train'].keys())[0]  # e.g., 'logloss' or 'error'
        train_metric = evals_result['train'][metric_name]
        val_metric = evals_result['eval'][metric_name]

        plt.figure(figsize=(10, 6))
        plt.plot(train_metric, label='Train')
        plt.plot(val_metric, label='Validation')
        plt.xlabel('Boosting Round')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'XGBoost {metric_name.capitalize()} Over Boosting Rounds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return bst

def test_xgboost(data_test, bst):
    
    dtest = xgb.DMatrix(data_test)
    preds = bst.predict(dtest)
    return preds

#TradaBoostR2
def train_twostagetradaboostr2(X_source_train, y_source_train, X_target_train, y_target_train, n_estimators=100, 
                       tree_size_depth=2, lr=1.0):
    model = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=tree_size_depth, min_samples_leaf=25), 
                         Xt=X_target_train, yt=y_target_train, n_estimators=n_estimators, lr=lr, verbose=0)

    model.fit(X_source_train, y_source_train)

    return model

def predict_twostagetradaboostr2(X_target_test, model):
    preds = model.predict(X_target_test)
    return preds


