import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from ucimlrepo import fetch_ucirepo

from baselines import *
from utils import *
import uci_config as c

# NOTE: Architecture based on "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021).
class ResNetBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout_rate):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm1d(d_main),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_main, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_hidden, d_main)
        )

    def forward(self, x):
        return x + self.seq(x)

class TabularResNet(nn.Module):
    def __init__(self, input_size, d_main, d_hidden, num_blocks, dropout_rate, output_size=1):
        super().__init__()
        self.first_layer = nn.Linear(input_size, d_main)
        self.blocks = nn.ModuleList([
            ResNetBlock(d_main, d_hidden, dropout_rate) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_main),
            nn.ReLU(),
            nn.Linear(d_main, output_size)
        )

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

def calculate_raw_metrics(model, dataloader, target_scaler):
    model.eval()
    preds_scaled, targets_scaled = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            preds_scaled.append(model(x_batch).detach().cpu().numpy())
            targets_scaled.append(y_batch.detach().cpu().numpy())
            
    preds_scaled = np.concatenate(preds_scaled).reshape(-1, 1)
    targets_scaled = np.concatenate(targets_scaled).reshape(-1, 1)
    
    preds_raw = target_scaler.inverse_transform(preds_scaled)
    targets_raw = target_scaler.inverse_transform(targets_scaled)
    
    rmse_raw = root_mean_squared_error(targets_raw, preds_raw)
    mae_raw = mean_absolute_error(targets_raw, preds_raw)
    return rmse_raw, mae_raw

uci_dataset_ids = [925, 165, 477] #162, 9, 291,

for dataset_id in uci_dataset_ids:
    log_columns = [
        'seed', 'learning_rate', 'dropout', 'd_main', 'num_blocks', 
        'val_rmse_scaled', 'val_mae_scaled', 'val_rmse_raw', 'val_mae_raw', 
        'test_rmse_scaled', 'test_mae_scaled', 'test_rmse_raw', 'test_mae_raw'
    ]
    results_df = pd.DataFrame(columns=log_columns)

    data = fetch_ucirepo(id=dataset_id)  
    X = data.data.features.reset_index(drop=True)
    y = data.data.targets.reset_index(drop=True).iloc[:, 0]
    
    data_full = X.copy()
    data_full['target'] = y 
    data_full = data_full.dropna() 
    
    for col in data_full.select_dtypes(include='object').columns:
        data_full[col] = data_full[col].astype('category')
    for col in data_full.select_dtypes(include='category').columns:
        data_full[col] = data_full[col].cat.codes

    predictor_columns = list(X.columns)

    corr_coefs = [X[col].corr(y) if np.issubdtype(X[col].dtype, np.number) else 0 for col in X.columns]
    idx_closest_to_threshold = min(range(len(corr_coefs)), key=lambda i: abs(abs(corr_coefs[i]) - 0.4))
    split_variable = X.columns[idx_closest_to_threshold]

    data_full = data_full.sort_values(by=split_variable).drop(columns=split_variable)
    predictor_columns.remove(split_variable)

    split_size = len(data_full) // 4
    data_splits = [data_full.iloc[i * split_size : (i + 1) * split_size] for i in range(4)]
    data_splits[-1] = data_full.iloc[3 * split_size :] 

    np.random.seed(dataset_id)
    target_split_idx = np.random.choice([0, 3])
    data_target = data_splits[target_split_idx]
    data_source = pd.concat([data_splits[i] for i in range(4) if i != target_split_idx], ignore_index=True)

    for seed in c.seed_list:
        data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
        data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state=3)

        X_source_train = data_source[predictor_columns].to_numpy()
        y_source_train = data_source['target'].to_numpy()
        
        X_target_train = data_train[predictor_columns].to_numpy()
        y_target_train = data_train['target'].to_numpy()
        
        X_target_val = data_val[predictor_columns].to_numpy()
        y_target_val = data_val['target'].to_numpy()
        
        X_target_test = data_test[predictor_columns].to_numpy()
        y_target_test = data_test['target'].to_numpy()

        X_train_comb_raw = np.vstack((X_target_train, X_source_train))
        y_train_comb_raw = np.concatenate((y_target_train, y_source_train)).reshape(-1, 1)


        # BILLIG LÖSNING
        X_train_comb_raw = X_target_train
        y_train_comb_raw = y_target_train

        feature_scaler = StandardScaler()
        X_train_comb_scaled = feature_scaler.fit_transform(X_train_comb_raw)
        X_target_val_scaled = feature_scaler.transform(X_target_val)
        X_target_test_scaled = feature_scaler.transform(X_target_test)

        target_scaler = StandardScaler()
        #y_train_comb_scaled = target_scaler.fit_transform(y_train_comb_raw).flatten()
        y_train_comb_scaled = target_scaler.fit_transform(y_train_comb_raw.reshape(-1, 1)).flatten()
        y_target_val_scaled = target_scaler.transform(y_target_val.reshape(-1, 1)).flatten()
        y_target_test_scaled = target_scaler.transform(y_target_test.reshape(-1, 1)).flatten()

        #target_train_indicator = np.ones((X_target_train.shape[0], 1))
        #source_indicator = np.zeros((X_source_train.shape[0], 1))
        #train_comb_indicator = np.vstack((target_train_indicator, source_indicator))
        
        #X_train_comb_final = np.hstack((X_train_comb_scaled, train_comb_indicator))
        #X_train_comb_final = X_target_train
        
        #val_indicator = np.ones((X_target_val_scaled.shape[0], 1))
        #X_target_val_final = np.hstack((X_target_val_scaled, val_indicator))

        #test_indicator = np.ones((X_target_test_scaled.shape[0], 1))
        #X_target_test_final = np.hstack((X_target_test_scaled, test_indicator))

        for config in c.param_grid_ResNet:
            learning_rate, dropout, d_main, num_blocks = config
            
            resnet = TabularResNet(
                input_size=X_train_comb_scaled.shape[1], 
                d_main=d_main, 
                d_hidden=d_main * 2, 
                num_blocks=num_blocks, 
                dropout_rate=dropout
            )
        
            # TODO: Ensure batch_size > 16 if using BatchNorm on very small datasets to avoid unstable statistics
            train_dataloader, val_dataloader, test_dataloader = process_datasets_for_finetuning(
                X_train_comb_scaled, y_train_comb_scaled, X_target_val_scaled, y_target_val_scaled, X_target_test_scaled, y_target_test_scaled, batch_size=16
            )
            
            resnet, train_loss, val_loss = finetune_mlp_on_target(train_dataloader, val_dataloader, resnet, epochs=1000, learning_rate=learning_rate)
            
            test_rmse_scaled, test_mae_scaled = test_final_mlp(dataloader_test=test_dataloader, mlp=resnet)
            val_rmse_scaled, val_mae_scaled = test_final_mlp(dataloader_test=val_dataloader, mlp=resnet)
            
            test_rmse_raw, test_mae_raw = calculate_raw_metrics(resnet, test_dataloader, target_scaler)
            val_rmse_raw, val_mae_raw = calculate_raw_metrics(resnet, val_dataloader, target_scaler)
            
            results_df.loc[len(results_df)] = [
                seed, learning_rate, dropout, d_main, num_blocks, 
                val_rmse_scaled, val_mae_scaled, val_rmse_raw, val_mae_raw, 
                test_rmse_scaled, test_mae_scaled, test_rmse_raw, test_mae_raw
            ]
            results_df.to_csv(f'results/ResNet_TARGETONLY_{dataset_id}.csv', index=False)