import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from baselines import *
from utils import *
import stem_config as c
import copy

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

log_columns = [
    'seed', 'learning_rate', 'dropout', 'd_main', 'num_blocks', 
    'val_rmse_scaled', 'val_mae_scaled', 'val_rmse_raw', 'val_mae_raw', 
    'test_rmse_scaled', 'test_mae_scaled', 'test_rmse_raw', 'test_mae_raw'
]
results_df = pd.DataFrame(columns=log_columns)

# data (as pandas dataframes) 
data = pd.read_csv('../datasets/stem_data.csv')
data_target = data[data['Species'] == 'Spruce']
data_source = data[data['Species'] == 'Pine']

# #split according to latitude
# q3 = np.percentile(data.copy()['Lat'], 25)
# data_source = data[data['Lat'] >= q3]
# data_target = data[data['Lat'] < q3]

X_source_train_raw = data_source[c.predictor_columns].to_numpy()
y_source_train_raw = data_source["Height"].to_numpy() # change to Height

# Standardize based on the source domain exclusively to preserve feature space alignment during transfer
feature_scaler_src = StandardScaler()
X_source_train_scaled = feature_scaler_src.fit_transform(X_source_train_raw)

target_scaler_src = StandardScaler()
y_source_train_scaled = target_scaler_src.fit_transform(y_source_train_raw.reshape(-1, 1)).flatten()

for config in c.param_grid_ResNet:
    learning_rate, dropout, d_main, num_blocks = config
    
    resnet_base = TabularResNet(
        input_size=X_source_train_scaled.shape[1], 
        d_main=d_main, 
        d_hidden=d_main * 2, 
        num_blocks=num_blocks, 
        dropout_rate=dropout
    )
    
    dataloader_train_source = process_dataset_for_base_network(X_source_train_scaled, y_source_train_scaled, batch_size=16)
    resnet_base, train_loss_src, val_loss_src = train_mlp_on_source(dataloader_train_source, resnet_base, learning_rate=learning_rate, epochs=1000)
    
    base_state_dict = copy.deepcopy(resnet_base.state_dict())

    for seed in c.seed_list:
        resnet_finetuned = TabularResNet(
            input_size=X_source_train_scaled.shape[1], 
            d_main=d_main, 
            d_hidden=d_main * 2, 
            num_blocks=num_blocks, 
            dropout_rate=dropout
        )
        resnet_finetuned.load_state_dict(base_state_dict)

        data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
        data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state=3)

        X_target_train_raw = data_train[c.predictor_columns].to_numpy()
        y_target_train_raw = data_train[c.target_column].to_numpy()
        X_target_val_raw = data_val[c.predictor_columns].to_numpy()
        y_target_val_raw = data_val[c.target_column].to_numpy()
        X_target_test_raw = data_test[c.predictor_columns].to_numpy()
        y_target_test_raw = data_test[c.target_column].to_numpy()

        feature_scaler_tgt = StandardScaler()
        X_target_train_scaled = feature_scaler_tgt.fit_transform(X_target_train_raw)
        X_target_val_scaled = feature_scaler_tgt.transform(X_target_val_raw)
        X_target_test_scaled = feature_scaler_tgt.transform(X_target_test_raw)

        target_scaler_tgt = StandardScaler()
        y_target_train_scaled = target_scaler_tgt.fit_transform(y_target_train_raw.reshape(-1, 1)).flatten()
        y_target_val_scaled = target_scaler_tgt.transform(y_target_val_raw.reshape(-1, 1)).flatten()
        y_target_test_scaled = target_scaler_tgt.transform(y_target_test_raw.reshape(-1, 1)).flatten()
        
        train_dataloader, val_dataloader, test_dataloader = process_datasets_for_finetuning(
            X_target_train_scaled, y_target_train_scaled, 
            X_target_val_scaled, y_target_val_scaled, 
            X_target_test_scaled, y_target_test_scaled, 
            batch_size=16
        )
        
        resnet_finetuned, train_loss_tgt, val_loss_tgt = finetune_mlp_on_target(train_dataloader, val_dataloader, resnet_finetuned, epochs=1000, learning_rate=learning_rate)
        
        test_rmse_scaled, test_mae_scaled = test_final_mlp(dataloader_test=test_dataloader, mlp=resnet_finetuned)
        val_rmse_scaled, val_mae_scaled = test_final_mlp(dataloader_test=val_dataloader, mlp=resnet_finetuned)

        test_rmse_raw, test_mae_raw = calculate_raw_metrics(resnet_finetuned, test_dataloader, target_scaler_tgt)
        val_rmse_raw, val_mae_raw = calculate_raw_metrics(resnet_finetuned, val_dataloader, target_scaler_tgt)
        
        results_df.loc[len(results_df)] = [
            seed, learning_rate, dropout, d_main, num_blocks, 
            val_rmse_scaled, val_mae_scaled, val_rmse_raw, val_mae_raw, 
            test_rmse_scaled, test_mae_scaled, test_rmse_raw, test_mae_raw
        ]
        results_df.to_csv('results_height/ResNet_finetuned_transformed2_EXTEND.csv', index=False)
