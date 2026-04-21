import sys
sys.path.append('../')

from baselines import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from utils import *
import uci_config as c
from ucimlrepo import fetch_ucirepo 
import copy

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

id_list = [925, 165, 477, 162] #InfraRed = 925, concrete = 165, Auto MPG = 9, 
        #Real estate valuation = 477, Air-foil self-noise = 291, forest fires = 162

for dataset_id in id_list:
    df_exp = pd.DataFrame(columns=['seed', 'learning_rate', 'dropout', 'd_main', 'num_blocks', 'val_rmse', 'val_mae', 'rmse', 'mae'])

    data = fetch_ucirepo(id=dataset_id)  
    X = data.data.features.reset_index(drop=True)
    y = data.data.targets.reset_index(drop=True)
    y = y[y.columns[0]] 
    
    data_full = X.copy()
    data_full['target'] = y 
    data_full = data_full.dropna() 
    
    for col in data_full.select_dtypes(include='object').columns:
        data_full[col] = data_full[col].astype('category')
    for col in data_full.select_dtypes(include='category').columns:
        data_full[col] = data_full[col].cat.codes

    target_column = 'target'
    predictor_columns = list(X.columns)

    corr_coefs = []
    for feature in X.columns:
        feature_data = X[feature]
        if np.issubdtype(feature_data.dtype, np.number):
            corr_coefs.append(feature_data.corr(y))
        else:
            corr_coefs.append(0)

    threshold = 0.4
    idx = min(range(len(corr_coefs)), key=lambda i: abs(abs(corr_coefs[i]) - threshold)) 
    selected_variable = X.columns[idx]

    data_full = data_full.sort_values(by=selected_variable)
    data_full = data_full.drop(columns=selected_variable)
    predictor_columns.remove(selected_variable)

    n = len(data_full)
    t = n // 4 
    df_splits = [data_full.iloc[:t], data_full.iloc[t:2*t], data_full.iloc[2*t:3*t], data_full.iloc[3*t:]]

    np.random.seed(dataset_id)
    random_index = np.random.choice([0, 3])
    data_target = df_splits[random_index]
    data_source = pd.concat([df_splits[i] for i in range(4) if i != random_index], ignore_index=True)

    X_source_train = np.array(data_source[predictor_columns])
    y_source_train = np.array(data_source[target_column]) 

    for config in c.param_grid_ResNet:
        learning_rate, dropout, d_main, num_blocks = config
        
        resnet_base = TabularResNet(
            input_size=X_source_train.shape[1], 
            d_main=d_main, 
            d_hidden=d_main * 2, 
            num_blocks=num_blocks, 
            dropout_rate=dropout
        )
        
        dataloader_train_source = process_dataset_for_base_network(X_source_train, y_source_train, batch_size=16)
        
        resnet_base, train_loss_src, val_loss_src = train_mlp_on_source(dataloader_train_source, resnet_base, learning_rate=learning_rate, epochs=1000)
        
        # NOTE: Deepcopy used to ensure the base state remains pristine across seed iterations
        base_state_dict = copy.deepcopy(resnet_base.state_dict())

        for seed in c.seed_list:
            resnet_finetuned = TabularResNet(
                input_size=X_source_train.shape[1], 
                d_main=d_main, 
                d_hidden=d_main * 2, 
                num_blocks=num_blocks, 
                dropout_rate=dropout
            )
            resnet_finetuned.load_state_dict(base_state_dict)

            data_temp, data_test = train_test_split(data_target, test_size=0.2, random_state=seed)
            data_train, data_val = train_test_split(data_temp, test_size=0.25, random_state=3)

            X_target_train = np.array(data_train[predictor_columns])
            y_target_train = np.array(data_train[target_column])
            X_target_val = np.array(data_val[predictor_columns])
            y_target_val = np.array(data_val[target_column])
            X_target_test = np.array(data_test[predictor_columns])
            y_target_test = np.array(data_test[target_column])
            
            train_dataloader, val_dataloader, test_dataloader = process_datasets_for_finetuning(
                X_target_train, y_target_train, X_target_val, y_target_val, X_target_test, y_target_test, batch_size=16
            )
            
            resnet_finetuned, train_loss_tgt, val_loss_tgt = finetune_mlp_on_target(train_dataloader, val_dataloader, resnet_finetuned, epochs=1000, learning_rate=learning_rate)
            
            rmse, mae = test_final_mlp(dataloader_test=test_dataloader, mlp=resnet_finetuned)
            val_rmse, val_mae = test_final_mlp(dataloader_test=val_dataloader, mlp=resnet_finetuned)
            
            df_exp.loc[len(df_exp)] = [seed, learning_rate, dropout, d_main, num_blocks, val_rmse, val_mae, rmse, mae]
            df_exp.to_csv(f'results/ResNet_finetuned2_{dataset_id}.csv', index=False)