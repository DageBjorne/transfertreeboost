import sys

sys.path.append('../')
from baselines import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *  #only needed for xgboost
import rs_config as c
import torch
import torch.nn as nn
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
    

df_exp = pd.DataFrame(columns=['seed', 'learning_rate', 'dropout', 'd_main', 'num_blocks', 'val_rmse', 'val_mae', 'rmse', 'mae'])

# data (as pandas dataframes) 
# data_target = pd.read_csv('../datasets/rs_lettland.csv')[0:300]
# data_source = pd.read_csv('../datasets/rs_sweden.csv')[0:2000]

#split according to latitude
data_target = pd.read_csv('../datasets/rs_lettland.csv')[0:300]
data_source = pd.read_csv('../datasets/rs_sweden.csv')
q3 = np.percentile(data_source.copy()['north_processed'], 75)
data_source = data_source[data_source['north_processed'] >= q3][0:2000]

X_source_train = np.array(data_source[c.predictor_columns])
y_source_train = np.array(data_source[c.target_column]) #change this to "Height" to use Height as source label!

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

        X_target_train = np.array(data_train[c.predictor_columns])
        y_target_train = np.array(data_train[c.target_column])
        X_target_val = np.array(data_val[c.predictor_columns])
        y_target_val = np.array(data_val[c.target_column])
        X_target_test = np.array(data_test[c.predictor_columns])
        y_target_test = np.array(data_test[c.target_column])
        
        train_dataloader, val_dataloader, test_dataloader = process_datasets_for_finetuning(
            X_target_train, y_target_train, X_target_val, y_target_val, X_target_test, y_target_test, batch_size=16
        )
        
        resnet_finetuned, train_loss_tgt, val_loss_tgt = finetune_mlp_on_target(train_dataloader, val_dataloader, resnet_finetuned, epochs=1000, learning_rate=learning_rate)
        
        rmse, mae = test_final_mlp(dataloader_test=test_dataloader, mlp=resnet_finetuned)
        val_rmse, val_mae = test_final_mlp(dataloader_test=val_dataloader, mlp=resnet_finetuned)
        
        df_exp.loc[len(df_exp)] = [seed, learning_rate, dropout, d_main, num_blocks, val_rmse, val_mae, rmse, mae]
        df_exp.to_csv(f'results_location/ResNet_finetuned2.csv', index=False)


