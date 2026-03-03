import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from adapt.instance_based import TwoStageTrAdaBoostR2
from utils import *

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#xgboost (can also be used for naive transfer xgboost and train on source/refine on target)
def train_xgboost(data_train, labels_train, data_val, labels_val, boosting_rounds, params, show_curve = False, early_stopping_rounds=8, xgb_model = None):
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
        verbose_eval=False, 
        xgb_model = xgb_model
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

#MLP
import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, dropout_rate=0.0, include_batch_norm=True):
        super(MLP, self).__init__()
        
        self.include_batch_norm = include_batch_norm
        
        # Layers
        self.fc1 = torch.nn.Linear(input_size, hidden_size_1)
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = torch.nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = torch.nn.Linear(hidden_size_3, output_size)
        
        # Optional BatchNorm
        if include_batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(hidden_size_1)
            self.bn2 = torch.nn.BatchNorm1d(hidden_size_2)
            self.bn3 = torch.nn.BatchNorm1d(hidden_size_3)
        
        # Activation + Dropout
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        if self.include_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        if self.include_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.fc3(x)
        if self.include_batch_norm:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output
        x = self.fc4(x)
        return x

    
def process_dataset_for_base_network(X_source_train, y_source_train, batch_size = 32):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_source_train)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_source_train, dtype=torch.float32).view(-1,1)

    # --- Wrap in TensorDataset ---
    dataset = TensorDataset(X, y)

    # --- Create DataLoader ---
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader_train

import torch
import numpy as np
from torch.utils.data import random_split, DataLoader

def train_mlp_on_source(dataloader_train, mlp, learning_rate = 1e-4, epochs=1000):

    # ---- Split dataset ----
    dataset = dataloader_train.dataset
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader_train = DataLoader(train_dataset, batch_size=dataloader_train.batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=dataloader_train.batch_size, shuffle=False)

    # ---- Loss & Optimizer ----
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    # ---- Training Loop ----
    for epoch in range(epochs):
        mlp.train()
        running_train_loss = []

        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss.append(np.sqrt(loss.item()))

        train_loss = np.mean(running_train_loss)
        train_losses.append(train_loss)

        # ---- Validation ----
        mlp.eval()
        running_loss = []
        for i, data in enumerate(dataloader_val):
            inputs, labels = data
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            running_loss.append(np.sqrt(loss.item()))

        running_loss = np.mean(running_loss)  
        val_losses.append(running_loss)  

        if epoch > 0:
            if early_stopping(4, val_losses, tol=1e-6):
                break

    return mlp, train_losses, val_losses

    


def process_datasets_for_finetuning(X_target_train, y_target_train,
                                    X_target_val, y_target_val, X_target_test, y_target_test, batch_size=32):
    
    
    # --- Scale using train stats only ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_target_train)
    X_val   = scaler.transform(X_target_val)
    X_test  = scaler.transform(X_target_test)


    # --- Convert to tensors ---
    X_train = torch.tensor(X_target_train, dtype=torch.float32)
    y_train = torch.tensor(y_target_train, dtype=torch.float32).view(-1, 1)

    X_val   = torch.tensor(X_target_val, dtype=torch.float32)
    y_val   = torch.tensor(y_target_val, dtype=torch.float32).view(-1, 1)

    X_test  = torch.tensor(X_target_test, dtype=torch.float32)
    y_test  = torch.tensor(y_target_test, dtype=torch.float32).view(-1, 1)

    # --- Wrap in TensorDataset ---
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)

    # --- Create DataLoaders ---
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl

def finetune_mlp_on_target(dataloader_train, dataloader_val, mlp, epochs=100, learning_rate = 5e-5, freeze_layers=None):
    # --- Freeze layers if requested ---
    if freeze_layers is not None:
        for name, param in mlp.named_parameters():
            if any(name.startswith(layer) for layer in freeze_layers):
                param.requires_grad = False
                print(f"Freezing {name}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=learning_rate)
    
    train_loss = []
    val_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        mlp.train()
        running_loss = []
        for i, data in enumerate(dataloader_train):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(np.sqrt(loss.item()))

        running_loss = np.mean(running_loss)  
        train_loss.append(running_loss)  

        mlp.eval()
        running_loss = []
        for i, data in enumerate(dataloader_val):
            inputs, labels = data
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())

        running_loss = np.mean(running_loss)  
        val_loss.append(running_loss)  

        if epoch > 0:
            if early_stopping(8, val_loss, tol=1e-6):
                break

    return mlp, train_loss, val_loss

def test_final_mlp(dataloader_test, mlp):
    mlp.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader_test:
            outputs = mlp(inputs)

            # Move to CPU + NumPy
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Compute RMSE on the whole dataset
    rmse = compute_rmse(all_preds, all_targets)
    mae = compute_mae(all_preds, all_targets)
    return rmse, mae



