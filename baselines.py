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

#TradaBoostR2 (currently unused)
def train_twostagetradaboostr2(X_source_train, y_source_train, X_target_train, y_target_train, n_estimators=100, 
                       tree_size_depth=2, lr=1.0):
    model = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=tree_size_depth, min_samples_leaf=25), 
                         Xt=X_target_train, yt=y_target_train, n_estimators=n_estimators, lr=lr, verbose=0)

    model.fit(X_source_train, y_source_train)

    return model

def predict_twostagetradaboostr2(X_target_test, model):
    preds = model.predict(X_target_test)
    return preds

#MLP
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size_1)  # first layer
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)  # first layer
        self.fc3 = torch.nn.Linear(hidden_size_2, hidden_size_3)  # first layer
        self.relu = torch.nn.ReLU()                          # activation
        self.fc4 = torch.nn.Linear(hidden_size_3, output_size) # output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
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

def train_mlp_on_source(dataloader_train, mlp, epochs=100):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    train_loss = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        mlp.train()
        running_loss = []
        for i, data in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.append(np.sqrt(loss.item()))

        running_loss = np.mean(running_loss)  
        train_loss.append(running_loss)  
    return mlp

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

def finetune_mlp_on_target(dataloader_train, dataloader_val, mlp, epochs=100, freeze_layers=None):
    # --- Freeze layers if requested ---
    if freeze_layers is not None:
        for name, param in mlp.named_parameters():
            if any(name.startswith(layer) for layer in freeze_layers):
                param.requires_grad = False
                print(f"Freezing {name}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=5e-5)
    
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
            running_loss.append(np.sqrt(loss.item()))

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



