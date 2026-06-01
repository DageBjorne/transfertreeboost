<div align="center">
  
# TransferTreeBoost
by [Dag Björnberg](), [Jonas Nordqvist](), [Morgan Ericsson](), [Johan Lindeberg](), [Welf Löwe]() and [Johan E.S. Fransson]().

[![Paper link](https://img.shields.io/badge/TMLR-openreview-openreview.svg)](https://openreview.net/forum?id=b29TPa8NPT)
</div> 

This is the repository for ```TransferTreeBoost```, a method for transfer learning based on Gradient Tree Boosting. 

Currently, three methods are available:
- ```LSTransferTreeBoost:``` For least-squares loss, 
- ```LADTransferTreeBoost:``` For least absolute deviation loss 
- `MTransferTreeBoost` For Huber loss. 

## Installation
The packages used for TransferTreeBoost are fairly standard, and most likely other versions than the specified ones can be used. However, to make sure it works, do the following:

- Create a virtual environment with Python 3.13.
- Run `pip install -r requirements.txt`. This will install all needed packages for our method.
- To run experiments, including benchmark methods, also run `pip install -r requirements_dev.txt`.

## Using TransferTreeBoost

Simple examples follow, with default parameters (same default parameters for LAD, LS, and M):

```
from core import LSTransferTreeBoost # or LADTransferTreeBoost, MTransferTreeBoost
#instantiate model
model = LSTransferTreeBoost(v=0.1, epochs=100, target_tree_size=2,
                            source_tree_size=2, k=0.05, m_0=0.9, min_samples_leaf=1)

```
### Model parameters

- **v**: Learning rate, or shrinkage parameter. Shrinks contribution from each base learner to prevent overfitting.

- **epochs**: How many iterations to use.

- **target_tree_size**: Maximum depth of tree fitted on target domain.

- **source_tree_size**: Maximum depth of tree fitted on source domain.

- **k**: Slope of exponential decay scheduler. Determines how fast source trees are decayed.

- **m_0**: Initial value of weight parameter. Determines the initial source tree contribution.

- **min_samples_leaf**: Minimum amount of leaves in nodes (same for source and target).
  
- **quantile**: defines the threshold for outliers in ```MTransferTreeBoost``` (default=0.9).

💡 *Tip:* The interaction between `m_0` and `k` controls the **transfer strength** —  
`m_0` sets how much you start with from the source domain, while `k` determines how quickly that influence fades.
### Fit model
```
#fit model using source and target training data (have to be NumPy arrays)
model.fit(x_train_target, y_train_target, x_train_source, y_train_source,
          show_curves=False, val_x=None, val_y=None, early_stopping_rounds=5)

#evaluate on a target test set
mse = model.evaluate(x_test_target, y_test_target, metric = 'mse')

#predict for new unseen data
preds = model.predict(x_test_target_unseen)
```

### Fit Method Parameters

- **x_train_target** *(array-like, required)*: Target train predictors.

- **y_train_target** *(array-like, required)*: Target train responses.

- **x_train_source** *(array-like, required)*: Source train predictors.

- **y_train_source** *(array-like, required)*: Source train responses.

- **show_curves** *(bool, default=False)*: Set to `True` to display train loss. If `val_x` and `val_y` are not `None`, it will also display the validation loss.

- **val_x** *(array-like, default=None)*: Target validation predictors.

- **val_y** *(array-like, default=None)*: Target validation responses.

- **early_stopping_rounds** *(int, default=5)*: Number of rounds with no improvement on the validation set before early stopping is triggered.


### Save and load models:

The fitted model can be saved with e.g. `joblib`. 

```
import joblib
joblib.dump(model, 'model.joblib') #for saving fitted model
model = joblib.load('model.joblib') #to load saved model
```

## Run experiments
To replicate our experiments, run the scripts in the following folders:

- `friedman1_simul`: experiments for the friedman #1 dataset. All results and visualizations are saved, but all of it can be reproduced following the provided scripts and notebooks.
- `uci`: experiments on UCI datasets 
- `remote-sensing`: experiments for the remote sensing experiments. Unfortunately, access to the datasets are restricted due to confidentiality, so they cannot be shared. However, we have saved the results, so that the visualizations can be reproduced. Moreover, the training logic is provided. We decided to include this to be as transparent as we possible can.
- `stem-profile`: experiments for the stem volume prediction task. As for the `remote-sensing` data, this dataset cannot be shared, but the results are saved in the folder. Also, training logic and visualization notebooks are provided.
- `final_simulations`: experiments for the final friedman experiments.

Each folder has a similar structure, with similar names. For example, in `stem-profile`:
- `stem_exp_transfertreeboost_LS.py`: runs our method (under least-squares loss)
- `stem_exp_xgboost.py`: runs target-only xgboost
- `stem_exp_xgboost_warmstart.py`: runs XGBoost warmstart
- `stem_exp_xgboost_naive.py`: runs Pooled XGBoost
- `stem_exp_trada.py`: runs TrAdaBoost.R2
- `stem_exp_ResNet_targetonly.py`: runs target-only ResNet
- `stem_exp_ResNet_finetuned.py`: runs train on source, refine on target ResNet

## Citation
If you find our work useful, please cite it
```
bibtex
@article{
bj{\"o}rnberg2026gradient,
title={Gradient Tree Boosting for Regression Transfer},
author={Dag Bj{\"o}rnberg and Jonas Nordqvist and Morgan Ericsson and Johan Lindeberg and Welf L{\"o}we and Johan E.S. Fransson},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=b29TPa8NPT},
note={}
}
```


