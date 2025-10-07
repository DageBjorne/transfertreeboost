# TransferTreeBoost

This repository is used for employing ```TransferTreeBoost```, a method for transfer learning based on Gradient Tree Boosting. 

Currently, three methods are available:
- ```LSTransferTreeBoost:``` For least-squares loss, 
- ```LADTransferTreeBoost:``` For least absolute deviation loss 
- `MTransferTreeBoost` For Huber loss. This is currently at a very initial stage.

## Installation
The packages used for TransferTreeBoost are fairly standard, and most likely other versions than the specified ones can be used. However, to make sure it works, do the following:

- Create a virtual environment with Python 3.13.
- Run `pip install -r requirements.txt`. This will install all needed packages for our method.
- To run experiments, including benchmark methods, also run `pip install -r requirements_dev.txt`.

## Using TransferTreeBoost

Simple examples follow, with default parameters (same default parameters for LAD as LS):

```
from core import LSTransferTreeBoost # or LADTransferTreeBoost
#instantiate model
model = LSTransferTreeBoost(v=0.1, epochs=100, target_tree_size=2,
                            source_tree_size=2, k=0.05, m_0=0.9, min_samples_leaf=25)

```
### Model parameters

- **v**: Learning rate, or shrinkage parameter. Shrinks contribution from each base learner to prevent overfitting.

- **epochs**: How many iterations to use.

- **target_tree_size**: Maximum depth of tree fitted on target domain.

- **source_tree_size**: Maximum depth of tree fitted on source domain.

- **k**: Slope of exponential decay scheduler. Determines how fast source trees are decayed.

- **m_0**: Initial value of weight parameter. Determines the initial source tree contribution.

- **min_samples_leaf**: Minimum amount of leaves in nodes (same for source and target).

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
