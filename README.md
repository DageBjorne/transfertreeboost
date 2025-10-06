# TransferTreeBoost

This repository is used for employing ```TransferTreeBoost```, a method for transfer learning based on Gradient Tree Boosting. 

Currently, three methods are available: ```LSTransferTreeBoost``` (least squares loss), ```LADTransferTreeBoost``` (least absolute deviation loss) , `MTransferTreeBoost` (Huber loss, but at initial stage)

To test it, create a virtual environment and install the requirements.

## Using TransferTreeBoost

Simple examples follow, with default parameters (same default parameters for LAD as LS):

```
from core import LSTransferTreeBoost
#instantiate model
model = LSTransferTreeBoost(v=0.1, epochs=100, target_tree_size=2,
                            source_tree_size=2, k=0.05, m_0=0.9, min_samples_leaf=25)

```
**Parameters**:
```
v: learning rate, or shrinkage parameter. Shrinks contribution from each base learner to rpevent overfitting.

epochs: how many iterations to use.

target_tree_size: maximum depth of tree fitted on target domain.

source_tree_size: maximum depth of tree fitted on source domain.

k: slope of exponential decay.

m_0: initial value of weigth parameter.

min_samples_leaf: minimum amount of leaves in nodes (same for source and target).
```
**Fit model**:
```
#fit model using source and target training data (have to be NumPy arrays)
model.fit(x_train_target, y_train_target, x_train_source, y_train_source)

#evaluate on a target test set
mse = model.evaluate(x_test_target, y_test_target, metric = 'mse')

#predict for new unseen data
preds = model.predict(x_test_target_unseen)
```

**Save and load models**:

The fitted model can be saved with e.g. `joblib`. 

```
import joblib
joblib.dump(model, 'model.joblib') #for saving fitted model
model = joblib.load('model.joblib') #to load saved model
```
