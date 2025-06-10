# TransferTreeBoost

This repository is used for employing ```TransferTreeBoost```, a method for transfer learning based on Gradient Tree Boosting. 

Currently, two methods are available: ```L2TransferTreeBoost``` (least squares loss) and ```LADTransferTreeBoost``` (least absolute deviation loss).

To test it, create a virtual environment and install the requirements.

## Installation

To get the latest updates from the repository run: 

```
git clone https://github.com/DageBjorne/transfertreeboost.git
pip install -r requirements.txt
```

## Using TransferTreeBoost

A simple example follows below:

```
from utils import *
from core import LSTransferTreeBoost
#instantiate model
model = LSTransferTreeBoost()

#fit model using source and target training data (have to be NumPy arrays)
model.fit(x_train_target, y_train_target, x_train_source, y_train_source)

#evaluate on a target test set
mse = model.evaluate(x_test_target, y_test_target, metric = 'mse')

#predict for new unseen data
preds = model.predict(x_test_target_unseen)
```

## Save and load models

The fitted model can be saved with e.g. `joblib`. 

```
import joblib
joblib.dump(model, 'model.joblib') #for saving fitted model
model = joblib.load('model.joblib') #to load saved model
