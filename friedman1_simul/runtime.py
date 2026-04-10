import sys
sys.path.append('../')
from core import LSTransferTreeBoost, MTransferTreeBoost, LADTransferTreeBoost
import pandas as pd
import numpy as np
from utils import * #only needed for xgboost

from friedman1 import *

import time

df = pd.DataFrame(columns = ['n_samples', 'time (s)', 'Method'])
for n_samples in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    for i in range(50):

        X_target_train, y_target_train = friedman1(n_samples=n_samples, add_noise = False, noise_distribution = 'gaussian', n_features=10, random_seed=i+1) #add noise to train set
        X_source_train, y_source_train = friedman1_altered(n_samples=1000, add_noise = False, noise_distribution = 'gaussian',
                                                            n_features=10, d=5, shift_seed=i, random_seed = i)


        fiter = MTransferTreeBoost(epochs=100, v=0.1, source_tree_size=2, 
                                target_tree_size=2, k=0.0, m_0=0.5)
        start_time = time.time()  # start timer
        fiter.fit(X_target_train, y_target_train, X_source_train, y_source_train)
        end_time = time.time()    # end timer


        fit_duration = end_time - start_time
        df.loc[len(df)] = [n_samples, fit_duration, 'LAD']
        df.to_csv('vizes/runtime_M.csv')
        print(f'iteration {i} complete, fit time: {fit_duration:.4f} seconds')
        
