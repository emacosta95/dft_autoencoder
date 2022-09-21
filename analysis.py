#%%
from typing import List, Dict, Tuple
import numpy as np
import argparse
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm.notebook import tqdm, trange
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torchmetrics import R2Score
from src.model import Energy
from src.utils_analysis import dataloader


#%% Meyer study

ls = 16
model_name = f"meyer_case/cnn_for_gaussian_test_3_60_hc_13_ks_2_ps_16_ls_0.001_vb" 
n_ensambles = 1
n_instances = 250
epochs = 10000
diff_soglia = -1
variable_lr = False
early_stopping = False
lr = 1


#%%

n_min,n_gs,_=dataloader('density',model_name=model_name,n_instances=n_instances,lr=lr,diff_soglia=diff_soglia,epochs=epochs,early_stopping=early_stopping,variable_lr=variable_lr,n_ensambles=n_ensambles)
e_min,e_gs=dataloader('energy',model_name=model_name,n_instances=n_instances,lr=lr,diff_soglia=diff_soglia,epochs=epochs,early_stopping=early_stopping,variable_lr=variable_lr,n_ensambles=n_ensambles)


#%%
for i in range(n_min.shape[0]):

    plt.plot(n_min[i])
    plt.plot(n_gs[i])
    plt.show()

# %%
v=np.load('data/dataset_meyer/dataset_meyer_test_256_100.npz')['potential']

f_gs=e_gs-np.average(n_gs*v[:n_gs.shape[0]])
f_min=e_min-np.average(n_min*v[:n_min.shape[0]])

print(np.average(np.abs(f_gs-f_min)))
# %%
