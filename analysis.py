#%%
from src.gradient_descent import SimulatedAnnealing
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
data = np.load(
    "simulated_annealing_numpy/data/density_emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb_number_istances_10_n_ensamble_1_different_initial_epochs_500_ann_step_500_beta_1.0_final_500.0.npz"
)

n_min = data["min_density"]
n_gs = data["gs_density"]

#%%
for i in range(10):
    plt.plot(n_min[i])
    plt.plot(n_gs[i])

    plt.show()


# %%
