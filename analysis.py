# %%
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
from src.utils_analysis import dataloader, test_models_dft, test_models_vae

# %% Meyer study

ls = 16
n_test = 2
model_name = (
    f"3d_speckle/cnn_softplus_for_3dspeckle_271222_60_hc_5_ks_2_ps_16_ls_0.001_vb"
)
# model_name = "meyer_case/cnn_for_gaussian_60_hc_13_ks_2_ps_16_ls_1e-06_vb"
# model_name=f"meyer_case/cnn_for_gaussian_test_5_60_hc_13_ks_2_ps_16_ls_0.1_vb"
# model_name=f"meyer_case/cnn_for_gaussian_test_4_60_hc_13_ks_2_ps_16_ls_0.01_vb"
n_ensambles = 1
n_instances = 250
epochs = 10000
diff_soglia = -1
variable_lr = False
early_stopping = False
lr = 1

data_path_test = "data/dataset_speckle_3d/test.npz"
f = np.load(data_path_test)["F"]
e = np.load(data_path_test)["energy"]


plt.hist(f, bins=200)
plt.show()

plt.hist(e, bins=200)
plt.show()
# %%
# test the models
dn, n_std, n_recons = test_models_vae(
    model_name=model_name,
    data_path=data_path_test,
    batch_size=10,
    plot=True,
    text="test",
    d3=True,
)
r2, mae = test_models_dft(
    model_name=model_name, data_path=data_path_test, text="test", d3=True, bs=10
)

print(dn)
print(r2, mae * 627)

# %%

n_min, n_gs, _ = dataloader(
    "density",
    model_name=model_name,
    n_instances=n_instances,
    lr=lr,
    diff_soglia=diff_soglia,
    epochs=epochs,
    early_stopping=early_stopping,
    variable_lr=variable_lr,
    n_ensambles=n_ensambles,
)
e_min, e_gs = dataloader(
    "energy",
    model_name=model_name,
    n_instances=n_instances,
    lr=lr,
    diff_soglia=diff_soglia,
    epochs=epochs,
    early_stopping=early_stopping,
    variable_lr=variable_lr,
    n_ensambles=n_ensambles,
)

# %%
n_recons = n_recons.reshape(10, 18, 18, 18)
print(n_std.shape)

# %%
for i in range(n_recons.shape[1]):
    plt.imshow(n_recons[0, i] - n_std[0, i])
    plt.colorbar()
    plt.show()

for i in range(n_recons.shape[1]):
    plt.imshow(n_recons[0, i])
    plt.colorbar()
    plt.show()
    plt.imshow(n_std[0, i])
    plt.colorbar()
    plt.show()


# %%
print(np.sum(n_recons[0]) * (1.5 / 18) ** 3)


# %%
for i in range(10):
    plt.plot(n_min[i])
    plt.plot(n_gs[i])
    plt.show()

# %%
v = np.load("data/dataset_meyer/dataset_meyer_test_256_100.npz")["potential"]

f_gs = e_gs - np.average(n_gs * v[: n_gs.shape[0]], axis=-1)
f_min = e_min - np.average(n_min * v[: n_min.shape[0]], axis=-1)

print(f_gs.shape, f_min.shape)

print(np.average(np.abs(f_gs[0:10] - f_min[0:10])) * 627)
# %%
print(np.abs(f_gs - f_min)[0:10] * 627)
# %%
model = pt.load("model_dft_pytorch/" + model_name, map_location="cpu")
model.eval()

f_ml = (
    model.functional(pt.tensor(n_min, dtype=pt.double)[0:1000])
    .view(-1)
    .detach()
    .numpy()
)

# %%
print(f_ml.shape)
print(np.abs(f[: f_ml.shape[0]] - f_ml) * 627)
# %%
print(np.abs(e_min - e_gs) * 627)

print(np.average(np.abs(e_min - e_gs) * 627))
# %%
plt.plot(n_min[4] - n_gs[4])
# plt.plot(n_gs[4])
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/dataset_speckle_3d/test.npz")

f = data["F"]
n = data["density"]
v = data["potential"]
e = data["energy"]

# %%
# print(f[0], e[0])

# f_new = e - np.einsum("axyz,axyz->a", v, n) * (2 / 18) ** 3

# print(f_new[0])
# # %%
# np.savez("data/dataset_speckle_3d/train.npz", density=n, potential=v, energy=e, F=f_new)
# %%
plt.hist(f, bins=100)
plt.show()
# %%
