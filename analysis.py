#%%
from src.gradient_descent import SimulatedAnnealing
from src.training.utils import ResultsAnalysis
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
data = np.load(
    "gradient_descent_ensamble_numpy/min_density_emodel_60_hc_13_ks_2_ps_16_ls_1e-06_vb_number_istances_500_n_ensamble_1_different_initial_epochs_15000_lr_1.npz"
)

n_min = data["min_density"]
n_gs = data["gs_density"]

#%%
for i in range(n_min.shape[0]):
    plt.plot(n_min[i])
    plt.plot(n_gs[i])

    plt.show()

#%% only for testing
vb = [10 ** -6]

models_name = [[f"emodel_20_hc_13_ks_2_ps_16_ls_{v}_vb" for v in vb]]

n_sample = len(vb)
n_ensambles = None
n_instances = None
epochs = None
diff_soglia = None
text = [[f"vb={v}" for v in vb]]
variable_lr = None
early_stopping = None
lr = None

only_testing = True

#%% Gradient descent
hparam = [1, 2]
labels = ["low vb", "high hc"]
yticks = {
    "de": [0.7, 0.1, 0.05, 0.01, 0.005],
    "devde": None,
    "dn": [0.2, 0.1, 0.02, 0.01],
    "devdn": [0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for hc in hparam]
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [
    [f"emodel_20_hc_13_ks_2_ps_16_ls_1e-06_vb"] * n_sample,
    [f"emodel_60_hc_13_ks_2_ps_16_ls_0.001_vb"] * n_sample,
]
text = [
    [f"mode={label} epochs={epoch}" for epoch in epochs[i]] * n_sample
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False

# Histogram settings
idx = [0, 1]
jdx = [30]
hatch = [["."], ["None"], ["//"]]
color = [["black"], ["red"], ["green"]]
fill = [[False], [True], [False]]
density = True
alpha = 0.5
range_eng = (-0.04, 0.015)
range_n = (0, 0.8)
bins = 50

# %%
ra = ResultsAnalysis(
    only_testing=only_testing,
    n_sample=n_sample,
    n_instances=n_instances,
    n_ensambles=n_ensambles,
    epochs=epochs,
    diff_soglia=diff_soglia,
    models_name=models_name,
    text=text,
    variable_lr=variable_lr,
    early_stopping=early_stopping,
    lr=lr,
    dx=14 / 256,
)

# %% Qualitative plots

ra.plot_results(
    xticks=xticks,
    xposition=xticks,
    yticks=yticks,
    position=epochs,
    xlabel="epochs",
    labels=labels,
    title="Evolution comparison between CNN Softplus and ACNN",
    loglog=False,
)


#%%
ra.histogram_plot(
    idx=idx,
    jdx=jdx,
    bins=bins,
    title=title,
    density=density,
    range_eng=range_eng,
    range_n=range_n,
    alpha=alpha,
    hatch=hatch,
    color=color,
    fill=fill,
)

# %%

idx = [0]

jdx = [0]

ra.test_models_dft(idx, jdx, "data/final_dataset/data_test.npz")

ra.test_models_vae(idx, jdx, "data/final_dataset/data_test.npz")

# %%

idx = [0]

ra.plot_samples(idx, jdx=[30], n_samples=100, title=None, l=14)

# %% study the latent space
from src.training.utils import initial_ensamble_random

model = torch.load(
    "model_dft_pytorch/emodel_20_hc_13_ks_2_ps_16_ls_1e-06_vb", map_location="cpu"
)
model = model.double()
model.eval()

x_init = initial_ensamble_random(20).unsqueeze(1)

z, _ = model.Encoder(x_init.to(dtype=torch.double))
x_recon = model.Decoder(z).squeeze()

for i in range(20):
    print(f"norm={(14/256)*torch.sum(x_recon[i]).item()}")
    plt.plot(x_recon[i].detach().numpy())
    plt.plot(x_init.squeeze()[i].detach().numpy(), linewidth=3, linestyle="--")
    plt.show()

# %%
