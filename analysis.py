#%%
from src.gradient_descent import SimulatedAnnealing
from src.training.utils import ResultsAnalysis
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
data = np.load(
    "gradient_descent_ensamble_numpy/min_density_emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb_number_istances_500_n_ensamble_1_different_initial_epochs_15000_lr_1.npz"
)

n_min = data["min_density"]
n_gs = data["gs_density"]

#%%
for i in range(n_min.shape[0]):
    plt.plot(n_min[i])
    plt.plot(n_gs[i])

    plt.show()

#%% only for testing
vb = [0.01, 0.001, 0.0001]

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
hparam = ["ACNN"]
labels = ["ACNN"]
yticks = {
    "de": [0.7, 0.1, 0.05, 0.01, 0.005],
    "devde": None,
    "dn": [0.2, 0.1, 0.02, 0.01],
    "devdn": [0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 2000 for i in range(8)]

n_sample = 16
n_hc = len(hparam)
n_instances = [[500] * n_sample] * n_hc
n_ensambles = [[20] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [
    ["emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb"] * n_sample,
]
text = [
    [f"dataset={hc}k epochs={epoch}" for epoch in epochs[i]] * n_sample
    for i, hc in enumerate(hparam)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample]

n_sample = [n_sample] * n_hc

only_testing = False

# Histogram settings
idx = [0]
jdx = [15]
hatch = [["."], ["//"], [None]]
color = [["black"], ["black"], ["red"]]
fill = [[False], [False], [True]]
density = False
alpha = 0.5
range_eng = (-0.04, 0.015)
range_n = (0, 0.2)
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
    position=epochs[0],
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

jdx = [2]

ra.test_models_dft(idx, jdx, "data/final_dataset/data_test.npz")

ra.test_models_vae(idx, jdx, "data/final_dataset/data_test.npz")

# %%

ra.plot_samples(idx, jdx=[15], n_samples=10, title=None, l=14)

# %%
