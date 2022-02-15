#%%
from src.gradient_descent import SimulatedAnnealing
from src.training.utils import ResultsAnalysis
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
data = np.load("data/final_dataset/data_test.npz")

v = data["potential"]


#%% only for testing
vb = [10 ** -6]

models_name = [[f"normMSE_60_hc_13_ks_2_ps_16_ls_{v}_vb" for v in vb]]

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
hparam = [100, 102]
n_hc = len(hparam)
labels = [f"instances={init}" for init in hparam]
yticks = {
    "de": [0.7, 0.1, 0.05, 0.01, 0.005],
    "devde": None,
    "dn": [0.2, 0.1, 0.02, 0.01],
    "devdn": [0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[102] * n_sample, [100] * n_sample]
n_ensambles = [[1] * n_sample] * n_hc
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"emodelMSE_20_hc_13_ks_2_ps_16_ls_{v}_vb"] * n_sample for v in hparam]
text = [
    [f"mode={label} epochs={epoch}" for epoch in epochs[i]]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False

# Histogram settings
idx = [0, 1, 2]
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
    postnormalization=False,
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

ra.test_models_vae(
    idx, jdx, "data/final_dataset/data_test.npz", batch_size=100, plot=False
)

# %%

idx = [0, 1, 2]

ra.plot_samples(
    style=["-", "-", "-"], idx=idx, jdx=[30], n_samples=100, title=None, l=14, v=v
)

# %% study the latent space
from src.training.utils import initial_ensamble_random

model = torch.load(
    "model_dft_pytorch/emodel_60_hc_13_ks_2_ps_16_ls_1e-06_vb", map_location="cpu"
)
model = model.double()
model.eval()

ns = np.load("data/final_dataset/data_train.npz")["density"]
ns = torch.tensor(ns, dtype=torch.double)
for i in range(20):

    idx = torch.randint(0, ns.shape[0], size=(1,))

    if i == 0:
        x_init = ns[idx].view(1, 1, -1)
    else:
        x_init = torch.cat((x_init, ns[idx].view(1, 1, -1)), dim=0)

print(x_init.shape)


z, _ = model.Encoder(x_init.to(dtype=torch.double))
x_recon = model.Decoder(z).squeeze()

for i in range(20):
    print(f"norm={(14/256)*torch.sum(x_recon[i]).item()}")
    plt.plot(x_recon[i].detach().numpy())
    plt.plot(x_init.squeeze()[i].detach().numpy(), linewidth=3, linestyle="--")
    plt.show()

# %%
dz = ra.z_analysis([0, 1], [-1])
dz[0][0].shape
plt.hist((dz[0][0], dz[1][0]), bins=10, label=["hard", "soft"], density=True)
plt.legend(fontsize=15)
plt.xlabel(r"$|\Delta z|/|z|$", fontsize=30)
plt.show()
# %%

from src.gradient_descent import GradientDescent

gd = GradientDescent(
    n_instances=1,
    loglr=1,
    cut=128,
    logdiffsoglia=1,
    n_ensambles=20,
    target_path="data/final_dataset/data_test.npz",
    model_name="emodelMSE_20_hc_13_ks_2_ps_16_ls_0.001_vb",
    epochs=1,
    variable_lr=1,
    final_lr=1,
    early_stopping=1,
    L=14,
    resolution=256,
    latent_dimension=16,
    seed=42,
    num_threads=10,
    device="cpu",
    mu=1,
    init_path="data/final_dataset/data_test.npz",
)


z = gd.initialize_z()

print(z[0], z[1])

# %%
