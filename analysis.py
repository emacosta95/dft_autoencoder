#%%
from src.gradient_descent import SimulatedAnnealing
from src.training.utils import ResultsAnalysis
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/final_dataset/data_test.npz")

v = data["potential"]


#%% only for testing
vb = [10 ** -9]

models_name = [[f"normMSE_20_hc_13_ks_2_ps_16_ls_{v}_vb" for v in vb]]

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
hparam = [1, 20]
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
n_instances = [[104] * n_sample] * n_hc
n_ensambles = [[n_init] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_16_ls_0.001_vb"] * n_sample for v in hparam]
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

#%% Histogram settings
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

#%%
ra.min_n[0][0].shape

# %%

idx = [0, 1]

ra.plot_samples(
    style=["-", "--"], idx=idx, jdx=[30], n_samples=10, title=None, l=14, v=v
)

# %% study the latent space
from src.training.utils import initial_ensamble_random

model = torch.load(
    "model_dft_pytorch/normMSE_20_hc_13_ks_2_ps_16_ls_0.001_vb", map_location="cpu"
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
print(len(dz[1][0]))
plt.hist(
    ([dz[i][0] for i in range(2)]), bins=20, label=["init=1", "init=20"], density=True
)
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

# Study the structure of the latent space

#%% see the difference between z_min and z_gs

print(f"{ra.z_min[0][0]} \n")
print(f"{ra.z_gs[0][0]} \n")
# %% difference for each dimension
dz_vector = torch.abs(ra.z_min[0] - ra.z_gs[0]) / torch.abs(ra.z_gs[0])
dz_vector = dz_vector.detach().numpy()

dz_vector2 = torch.abs(ra.z_min[1] - ra.z_gs[1]) / torch.abs(ra.z_gs[1])
dz_vector2 = dz_vector2.detach().numpy()

print(dz_vector.shape[0])
print(dz_vector.shape[1])

for i in range(16):
    plt.hist(
        (dz_vector[:, i], dz_vector2[:, i]),
        bins=40,
        label=[f"dimension={i} # init={1}", f"dimension={i} # init={20}"],
    )
    plt.legend()
    plt.show()

# %% Spacing the latent space for a single case
index = 1
ls = 11
res = 10

print(f"delta z={ra.z_min[0][index, ls] - ra.z_gs[0][index, ls]}")

zs = torch.linspace(
    0, ra.z_gs[0][index, ls].item() - ra.z_min[0][index, ls].item(), res
)

x_init = ra.decoding(idx=[0], jdx=[0], z=ra.z_min[0])[0].detach().numpy()
x_fin = ra.decoding(idx=[0], jdx=[0], z=ra.z_gs[0])[0].detach().numpy()
plt.plot(x_init[index], label="init", color="black", linestyle="--", linewidth=2)
plt.plot(x_fin[index], label="final", color="red", linestyle="--", linewidth=2)
for z in zs:

    z_propose = ra.z_min[0][index].squeeze().clone()
    z_propose[ls] = z_propose[ls] + z
    # print(z_propose - ra.z_min[0][index].squeeze().clone())
    x = ra.decoding(idx=[0], jdx=[0], z=z_propose)[0].detach().numpy()
    plt.plot(x[0])
plt.legend()
plt.show()

#%% difference between z_mins for both 1 and 20
# initial configuration
dz_vector = torch.abs(ra.z_min[0] - ra.z_min[1]) / torch.abs(ra.z_gs[0])
dz_vector = dz_vector.detach().numpy()

for i in range(16):
    plt.hist(
        (dz_vector[:, i]),
        bins=40,
        label=[f"dimension={i} # init={1}", f"dimension={i} # init={20}"],
    )
    plt.legend()
    plt.show()

plt.hist(np.average(dz_vector, axis=0), bins=40)
plt.xlabel(r"$\Delta z_{min}/ z_{gs}$", fontsize=20)
plt.ylabel("Counts", fontsize=20)
plt.show()


# %% Convex combination
for index in range(100):

    res = 10
    alpha = torch.linspace(0, 1, res)

    z_propose_alpha = (
        ra.z_gs[0][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[0][None, index].squeeze().clone()
    )
    z_propose_beta = (
        ra.z_gs[0][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[1][None, index].squeeze().clone()
    )
    z_propose_gamma = (
        ra.z_min[0][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[1][None, index].squeeze().clone()
    )

    print(z_propose.shape)
    eng_alpha = ra.energy_computation(
        idx=[0],
        jdx=[0],
        z=z_propose_alpha,
        v=torch.tensor(v[index], dtype=torch.double),
    )[0]
    eng_beta = ra.energy_computation(
        idx=[0], jdx=[0], z=z_propose_beta, v=torch.tensor(v[index], dtype=torch.double)
    )[0]
    eng_gamma = ra.energy_computation(
        idx=[0],
        jdx=[0],
        z=z_propose_gamma,
        v=torch.tensor(v[index], dtype=torch.double),
    )[0]

    x = ra.decoding(idx=[0], jdx=[0], z=z_propose)[0].detach().numpy()
    x_2 = ra.decoding(idx=[0], jdx=[0], z=z_propose_2)[0].detach().numpy()
    x_init = ra.decoding(idx=[0], jdx=[0], z=ra.z_min[0])[0].detach().numpy()
    x_fin = ra.decoding(idx=[0], jdx=[0], z=ra.z_gs[0])[0].detach().numpy()
    x_init_2 = ra.decoding(idx=[0], jdx=[0], z=ra.z_min[1])[0].detach().numpy()

    plt.plot(
        x_init[index],
        label=f"init 1 index={index}",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(x_fin[index], label="final", color="red", linestyle="--", linewidth=2)
    for i in range(res):
        plt.plot(x[i], alpha=0.2)
    plt.legend(fontsize=15)
    plt.show()

    plt.plot(
        x_init_2[index],
        label=f"init 20 index={index}",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(x_fin[index], label="final", color="red", linestyle="--", linewidth=2)
    for i in range(res):
        plt.plot(x_2[i], alpha=0.2)
    plt.legend(fontsize=15)
    plt.show()

    plt.plot(
        alpha,
        eng.detach().numpy(),
        label="init 1",
        color="red",
        linestyle=":",
        linewidth=3,
    )
    plt.plot(
        alpha,
        eng_2.detach().numpy(),
        label="init 20",
        color="green",
        linestyle="--",
        linewidth=3,
    )
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel(r"$e(\alpha)$", fontsize=20)
    plt.legend()
    plt.show()
# %%
