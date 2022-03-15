#%%
from src.gradient_descent import SimulatedAnnealing
from src.training.utils import ResultsAnalysis
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/final_dataset_simple_rule/data_test.npz")

v = data["potential"]
e_target = data["energy"]
f=data['F']

#%% only for testing
ls = [16]

models_name = [[f"normMSE_20_hc_13_ks_2_ps_{v}_ls_0.02_vb" ] for v in ls]

n_sample = len(ls)
n_ensambles = None
n_instances = None
epochs = None
diff_soglia = None
text = [[f"ls={v}" ] for v in ls]
variable_lr = None
early_stopping = None
lr = None

only_testing = True

#%% Gradient descent
hparam = [1, 20]
n_hc = len(hparam)
labels = [f"instances={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[n_init] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_4_ls_0.001_vb"] * n_sample for v in hparam]
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

#%% Study with different vb
hparam = [0.1,0.05,0.01,0.001,10**-6]
n_hc = len(hparam)
labels = [f"vb={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_16_ls_{v}_vb"] * n_sample for v in hparam]
text = [
    [f"vb={label} " for epoch in epochs[i]]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False

#%% Study different ls dimension
hparam = [4,8,16,32]
n_hc = len(hparam)
labels = [f"ls={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_{v}_ls_0.001_vb"] * n_sample for v in hparam]
text = [
    [f"ls={label} " for epoch in epochs[i]]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False


#%%
hparam = [10**-6, 0.001]
n_hc = len(hparam)
labels = [f"beta={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_16_ls_{v}_vb"] * n_sample for v in hparam]
text = [
    [f"{label}" for epoch in epochs[i]]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False

#%%
hparam = [20, 60]
n_hc = len(hparam)
labels = [f"hc={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 31
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_{v}_hc_13_ks_2_ps_16_ls_1e-06_vb"] * n_sample for v in hparam]
text = [
    [f"{label}" for epoch in epochs[i]]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False


#%% Histogram settings
idx = [0,1]
jdx = [-1]
hatch = [["."], ["None"], ["//"]]
color = [["black"], ["red"], ["green"]]
fill = [[False], [True], [False]]
density = True
alpha = 0.5
range_eng = (-0.01,0.2)
range_n = (0.0,1.3)
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


#%% DE VS DN

ra.dn_vs_de([0,1],[-1])



#%%
dn_overall,de_overall=ra.histogram_plot(
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

de_local_minima=np.abs(de_overall[0][0:68])-np.abs(de_overall[1][0:68])
dn_local_minima=dn_overall[0][0:68]-dn_overall[1][0:68]

count=dn_local_minima[dn_local_minima>0.05]
p_local=count.shape[0]/dn_local_minima.shape[0]

plt.hist(de_local_minima,bins=20,label=f'p(local minima)={p_local:.3f}')
plt.xlabel(r'$|\Delta e_{init=1}/e|-|\Delta e_{init=20}/e|$',fontsize=20)
plt.legend(fontsize=20)
plt.show()

plt.hist(dn_local_minima,bins=20,label=f'p(local minima)={p_local:.3f}')
plt.xlabel(r'$|\Delta n_{init=1}/|n||-|\Delta n_{init=20}/|n||$',fontsize=20)
plt.legend(fontsize=20)
plt.show()


# %%

idx = [0,1,2,3]

jdx = [0]

ra.test_models_dft(idx, jdx, "data/final_dataset_simple_rule/data_test.npz")

ra.test_models_vae(
    idx, jdx, "data/final_dataset_simple_rule/data_test.npz", batch_size=100, plot=False
)

#%%
ra.min_n[0][0].shape

# %%

idx = [0, 1]

ra.plot_samples(
    style=["-", "--"], idx=idx, jdx=[-1], n_samples=100, title=None, l=14, v=v
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

# %% Energy difference test for the integration rule
idx=[0,1]
jdx=[-1]

dz = ra.z_analysis(idx, jdx)
v=torch.tensor(v[0:105],dtype=torch.double)

eng=ra.energy_computation(
        idx=idx,
        jdx=jdx,
        z=ra.z_gs[idx[0]][jdx[0]],
        v=torch.tensor(v, dtype=torch.double),
        batch=True,
    )

plt.hist(eng[0][0].detach().numpy()-e_target[0:105],bins=40)
plt.hist(eng[0][1].detach().numpy()-e_target[0:105],bins=40)
plt.show()



# %% Energy difference test between prediction and gradient descent

idxs=[0,1]
jdxs=[-1]
list_dde=[]
label=['init=1','init=20']

for idx in idxs:
    for jdx in jdxs:

        eng_pred=ra.energy_computation(
                idx=[idx],
                jdx=[jdx],
                z=ra.z_gs[idx][jdx],
                v=torch.tensor(v, dtype=torch.double),
                batch=True
            )

        eng_gd=ra.energy_computation(
                idx=[idx],
                jdx=[jdx],
                z=ra.z_min[idx][jdx],
                v=torch.tensor(v, dtype=torch.double),
                batch=True,
            )

        plt.hist(eng_pred[0][0].detach().numpy()-e_target[0:eng_pred[0][0].shape[0]],bins=40,label='pred'+ label[idx])
        plt.hist(eng_pred[0][0].detach().numpy()-eng_gd[0][0].detach().numpy(),bins=40,label='gd')
        plt.xlabel(r'$\Delta e$',fontsize=20)
        plt.legend(fontsize=20)
        plt.show()

        dde=(np.abs(eng_pred[0][0].detach().numpy()-e_target[0:105])-np.abs(eng_pred[0][0].detach().numpy()-eng_gd[0][0].detach().numpy()))
        plt.hist(dde,bins=30,label=label[idx])
        plt.legend(fontsize=20)
        plt.xlabel(r'$ |\Delta e_{pred}| - |\Delta e_{gd}| $',fontsize=20)
        plt.show()

        list_dde.append(dde)

#%% DDE analysis versus vb

fig=plt.figure(figsize=(10,10))
for i,dde in enumerate(list_dde):

    plt.hist(dde,bins=50,range=(-0.15,0.07),label=label[i])
plt.legend(fontsize=20)
plt.xlabel(r'$ |\Delta e_{pred}| - |\Delta e_{gd}| $',fontsize=20)
plt.show()

#%%
print(np.arange(105)[(eng_pred[0][0].detach().numpy()-eng_gd[0][0].detach().numpy())<-0.002])

print(ra.min_n[0][-1].shape)
plt.plot(ra.min_n[0][-1][16],label=f'eng={ra.min_eng[0][-1][16]:.4f}')
plt.plot(ra.gs_n[0][0][16],label=f'eng={ra.gs_eng[0][0][16]:.4f}')
plt.legend(fontsize=20)
plt.show()
# %% Convex combination (After Histogram analysis only)
idx=[0,1]
jdx=[-1]

index_set=np.where(dn_local_minima>0.05)
print(index_set)
# Comparison between prediction error and accuracy error
de_ml = []
de_gd = []
for index in range(10):
    print(index)
    res = 10
    alpha = torch.linspace(0, 1, res)
    z_propose_alpha = (
        ra.z_gs[idx[0]][jdx[0]][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[idx[0]][jdx[0]][None, index].squeeze().clone()
    )
    z_propose_beta = (
        ra.z_gs[idx[1]][jdx[0]][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[idx[1]][jdx[0]][None, index].squeeze().clone()
    )
    z_propose_gamma = (
        ra.z_min[idx[1]][jdx[0]][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[idx[0]][jdx[0]][None, index].squeeze().clone()
    )
    eng_1 = ra.energy_computation(
        idx=[0],
        jdx=[0],
        z=z_propose_alpha,
        v=torch.tensor(v[index], dtype=torch.double),
        batch=False,
    )[0][0]
    eng_2 = ra.energy_computation(
        idx=[0],
        jdx=[0],
        z=z_propose_alpha,
        v=torch.tensor(v[index], dtype=torch.double),
        batch=False,
    )[0][1]
    eng_alpha = ra.energy_computation(
        idx=[0],
        jdx=[0],
        z=z_propose_alpha,
        v=torch.tensor(v[index], dtype=torch.double),
        batch=False,
    )[0][0]
    eng_beta = ra.energy_computation(
        idx=[0], jdx=[0], z=z_propose_beta, v=torch.tensor(v[index], dtype=torch.double),batch=False)[0][0]
    eng_gamma = ra.energy_computation(
        idx=[0],
        jdx=[0],
        z=z_propose_gamma,
        v=torch.tensor(v[index], dtype=torch.double),
        batch=False,
    )[0][0]
    de_gd.append(np.abs(eng_alpha[0].detach().numpy() - eng_alpha[-1].detach().numpy()))
    de_ml.append(np.abs(eng_alpha[-1].detach().numpy() - e_target[index]))

    x_alpha = ra.decoding(idx=[0], jdx=[0], z=z_propose_alpha)[0].detach().numpy()
    x_beta = ra.decoding(idx=[0], jdx=[0], z=z_propose_beta)[0].detach().numpy()
    x_init = ra.decoding(idx=[0], jdx=[0], z=ra.z_min[idx[0]][jdx[0]])[0].detach().numpy()
    x_fin = ra.decoding(idx=[0], jdx=[0], z=ra.z_gs[idx[0]][jdx[0]])[0].detach().numpy()
    x_init_2 = ra.decoding(idx=[0], jdx=[0], z=ra.z_min[idx[1]][jdx[0]])[0].detach().numpy()
    plt.plot(
        x_init[index],
        label=f"init 1 index={index}",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        ra.gs_n[0][0][index],
        label=f"exact n index={index}",
        color="green",
        linewidth=4,
        alpha=0.5,
    )
    plt.plot(x_fin[index], label="final", color="red", linestyle="--", linewidth=2)
    for i in range(res):
        plt.plot(x_alpha[i], alpha=0.2)
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
        plt.plot(x_beta[i], alpha=0.2)
    plt.legend(fontsize=15)
    plt.show()

    plt.plot(
        alpha,
        eng_alpha.detach().numpy(),
        label="alpha path 1-> gs",
        color="red",
        linestyle=":",
        linewidth=3,
    )
    plt.axhline(
        y=ra.gs_eng[0][0][index],
        linewidth=3,
        color="red",
        label=f"eng={e_target[index]:.3f}",
    )
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel(r"$e(\alpha)$", fontsize=20)
    plt.plot(
        alpha,
        eng_beta.detach().numpy(),
        label="beta path 20 -> gs",
        color="green",
        linestyle="--",
        linewidth=3,
    )
    plt.xlabel(r"$\beta$", fontsize=20)
    plt.ylabel(r"$e(\beta)$", fontsize=20)
    plt.legend()
    plt.show()

    plt.plot(
        alpha,
        eng_gamma.detach().numpy(),
        label="gamma path 1 -> 20",
        color="black",
        linestyle="--",
        linewidth=3,
    )
    plt.axhline(
        y=ra.gs_eng[0][0][index],
        linewidth=3,
        color="red",
        label=f"eng={e_target[index]:.3f}",
    )
    plt.xlabel(r"$\gamma$", fontsize=20)
    plt.ylabel(r"$e(\gamma)$", fontsize=20)
    plt.legend()
    plt.show()

    plt.plot(
        alpha,
        eng_1.detach().numpy(),
        label="standard",
        color="blue",
        linestyle="--",
        linewidth=3,
    )
    plt.plot(
        alpha,
        eng_2.detach().numpy(),
        label="trapz",
        color="green",
        linestyle="--",
        linewidth=3,
    )
    plt.axhline(
        y=ra.gs_eng[0][0][index],
        linewidth=3,
        color="red",
        label=f"eng={e_target[index]:.3f}",
    )
    plt.xlabel(r"$s$", fontsize=20)
    plt.ylabel(r"$e(s)$", fontsize=20)
    plt.legend()
    plt.show()

# %% Dn_trap vs Dn_sum rule
idx=0
jdx=-1

f_trapz=ra.min_eng[idx][jdx]-np.trapz(v[0:ra.min_eng[idx][jdx].shape[0]]*ra.min_n[idx][jdx],dx=ra.dx)
f_sr=ra.min_eng[idx][jdx]-np.sum(v[0:ra.min_eng[idx][jdx].shape[0]]*ra.min_n[idx][jdx],axis=1)*ra.dx
f=f[0:ra.min_eng[idx][jdx].shape[0]]


plt.hist((f_trapz-f)/f,label='trapz',bins=30,range=(-0.1,0.1))
plt.hist((f_sr-f)/f,label='sum rule',bins=30,range=(-0.1,0.1))
plt.legend(fontsize=20)
plt.xlabel(r'$\Delta F /F$',fontsize=20)
plt.show()

#%% Comparison between The functional error and the potential error

idx=0
jdx=-1

f_sr=ra.min_eng[idx][jdx]-np.sum(v[0:ra.min_eng[idx][jdx].shape[0]]*ra.min_n[idx][jdx],axis=1)*ra.dx
f=f[0:ra.min_eng[idx][jdx].shape[0]]

e_ext=np.sum(v[0:ra.min_eng[idx][jdx].shape[0]]*ra.gs_n[idx][jdx],axis=1)*ra.dx
e_ext_min=np.sum(v[0:ra.min_eng[idx][jdx].shape[0]]*ra.min_n[idx][jdx],axis=1)*ra.dx

plt.hist((e_ext_min-e_ext)/e_ext,label='V part',bins=30,range=(-0.1,0.1))
plt.hist((f_sr-f)/f,label='T part',bins=30,range=(-0.1,0.1))
plt.legend(fontsize=20)
plt.xlabel(r'$\Delta e /e$',fontsize=20)
plt.show()



# %% comparison  between energies

plt.hist([de_ml, de_gd], bins=20, label=["ml", "gd"])
plt.show()
# %% Accuracy VAE vs de and dn
print(len(ra.list_de[0]))
#%%

plt.plot(ra.accuracy_vae,[ ra.list_dn[i][-1] for i in range(4)],marker='o')
plt.xlabel('Accuracy VAE reconstruction',fontsize=20)
plt.ylabel('$|\Delta n|/|n|$',fontsize=20)
plt.show()
# %%
