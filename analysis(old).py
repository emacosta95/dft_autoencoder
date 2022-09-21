#%%

from src.training.utils import ResultsAnalysis
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/dataset_meyer/dataset_meyer_test_256_100.npz")

v = data["potential"]
e_target = data["energy"]
f=data['F']

pot=v[0:500]
av_pot=np.sqrt(np.sum(pot**2,axis=1)*(14/256))
av_pot=np.average(av_pot)

print(f'av_pot={av_pot:.4f}')


#%% only for testing
ls = [16]

models_name = [[f"meyer_case/cnn_for_gaussian_test_4_60_hc_13_ks_2_ps_16_ls_0.01_vb" ] for v in ls]

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
hparam = [1]
n_hc = len(hparam)
labels = [f"instances={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 4000 for i in range(11)]

n_sample = 30
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[n_init] * n_sample for n_init in hparam]
epochs = [
    [ i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"speckle_case/normMSE_60_hc_13_ks_2_ps_16_ls_1e-06_vb"] * n_sample for v in hparam]
text = [
    [f"{label}" for epoch in epochs[i]]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = True

#%% Gradient descent different epochs
hparam = [30000,15000]
n_hc = len(hparam)
labels = [f"instances={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 3000 for i in range(11)]

n_sample = 30
n_hc = len(hparam)
n_instances = [[500] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [ i * 1000 for i in range(n_sample)],[ i * 1000 for i in range(16)]
]
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_60_hc_13_ks_2_ps_16_ls_1e-06_vb"] * n_sample for v in hparam]
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

#%% Study with different vb
hparam = [0.1,0.001,10**-6]
n_hc = len(hparam)
labels = [f"vb={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 4000 for i in range(11)]

n_sample = 41
n_hc = len(hparam)
n_instances = [[500] * n_sample] * n_hc
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

#%% Study with different ls in evolution
hparam = [4,8,16,32]
n_hc = len(hparam)
labels = [f"vb={init}" for init in hparam]
yticks = {
    "de": [0.5,0.4,0.3, 0.1, 0.05, 0.005],
    "devde": None,
    "dn": [0.8,0.6,0.4,0.2, 0.1, 0.02, 0.01],
    "devdn": [0.3,0.2, 0.1, 0.02, 0.01],
}
xticks = [i * 4000 for i in range(11)]

n_sample = 41
n_hc = len(hparam)
n_instances = [[500] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [i * 1000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_{v}_ls_1e-06_vb"] * n_sample for v in hparam]
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
hparam = ['sample']
n_hc = len(hparam)
labels = [f"sample" for init in hparam]
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [4,8,16,32]
position=[ xticks]*n_hc
n_sample = len(xticks)
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [30000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_{v}_ls_0.001_vb"  for v in xticks]]*n_hc
text = [
    [f"ls={ls} " for ls in xticks]
    for i, label in enumerate(labels)
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[1] * n_sample] * n_hc

n_sample = [n_sample] * n_hc

only_testing = False



#%% Study different vb
hparam = ['sample']
n_hc = len(hparam)
labels = [f"sample" for init in hparam]
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [0.1,0.05,0.01,0.001,10**-6]
position=[ xticks]*n_hc
n_sample = len(xticks)
n_hc = len(hparam)
n_instances = [[105] * n_sample] * n_hc
n_ensambles = [[1] * n_sample for n_init in hparam]
epochs = [
    [30000 for i in range(n_sample)],
] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [[f"normMSE_20_hc_13_ks_2_ps_16_ls_{v}_vb"  for v in xticks]]*n_hc
text = [
    [f"ls={ls} " for ls in xticks]
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
idx = [0,-1]
jdx = [-1]
hatch = [["."], ["None"], ["//"]]
color = [["black"], ["red"], ["green"]]
fill = [[False], [True], [False]]
density = True
alpha = 0.5
range_eng = (-0.01,0.1)
range_eng_l=(-0.01,0.1)
range_n = (0.0,0.7)
range_n_l=(0.0,0.7)
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
    dx=1 / 256,
    postnormalization=False,
    v=v[0:500]
)

# %% Qualitative plots

ra.plot_results(
    xticks=xticks,
    xposition=xticks,
    yticks=yticks,
    position=epochs,
    xlabel=r"$t$",
    labels=labels,
    title="Evolution comparison between CNN Softplus and ACNN",
    loglog=True,
)

#%%
print(ra.list_de_ml)



#%%
dn_overall,de_overall=ra.histogram_plot(
    idx=idx,
    jdx=jdx,
    bins=bins,
    title=title,
    density=density,
    range_eng=range_eng,
    range_n=range_n,
    range_eng_l=range_eng_l,
    range_n_l=range_n_l,
    alpha=alpha,
    hatch=hatch,
    color=color,
    fill=fill,
)







#%% effect of multiple initial configuration
fig=plt.figure(figsize=(10,10))
xs=[]
ys=[]
for idx in [0,-1]:
    for jdx in [-1]:
        dn_l=np.sqrt(np.sum((ra.recon_n[idx][jdx]-ra.min_n[idx][jdx])**2,axis=1))/np.sqrt(np.sum(ra.recon_n[idx][jdx],axis=1))
        de_l=np.abs(ra.r_eng[idx][jdx]-ra.min_eng[idx][jdx])/ra.r_eng[idx][jdx]

        #plots of convergence in gradient descent (local) for different initial conf
        plt.scatter(dn_l,de_l,label=ra.text[idx][jdx])

        xs.append(dn_l)
        ys.append(de_l)

for i in range(dn_l.shape[0]):
    plt.plot([xs[j][i] for j in range(2)],[ys[j][i] for j in range(2)],linestyle='--',alpha=0.5,color='orange')
    plt.plot([xs[j][i] for j in range(2)],[ys[j][i] for j in range(2)],linestyle='--',alpha=0.5,color='orange')



plt.xlabel(r'$|\Delta n_l|/|n|$',fontsize=20)
plt.ylabel(r'$|\Delta e_l|/|e|$',fontsize=20)
plt.legend(fontsize=20)
plt.show()

#%% Show the suspect instances

idx=0
jdx=-1

# Comparison between prediction error and accuracy error
de_ml = []
de_gd = []
for index in range(500):
    print(index)
    res = 100
    alpha = torch.linspace(0, 1, res)
    z_propose_alpha = (
        ra.z_gs[idx][jdx][None, index].squeeze().clone() * alpha[:, None]
        + (1 - alpha[:, None]) * ra.z_min[idx][jdx][None, index].squeeze().clone()
    )
    eng_alpha = ra.energy_computation(
        idx=[idx],
        jdx=[idx],
        z=z_propose_alpha,
        v=torch.tensor(v[index], dtype=torch.double),
        batch=False,
    )[0][0]
    
    
    x_alpha = ra.decoding_z(idx=[idx], jdx=[jdx], z=z_propose_alpha)[0][0]
    x_init = ra.decoding_z(idx=[idx], jdx=[jdx], z=ra.z_min[idx][jdx])[0][0]
    x_fin = ra.decoding_z(idx=[idx], jdx=[jdx], z=ra.z_gs[idx][jdx])[0][0]
    x_init_2 = ra.decoding_z(idx=[0], jdx=[0], z=ra.z_min[idx][jdx])[0][0]
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
        alpha,
        eng_alpha.detach().numpy(),
        label="alpha path 1-> gs",
        color="red",
        linestyle=":",
        linewidth=3,
    )
    # plt.axhline(
    #     y=eng_ml[idx][jdx][index],
    #     linewidth=3,
    #     color="black",
    #     label=f"eng={eng_ml[index]:.3f}",
    # )
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel(r"$e(\alpha)$", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
    
    






# %% GAUSSIAN CASE

idx = [0]

jdx = [0]

ra.test_models_dft(idx, jdx, "data/dataset_meyer/dataset_meyer_test_256_100.npz")

ra.test_models_vae(
    idx, jdx, "data/dataset_meyer/dataset_meyer_test_256_100.npz", batch_size=100, plot=True
)

# %% SPECKLE CASE
idx = [0]

jdx = [0]

ra.test_models_dft(idx, jdx, "data/final_dataset/data_test.npz")

ra.test_models_vae(
    idx, jdx, "data/final_dataset/data_test.npz", batch_size=100, plot=True
)


#%%
ra.min_n[0][0].shape

# %%

idx = [-2, -1]

ra.plot_samples(
    style=["-", "--"], idx=idx, jdx=[-1], n_samples=500, title=None, l=14, v=v
)

# %% study the latent space
from src.training.utils import initial_ensamble_random

model = torch.load(
    "model_dft_pytorch/normMSE_20_hc_13_ks_2_ps_16_ls_1e-06_vb", map_location="cpu"
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

idx=-1
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


print(np.average(np.abs(f-f_sr))*627)



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
idx=2

print(ra.ml_eng[0][0][idx],ra.gs_eng[0][0][idx])
# %%

print(np.average(np.abs(ra.ml_eng[0][0]-ra.gs_eng[0][0])/ra.gs_eng[0][0]))
# %% Data space correlation analysis

data_speckle=np.load('data/final_dataset/data_test.npz')
data_ising=np.load('data/ising_dataset/valid_sequential_periodic_64_l_0.5_h_0.5_delta_15000_n.npz')
n_speckle=data_speckle['density']
n=data_ising['density']


#%%correlations

cnn_ising=np.einsum('ia,ib->ab',n,n)
cnn_ising=(cnn_ising-np.average(n,axis=0)[None,:]*np.average(n,axis=0)[:,None])/np.einsum('ia,ia->a',n,n)[None,:]

plt.imshow(cnn_ising)
plt.colorbar()
plt.show()

#%%

cnn_speckle=np.einsum('ia,ib->ab',n_speckle,n_speckle)
cnn_speckle=(cnn_speckle-np.average(n_speckle,axis=0)[None,:]*np.average(n_speckle,axis=0)[:,None])/np.einsum('ia,ia->a',n_speckle,n_speckle)[None,:]

plt.imshow(cnn_speckle)
plt.colorbar()
plt.show()


# %%
plt.plot(cnn_speckle[0],label='speckle')
plt.plot(cnn_ising[0],label='ising')
plt.legend(fontsize=10)
plt.show()

# %%
import torch

n=torch.from_numpy(n)

model=torch.load('model_dft_pytorch/isingVAE_h_2.4_20_hc_5_ks_2_ps_16_ls_0.0001_vb',map_location='cpu')


n_recon=model.reconstruct(n[0:10].double())
#%%
for i in range(10):
    plt.plot(n_recon[i].detach().numpy())
    plt.plot(n[i].detach().numpy())
    plt.show()
# %% 
import numpy as np

n = np.genfromtxt('data/speckle3DV5/densityprofile.dat') 

print(n.shape)


# %%
n=n.reshape(-1,4)

x=n[:,0]
y=n[:,1]
z=n[:,2]
density=n[:,3]


# %%
x=x.reshape(20000,18,18,18)
print(x.shape)
# %%
density=density.reshape(20000,18,18,18)
print(density.shape)
# %%
v = np.genfromtxt('data/speckle3DV5/speckle3D.dat') 

print(v.shape)

x=v[:,0]
y=v[:,1]
z=v[:,2]
potential=v[:,3]
# %%
potential=potential.reshape(-1,18,18,18)
print(potential.shape)
# %%
e=np.genfromtxt('data/speckle3DV5/eigenvalues.dat')
energy=e[:,1]

# %%
print(energy.shape)


f=energy-np.average(density*potential,axis=(1,2,3))*8

print(f[0:10])
print(energy[0:10])
# %%
ratio=0.8
n_ratio=int(ratio*f.shape[0])
print(n_ratio)
f_train=f[:n_ratio]
f_test=f[n_ratio:]

v_train=potential[:n_ratio]
v_test=potential[n_ratio:]

e_train=energy[:n_ratio]
e_test=energy[n_ratio:]

n_train=density[:n_ratio]
n_test=density[n_ratio:]

np.savez('data/dataset_speckle_3d/train.npz',energy=e_train,F=f_train,density=n_train,potential=v_train)
np.savez('data/dataset_speckle_3d/test.npz',energy=e_test,F=f_test,density=n_test,potential=v_test)

# %%
