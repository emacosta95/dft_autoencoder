#%% imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.training.utils import ResultsAnalysis

#%% load the model
model = torch.load(
    "model_dft_pytorch/emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb", map_location="cpu"
)
model.eval()
#%% generate instance

z = torch.randn((100, 16), dtype=torch.double)


# %%
n, f = model(z)
print(n.shape)

#%%
for i in range(100):
    plt.plot(n[i].detach().numpy(), label=f[i])
    plt.legend()
    plt.show()
    print((14 / 256) * torch.sum(n[i]))

# %% gd analysis
labels = ["ACNN"]
yticks = {
    "de": [0.3, 0.2, 0.1, 0.05, 0.005],
    "devde": [0.05, 0.03, 0.01, 0.001],
    "dn": [0.2, 0.15, 0.1, 0.05, 0.01],
    "devdn": [0.05, 0.03, 0.02, 0.005],
}
xticks = [i * 2000 for i in range(6)]


n_sample = 11
n_hc = 1
n_instances = [[3] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[1] * n_sample] * n_hc
models_name = [
    ["emodel_20_hc_13_ks_2_ps_16_ls_0.01_vb"] * n_sample,
]
text = [
    [f"emodel epochs={epoch}" for epoch in epochs[0]],
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
lr = [[3] * n_sample] * n_hc
n_sample = [n_sample] * n_hc


#%%
result = ResultsAnalysis(
    n_sample=n_sample,
    n_instances=n_instances,
    n_ensambles=n_ensambles,
    epochs=epochs,
    diff_soglia=diff_soglia,
    models_name=models_name,
    text=text,
    variable_lr=variable_lr,
    early_stopping=early_stopping,
    dx=14 / 256,
    lr=lr,
)


#%% Plot all the main results
result.plot_results(
    xticks=xticks,
    xposition=xticks,
    yticks=None,
    position=epochs[0],
    xlabel="epochs",
    labels=labels,
    title="Evolution comparison between CNN Softplus and ACNN",
    loglog=False,
)


# %% Plot single samples
idx = [0]
jdx = [1]

result.test_models(idx, jdx, "data/final_dataset/data_test.npz")

#%%
result.plot_samples(idx=idx, jdx=jdx, n_samples=3, title="hc comparison", l=14)

# %% Histogram plots
idx = [0]
jdx = [0]
result.histogram_plot(idx, jdx, bins=100, title=None, density=False)
# %% Testing models
model_name = "emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb"

model = torch.load("model_dft_pytorch/" + model_name, map_location="cpu")
model.eval()
n = np.load("data/final_dataset/data_test.npz")["density"][0:5000]

n = torch.from_numpy(n).unsqueeze(1)
latent_mu, latent_logvar = model.Encoder(n)
latent = model._latent_sample(latent_mu, latent_logvar)
x_recon = model.Decoder(latent)

n = n.squeeze()
x_recon = x_recon.squeeze()

# for i in range(100):
#     plt.plot(n.detach().numpy()[i])
#     plt.plot(x_recon.detach().numpy()[i])
#     plt.show()
# #  save the model
# np.savez(
#     f"gen_data/n_{model_name}.npz",
#     n_gen=x_recon.detach().numpy(),
#     n_exact=n.detach().numpy(),
# )

# %% plotting the hist

print(latent_mu.shape)

for i in range(latent_mu.shape[1]):

    plt.hist(latent_mu[:1000, i].detach().numpy(), bins=50, label=f"dimension={i}")
    plt.legend()
    plt.show()
# %% Loss analysis
ls = [32, 64, 128]
vb = [0.01] * len(ls)

loss_val = [
    torch.load(
        f"losses_dft_pytorch/emodel_20_hc_13_ks_2_ps_{ls[i]}_ls_{vb[i]}_vb_loss_valid_generative"
    )
    for i in range(len(ls))
]
loss_train = [
    torch.load(
        f"losses_dft_pytorch/emodel_20_hc_13_ks_2_ps_{ls[i]}_ls_{vb[i]}_vb_loss_train_generative"
    )
    for i in range(len(ls))
]
# plots
fig = plt.figure(figsize=(10, 10))
plt.plot(loss_val[0])
plt.plot(loss_val[1])
plt.plot(loss_val[2])
plt.semilogx()
plt.show()

# %% Analysis densities
data_1 = np.load("gen_data/n_emodel_20_hc_13_ks_2_ps_8_ls_0.001_vb.npz")
data_2 = np.load("gen_data/n_emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb.npz")

n_1 = data_1["n_exact"]
n_gen_1 = data_1["n_gen"]

n_2 = data_2["n_exact"]
n_gen_2 = data_2["n_gen"]

dn1 = np.sqrt(np.sum((n_1 - n_gen_1) ** 2, axis=1)) / np.sqrt(np.sum(n_1 ** 2, axis=1))
dn2 = np.sqrt(np.sum((n_2 - n_gen_2) ** 2, axis=1)) / np.sqrt(np.sum(n_2 ** 2, axis=1))

av_dn1 = np.average(dn1)
av_dn2 = np.average(dn2)

plt.hist(
    (dn1, dn2),
    bins=50,
    label=[f"ls=8 <dn>={av_dn1:.4f}", f"ls=16 <dn>={av_dn2:.4f}"],
)
plt.legend(fontsize=15)
plt.xlabel(r"$|\Delta n|/|n|$", fontsize=20)
plt.show()


# %%
