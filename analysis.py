#%% imports
import torch
import matplotlib.pyplot as plt
import numpy as np

#%% load the model
model = torch.load("model_dft_pytorch/emodel_20_hc_13_ks_2_ps", map_location="cpu")

#%% generate instance

z = torch.randn((100, 16), dtype=torch.double)
print(z)

# %%
n = model.proposal(z)
print(n.shape)

#%%
for i in range(100):
    plt.plot(n[i].detach().numpy())
    plt.show()
    print((14 / 256) * torch.sum(n[i]))

# %% gd analysis
