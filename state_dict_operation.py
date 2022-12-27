#%%
import torch

# the model trained with both DFT and VAE part
model_name = (
    "/meyer_case/cnn_softplus_for_gaussian_191222_60_hc_13_ks_2_ps_16_ls_0.001_vb"
)
model = torch.load("model_dft_pytorch" + model_name, map_location="cpu")
state_dict = model.DFTModel.state_dict()


# the model trained only with the VAE part but with the same DFT specifics of the previous model
model_name = (
    "/meyer_case/cnn_softplus_for_gaussian_231222_60_hc_13_ks_2_ps_16_ls_0.001_vb"
)
model = torch.load("model_dft_pytorch" + model_name, map_location="cpu")
state_dict = model.DFTModel.load_state_dict(state_dict)
torch.save(model, "model_dft_pytorch" + model_name)

# %%
