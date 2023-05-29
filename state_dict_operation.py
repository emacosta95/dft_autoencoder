# %%
import torch
from src.model_dft_autoencoder import DFTVAEnorm, DFTVAEnorm2ndGEN

# the model trained with both DFT and VAE part
model_name = "/meyer_case/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_1.0_l2_0.01"
model = torch.load("model_dft_pytorch" + model_name, map_location="cpu")
state_dict = model.DFTModel.state_dict()
print(model)
# torch.save(
#     state_dict,
#     "state_dict_save/meyer_case_2ndGEN/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_0.0_l2_0.01",
# )

# the model trained only with the VAE part but with the same DFT specifics of the previous model


# # model = torch.load("model_dft_pytorch" + model_name, map_location="cpu")
# model2.Decoder.load_state_dict(state_dict)

# %%
model_name2 = "meyer_case/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_0.0_l2_0.001"
model2 = torch.load("model_dft_pytorch/" + model_name2)
print(model2)
model2.DFTModel.load_state_dict(state_dict)

torch.save(model2, "model_dft_pytorch/" + model_name2)

# %%
