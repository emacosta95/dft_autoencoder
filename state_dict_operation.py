# %%
import torch
from src.model_dft_autoencoder import DFTVAEnorm, DFTVAEnorm2ndGEN

# the model trained with both DFT and VAE part
model_name = "/meyer_case/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_1.0_l2_0.01"
model = torch.load("model_dft_pytorch" + model_name, map_location="cpu")
state_dict = model.DFTModel.state_dict()

# torch.save(
#     state_dict,
#     "state_dict_save/meyer_case_2ndGEN/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_0.0_l2_0.01",
# )

# the model trained only with the VAE part but with the same DFT specifics of the previous model
model = DFTVAEnorm(
    latent_dimension=8,
    input_channels=1,
    input_size=256,
    padding=6,
    padding_mode="circular",
    kernel_size=13,
    pooling_size=2,
    loss={"None": 0.0},
    output_size=256,
    activation="Softplus",
    hidden_channels_vae=[60, 60, 60, 60, 60],
    hidden_channels_dft=[60, 60, 60],
    dx=1.0 / 256,
    training_restriction="None",
)


# model = torch.load("model_dft_pytorch" + model_name, map_location="cpu")
model.DFTModel.load_state_dict(state_dict)

# %%
model.name_checkpoint(model_directory="meyer_case/", training_description="_loaded")
model_name = model.model_name

torch.save(model, "model_dft_pytorch/" + model_name)

# %%
