# %%
from src.gradient_descent import GradientDescent
import argparse
import torch
import numpy as np
from src.training.utils import from_txt_to_bool
from src.model import Energy

# %% Data
model_name = "speckle_case/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_16_l1_0.0_l2_0.001"
latent_dimension = 16

# %% N threads


gd = GradientDescent(
    n_instances=100,
    loglr=-1,
    logdiffsoglia=100,
    n_ensambles=1,
    target_path="data/final_dataset/data_test.npz",
    model_name=model_name,
    epochs=50000,
    variable_lr=False,
    final_lr=0.0,
    early_stopping=False,
    L=14,
    resolution=256,
    latent_dimension=latent_dimension,
    seed=232,
    num_threads=3,
    device="cpu",
    init_path="data/final_dataset/data_test.npz",
    Energy=Energy,
    dimension="1D",
)


gd.run()

# %%
