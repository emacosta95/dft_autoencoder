# %%
from src.gradient_descent import GradientDescent
import argparse
import torch
import numpy as np
from src.training.utils import from_txt_to_bool
from src.model import Energy

# %% Data
model_name = "3d_speckle/DFTVAEnorm3D_hidden_channels_vae_[60, 60, 60]_hidden_channels_dft_[60, 60, 60, 60]_kernel_size_[3, 3, 3]_pooling_size_[2, 2, 2]_latent_dimension_32_l1_1.0_l2_0.001_36k"
latent_dimension = 16

# %% N threads


gd = GradientDescent(
    n_instances=100,
    loglr=-1,
    logdiffsoglia=100,
    n_ensambles=1,
    target_path="data/dataset_speckle_3d/test.npz",
    model_name=model_name,
    epochs=50000,
    variable_lr=False,
    final_lr=0.0,
    early_stopping=False,
    L=2,
    resolution=18,
    latent_dimension=latent_dimension,
    seed=232,
    num_threads=3,
    device="cpu",
    init_path="data/dataset_speckle_3d/test.npz",
    Energy=Energy,
    dimension="3D",
)


gd.run()

# %%
