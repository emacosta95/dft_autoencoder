from src.gradient_descent import GradientDescent
import argparse
import torch
import numpy as np
from src.training.utils import from_txt_to_bool
from src.model import Energy

gd = GradientDescent(
    n_instances=100,
    loglr=-1,
    logdiffsoglia=100,
    n_ensambles=1,
    target_path="data/dataset_meyer/dataset_meyer_test_256_100.npz",
    model_name="meyer_case/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_0.0_l2_0.001",
    epochs=15000,
    variable_lr=False,
    final_lr=0.0,
    early_stopping=False,
    L=1,
    resolution=256,
    latent_dimension=8,
    seed=232,
    num_threads=10,
    device="cpu",
    init_path="data/dataset_meyer/dataset_meyer_test_256_100.npz",
    Energy=Energy,
    dimension="1D",
)


gd.run()
