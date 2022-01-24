from typing import List, Dict
import numpy as np
import argparse
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm.notebook import tqdm, trange
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt


def make_data_loader(
    file_name: str, split: float, bs: int, generative: bool, img: bool = False
) -> tuple:
    """
    This function create a data loader from a .npz file

    Arguments

    file_name: name of the npz data_file (numpy format)
    pbc: if True the input data is extended in a periodic fashion with 128 components both on the top and bottom (128+256+128)
    split: the ratio valid_data/train_data
    bs: batch size of the data loader
    img: if True reshape the x data into a one dimensional image        (N_dataset,1,dimension)
    """

    data = np.load(file_name)
    n = data["density"]
    func = data["F"]

    if img is True:
        n = n.reshape(n.shape[0], 1, n.shape[1])
    func = data["F"]
    n_train = int(n.shape[0] * split)

    if generative:
        train_ds = TensorDataset(pt.tensor(n[0:n_train]))
        valid_ds = TensorDataset(pt.tensor(n[n_train:]))
    else:
        train_ds = TensorDataset(pt.tensor(n[0:n_train]), pt.tensor(func[0:n_train]))
        valid_ds = TensorDataset(pt.tensor(n[n_train:]), pt.tensor(func[n_train:]))

    train_dl = DataLoader(train_ds, bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, 2 * bs)
    return train_dl, valid_dl


def get_optimizer(model: pt.nn.Module, lr: int) -> pt.optim.Optimizer:
    """This function fixies the optimizer

    Argument:

    model: the model which should be trained, related to the Optimizer
    lr: learning rate of the optimization process
    """
    opt = pt.optim.Adam(model.parameters(), lr=lr)
    return opt


def count_parameters(model: pt.nn.Module) -> int:
    """Counts the number of trainable parameters of a module
    Arguments:
    param model: model that contains the parameters to count
    returns: the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vae_loss(recon_x, x, mu, logvar):
    variational_beta = 0.1
    recon_loss = F.binary_cross_entropy(
        recon_x.view(recon_x.shape[0], -1),
        x.view(x.shape[0], -1),
        reduction="sum",
    )
    kldivergence = -0.5 * pt.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + variational_beta * kldivergence
