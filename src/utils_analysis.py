from typing import List, Dict, Tuple
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
from torchmetrics import R2Score

from src.model import Energy


def dataloader(
    type: str,
    model_name: str,
    n_instances: int,
    lr: int,
    diff_soglia: int,
    n_ensambles: int,
    epochs: int,
    early_stopping: bool,
    variable_lr: bool,
):

    session_name = model_name

    name_istances = f"number_istances_{n_instances}"
    session_name = session_name + "_" + name_istances

    n_initial_name = f"n_ensamble_{n_ensambles}_different_initial"
    session_name = session_name + "_" + n_initial_name

    epochs_name = f"epochs_{epochs}"
    session_name = session_name + "_" + epochs_name

    lr_name = f"lr_{lr}"
    session_name = session_name + "_" + lr_name

    if variable_lr:
        variable_name = "variable_lr"
        session_name = session_name + "_" + variable_name

    if early_stopping:
        diff_name = f"diff_soglia_{diff_soglia}"
        session_name = session_name + "_" + diff_name

    if type == "density":

        data = np.load(
            "data/gradient_descent_data/" + session_name + "_density.npz",
            allow_pickle=True,
        )

        min_n = data["min_density"]
        gs_n = data["gs_density"]
        z = data["z"]
        return min_n, gs_n, z

    elif type == "energy":

        data = np.load(
            "data/gradient_descent_data/"
            + session_name
            + "_energy.npz",
            allow_pickle=True,
        )

        min_eng = data["min_energy"]
        gs_eng = data["gs_energy"]
        return min_eng, gs_eng

    elif type == "history":

        data = np.load(
            "gradient_descent_ensamble_numpy/history_" + session_name + ".npz",
            allow_pickle=True,
        )

        history = data["history"]

        history_n = data["history_n"]

        return history, history_n
