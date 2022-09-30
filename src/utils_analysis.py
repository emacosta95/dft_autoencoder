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
from src.training.utils import count_parameters
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

def test_models_dft(model_name, data_path: str,text:str):
        r2 = R2Score()
        n_std = np.load(data_path)["density"]
        F_std = np.load(data_path)["F"]
        ds = TensorDataset(pt.tensor(n_std).view(-1, n_std.shape[-1]), pt.tensor(F_std))
        dl = DataLoader(ds, batch_size=100)
        model = pt.load(
            "model_dft_pytorch/" + model_name, map_location="cpu"
        )
        model.eval()
        model = model.to(dtype=pt.double)

        mae_ave=0
        for batch in dl:
            model.eval()
            model.r2_computation(batch, device="cpu", r2=r2)
            model.to(device='cpu')
            x,f=batch
            f_ml=model.functional(x).view(-1)
            mae=pt.mean(pt.abs(f-f_ml)).item()
            mae_ave+=mae

        print(model)
        print(f"# parameters={count_parameters(model)}")
        print(f"R_square_test={r2.compute()} for {text} \n")

        r_square=r2.compute()
        r2.reset()

        return r_square,mae_ave/len(dl)

def test_models_vae(model_name, data_path: str, batch_size: int, plot: bool,text:str
):

    n_std = np.load(data_path)["density"]
    F_std = np.load(data_path)["F"]
    ds = TensorDataset(pt.tensor(n_std).view(-1, 1, n_std.shape[-1]))
    dl = DataLoader(ds, batch_size=batch_size)

    model = pt.load(
        "model_dft_pytorch/" + model_name, map_location="cpu"
    )
    model.eval()
    model = model.to(dtype=pt.double)
    Dn = 0

    for k, batch in enumerate(dl):
        model.eval()
        mu, _ = model.Encoder(batch[0].double())
        n_recon = model.Decoder(mu)
        n_recon = n_recon.squeeze().detach().numpy()
        print(n_recon.shape)
        print(batch[0].shape)
        if plot:
            plt.plot(
                batch[0][0].detach().squeeze().numpy(), label="original"
            )
            plt.plot(n_recon[0], label="reconstruction")
            plt.legend(fontsize=20)
            plt.show()
        dn = np.sqrt(
            np.sum(
                (n_recon - batch[0].detach().squeeze().numpy()) ** 2, axis=1
            )
        ) / np.sqrt(
            np.sum((batch[0].detach().squeeze().numpy()) ** 2, axis=1)
        )
        Dn += np.average(dn)

        print(model)
        print(f"# parameters={count_parameters(model)}")
        print(f"Dn={Dn/len(dl)} for {text} \n")
        accuracy_vae=(Dn / len(dl))

        return accuracy_vae
