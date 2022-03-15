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
    v = data["potential"]
    eng = data["energy"]

    func = eng - (14 / 256) * np.sum(v * n, axis=1)
    if img is True:
        n = n.reshape(n.shape[0], 1, n.shape[1])
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


def from_txt_to_bool(status: str):
    if status == "True" or status == "true":
        return True
    elif status == "False" or status == "false":
        return False
    else:
        return print("boolean symbol not recognized")


class VaeLoss(nn.Module):
    def __init__(self, variational_beta):

        super().__init__()
        self.variational_beta = variational_beta

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(
            recon_x.view(recon_x.shape[0], -1),
            x.view(x.shape[0], -1),
            reduction="sum",
        )
        kldivergence = -0.5 * pt.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.variational_beta * kldivergence, kldivergence


class VaeLossMSE(nn.Module):
    def __init__(self, variational_beta):

        super().__init__()
        self.variational_beta = variational_beta

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(
            recon_x.view(recon_x.shape[0], -1),
            x.view(x.shape[0], -1),
            reduction="sum",
        )
        kldivergence = -0.5 * pt.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.variational_beta * kldivergence, kldivergence


class ResultsAnalysis:
    def __init__(
        self,
        only_testing: bool,
        n_sample: List,
        n_instances: List,
        n_ensambles: List,
        epochs: List,
        diff_soglia: List,
        models_name: List,
        text: List,
        variable_lr: List,
        early_stopping: List,
        lr: List,
        dx: float,
        postnormalization: bool,
    ):
        self.models_name = models_name
        self.text = text
        self.dx = dx

        self.r_square_list = None
        self.accuracy_vae = None

        self.min_eng = []
        self.gs_eng = []
        self.min_n = []
        self.gs_n = []
        self.min_z = []

        self.z_min = []
        self.z_gs = []

        if not (only_testing):

            for i in range(len(n_sample)):

                x_min = []
                x_gs = []
                y_min = []
                y_gs = []
                z_min = []

                for j in range(len(epochs[i])):

                    min_eng, gs_eng = dataloader(
                        "energy",
                        model_name=models_name[i][j],
                        cut=128,
                        n_instances=n_instances[i][j],
                        lr=lr[i][j],
                        diff_soglia=diff_soglia[i][j],
                        n_ensambles=n_ensambles[i][j],
                        epochs=epochs[i][j],
                        early_stopping=early_stopping[i][j],
                        variable_lr=variable_lr[i][j],
                    )

                    min_n, gs_n, z = dataloader(
                        "density",
                        model_name=models_name[i][j],
                        cut=128,
                        n_instances=n_instances[i][j],
                        lr=lr[i][j],
                        diff_soglia=diff_soglia[i][j],
                        n_ensambles=n_ensambles[i][j],
                        epochs=epochs[i][j],
                        early_stopping=early_stopping[i][j],
                        variable_lr=variable_lr[i][j],
                    )

                    if postnormalization:
                        norm = np.sum(min_n, axis=1) * dx
                        min_n = min_n / norm[:, None]

                    x_min.append(min_eng)
                    x_gs.append(gs_eng)
                    y_min.append(min_n)
                    y_gs.append(gs_n)
                    z_min.append(z)

                self.min_eng.append(x_min)
                self.gs_eng.append(x_gs)
                self.min_n.append(y_min)
                self.gs_n.append(y_gs)
                self.min_z.append(z_min)

    def _comparison(self):

        self.list_de = []
        self.list_devde = []
        self.list_dn = []
        self.list_devdn = []
        self.list_delta_e = []
        self.list_delta_devde = []
        self.list_delta_A = []
        self.list_dev_A = []
        self.list_abs_err_n = []
        self.list_R_square = []
        self.list_R_square_energy = []

        for i in range(len(self.min_eng)):

            av_eng_values = []
            std_eng_values = []
            av_dn_values = []
            std_dn_values = []
            av_eng_valore = []
            std_eng_valore = []
            gradient_min_ns = []
            gradient_gs_ns = []
            delta_gradient_ns = []
            av_delta_gradient_ns = []
            dev_delta_gradient_ns = []
            r_square = []
            r_square_energy = []
            abs_err_n = []
            dn_abs_error = []
            min_engs = []
            gs_engs = []
            min_ns = []
            gs_ns = []
            dns = []
            des = []
            dx = self.dx

            for j in range(len(self.min_eng[i])):
                dns.append(
                    np.sqrt(
                        np.trapz(
                            (self.min_n[i][j] - self.gs_n[i][j]) ** 2,
                            dx=self.dx,
                            axis=1,
                        )
                    )
                    / np.sqrt(np.trapz(self.gs_n[i][j] ** 2, dx=self.dx, axis=1))
                )
                dn_abs_error.append(
                    np.trapz(
                        np.abs(self.min_n[i][j] - self.gs_n[i][j]), dx=self.dx, axis=1
                    )
                )
                gradient_min_ns.append(
                    np.trapz(
                        self.min_n[i][j]
                        * np.gradient(self.min_n[i][j], dx, axis=1) ** 2,
                        dx=self.dx,
                        axis=1,
                    )
                )
                gradient_gs_ns.append(
                    np.trapz(
                        self.gs_n[i][j] * np.gradient(self.gs_n[i][j], dx, axis=1) ** 2,
                        dx=self.dx,
                        axis=1,
                    )
                )
                delta_gradient_ns.append(
                    (
                        np.trapz(
                            (
                                np.gradient(self.min_n[i][j], dx, axis=1)
                                - np.gradient(self.gs_n[i][j], dx, axis=1)
                            )
                            ** 2,
                            dx=self.dx,
                            axis=1,
                        )
                    )
                )
                av_eng_values.append(
                    np.average(
                        np.abs(
                            (self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j]
                        )
                    )
                )
                r_square_energy.append(
                    1
                    - np.sum((self.gs_eng[i][j] - self.min_eng[i][j]) ** 2)
                    / (self.gs_eng[i][j].shape[0] * np.std(self.gs_eng[i][j]) ** 2)
                )
                av_eng_valore.append(
                    np.average(
                        (self.min_eng[i][j] - self.gs_eng[i][j]) / self.min_eng[i][j]
                    )
                )
                std_eng_valore.append(
                    np.std(
                        ((self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j])
                        #    / np.sqrt(self.min_eng[i][j].shape[0] - 1)
                    )
                )
                std_eng_values.append(
                    np.std(
                        np.abs(
                            (self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j]
                        )
                        #    / np.sqrt(self.min_eng[i][j].shape[0] - 1)
                    )
                )

                av_dn_values.append(np.average(dns[j]))
                std_dn_values.append(
                    np.std(dns[j])
                    # / np.sqrt(dns[j].shape[0] - 1)
                )
                av_delta_gradient_ns.append(np.average(delta_gradient_ns[j]))
                dev_delta_gradient_ns.append(
                    np.std(delta_gradient_ns[j])
                    # / np.sqrt(dns[j].shape[0] - 1)
                )
                abs_err_n.append(np.average(dn_abs_error[j]))

            self.list_de.append(av_eng_values)
            self.list_R_square.append(r_square)
            self.list_R_square_energy.append(r_square_energy)
            self.list_devde.append(std_eng_values)
            self.list_dn.append(av_dn_values)
            self.list_devdn.append(std_dn_values)
            self.list_delta_e.append(av_eng_valore)
            self.list_delta_devde.append(std_eng_valore)
            self.list_delta_A.append(av_delta_gradient_ns)
            self.list_dev_A.append(dev_delta_gradient_ns)
            self.list_abs_err_n.append(abs_err_n)

    def plot_results(
        self,
        xticks: List,
        xposition: List,
        yticks: Dict,
        position: List,
        labels: list,
        xlabel: str,
        title: str,
        loglog: bool,
    ):

        self._comparison()

        fig = plt.figure(figsize=(10, 10))
        for i, des in enumerate(self.list_de):
            plt.errorbar(
                x=position[i],
                y=des,
                yerr=self.list_devde[i] / np.sqrt(self.gs_eng[i][0].shape[0] - 1),
                label=labels[i],
                linewidth=3,
            )
        plt.ylabel(r"$\mathbb{E}(|\Delta e|)$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.xticks(labels=xticks, ticks=xposition)
        if yticks != None:
            plt.yticks(yticks["de"])
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.legend(fontsize=15)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, devde in enumerate(self.list_devde):
            plt.plot(
                position[i],
                devde,
                label=labels[i],
                linewidth=3,
            )
        plt.ylabel(r"$\sigma(\Delta e)$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.xticks(labels=xticks, ticks=xposition)
        if yticks != None:
            plt.yticks(yticks["devde"])
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.legend(fontsize=15)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, dn in enumerate(self.list_dn):
            plt.errorbar(
                x=position[i],
                y=dn,
                yerr=self.list_devdn[i] / np.sqrt(self.gs_eng[i][0].shape[0] - 1),
                label=labels[i],
                linewidth=3,
            )
        plt.ylabel(r"$\mathbb{E}(|\Delta n|/|n|)$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.xticks(labels=xticks, ticks=xposition)
        if yticks != None:
            plt.yticks(yticks["dn"])
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.legend(fontsize=15)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, dn in enumerate(self.list_abs_err_n):
            plt.errorbar(
                x=position[i],
                y=dn,
                yerr=self.list_devdn[i] / np.sqrt(self.gs_eng[i][0].shape[0] - 1),
                label=labels[i],
                linewidth=3,
            )
        plt.ylabel(r"$\mathbb{E}(|\Delta n|_{l0}/|n|_{l0})$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.xticks(labels=xticks, ticks=xposition)
        plt.legend(fontsize=15)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, devdn in enumerate(self.list_devdn):
            plt.plot(position[i], devdn, label=labels[i], linewidth=3)
        plt.xlabel(xlabel, fontsize=20)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )

        plt.xticks(labels=xticks, ticks=xposition)
        if yticks != None:
            plt.yticks(yticks["devdn"])
        plt.ylabel(r"$\sigma(\Delta n)$", fontsize=20)
        plt.legend(fontsize=15)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, des in enumerate(self.list_delta_e):
            plt.errorbar(
                x=position[i],
                y=des,
                yerr=self.list_delta_devde[i] / np.sqrt(self.gs_eng[i][0].shape[0] - 1),
                label=labels[i],
                linewidth=3,
            )
        plt.ylabel(r"$\mathbb{E}(\Delta e)$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.xticks(labels=xticks, ticks=xposition)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.legend(fontsize=15)
        plt.title(title)
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, das in enumerate(self.list_delta_A):
            plt.errorbar(
                x=position[i],
                y=das,
                yerr=self.list_dev_A[i] / np.sqrt(self.gs_eng[i][0].shape[0] - 1),
                label=labels[i],
                linewidth=3,
            )
        plt.ylabel(r"$\mathbb{E}(\Delta A[\rho])$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.xticks(labels=xticks, ticks=xposition)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.legend(fontsize=15)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for i, das in enumerate(self.list_R_square_energy):
            plt.plot(position[i], das, label=labels[i], linewidth=3)
        plt.ylabel(r"$R^2 energy$", fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.xticks(ticks=xposition, labels=xticks)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        plt.legend(fontsize=15)
        plt.title(title)
        plt.show()

    def plot_samples(
        self,
        style: list,
        idx: List,
        jdx: List,
        n_samples: int,
        title: str,
        l: float,
        v: np.array,
    ):

        space = np.linspace(0, l, self.min_n[0][0].shape[1])
        for k in range(n_samples):

            fig = plt.figure(figsize=(10, 10))
            ax1 = fig.add_subplot()
            ax2 = ax1.twinx()

            for i in idx:
                for j in jdx:
                    ax1.plot(
                        space,
                        self.min_n[i][j][k],
                        label=f"min (de={self.min_eng[i][j][k]-self.gs_eng[i][j][k]:.3f},dn={np.sum(np.abs(self.min_n[i][j][k]-self.gs_n[i][j][k]))*self.dx:.3f},norm={np.sum(self.min_n[i][j][k])*self.dx:.3f} {self.text[i][j]} )",
                        linewidth=4,
                        linestyle=style[i],
                        alpha=0.8,
                    )
            ax1.plot(
                space,
                self.gs_n[0][0][k],
                linestyle=":",
                alpha=1,
                linewidth=5,
                label="ground state",
            )
            ax2.plot(
                space,
                v[k],
                color="black",
                alpha=1,
                linewidth=3,
                label="ground state",
            )
            plt.xlabel("x", fontsize=20)
            plt.ylabel(r"$n(x)$", fontsize=20)
            ax1.legend(fontsize=15)
            ax2.legend(fontsize=15)
            plt.title(title)
            plt.show()

    def histogram_plot(
        self,
        idx: List,
        jdx: List,
        title: str,
        bins: int,
        density: bool,
        alpha: float,
        hatch: List,
        color: List,
        fill: List,
        range_eng: Tuple,
        range_n: Tuple,
    ):
        dn_overall=[]
        de_overall=[]

        fig = plt.figure(figsize=(10, 10))
        for eni, i in enumerate(idx):
            for enj, j in enumerate(jdx):
                dn = np.sqrt(
                    self.dx*np.sum(
                        (self.min_n[i][j] - self.gs_n[i][j]) ** 2, axis=1
                    )
                )/ np.sqrt(self.dx*np.sum(self.gs_n[i][j] ** 2, axis=1))
                plt.hist(
                    dn,
                    bins,
                    label=self.text[i][j],
                    range=range_n,
                    density=density,
                    alpha=alpha,
                    hatch=hatch[eni][enj],
                    fill=fill[eni][enj],
                    color=color[eni][enj],
                    histtype="step",
                )
                dn_overall.append(dn)
        plt.xlabel(r"$|\Delta n|/|n|$", fontsize=20)
        plt.legend(fontsize=15, loc="best")
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        if title != None:
            plt.title(title)
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        for eni, i in enumerate(idx):
            for enj, j in enumerate(jdx):
                de = (self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j]
                plt.hist(
                    de,
                    bins,
                    label=self.text[i][j],
                    density=density,
                    alpha=alpha,
                    range=range_eng,
                    hatch=hatch[eni][enj],
                    fill=fill[eni][enj],
                    color=color[eni][enj],
                    histtype="step",
                )
                de_overall.append(de)

        plt.xlabel(r"$\Delta e/e$", fontsize=20)
        plt.legend(fontsize=15, loc="upper left")
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=15,
            width=3,
        )
        if title != None:
            plt.title(title)
        plt.show()

        return dn_overall,de_overall

    def test_models_dft(self, idx: List, jdx: List, data_path: str):
        self.r_square_list = []
        r2 = R2Score()

        n_std = np.load(data_path)["density"]
        F_std = np.load(data_path)["F"]
        ds = TensorDataset(pt.tensor(n_std).view(-1, n_std.shape[-1]), pt.tensor(F_std))
        dl = DataLoader(ds, batch_size=100)
        for i in idx:
            for j in jdx:
                model = pt.load(
                    "model_dft_pytorch/" + self.models_name[i][j], map_location="cpu"
                )
                model.eval()
                model = model.to(dtype=pt.double)

                for batch in dl:
                    model.eval()
                    model.r2_computation(batch, device="cpu", r2=r2)
                print(model)
                print(f"# parameters={count_parameters(model)}")
                print(f"R_square_test={r2.compute()} for {self.text[i][j]} \n")

                self.r_square_list.append(r2.compute())
                r2.reset()

    def test_models_vae(
        self, idx: List, jdx: List, data_path: str, batch_size: int, plot: bool
    ):
        self.accuracy_vae = []

        n_std = np.load(data_path)["density"]
        F_std = np.load(data_path)["F"]
        ds = TensorDataset(pt.tensor(n_std).view(-1, 1, n_std.shape[-1]))
        dl = DataLoader(ds, batch_size=batch_size)
        for i in idx:
            for j in jdx:
                model = pt.load(
                    "model_dft_pytorch/" + self.models_name[i][j], map_location="cpu"
                )
                model.eval()
                model = model.to(dtype=pt.double)
                Dn = 0

                for k, batch in enumerate(dl):
                    model.eval()
                    mu, _ = model.Encoder(batch[0].double())
                    n_recon = model.Decoder(mu)
                    n_recon = n_recon.squeeze().detach().numpy()
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
                print(f"Dn={Dn/len(dl)} for {self.text[i][j]} \n")
                self.accuracy_vae.append(Dn / len(dl))

    def z_analysis(self, idx: list, jdx: list):

        dn_i = []
        for i in idx:
            dn_j = []
            zmin=[]
            zgs=[]
            for j in jdx:
                # load the model
                model = pt.load(
                    "model_dft_pytorch/" + self.models_name[i][j], map_location="cpu"
                )
                model.eval()
                model = model.to(dtype=pt.double)

                z_min, _ = model.Encoder(
                    pt.tensor(self.min_n[i][j]).double().unsqueeze(1)
                )
                z_min = z_min.squeeze()
                zmin.append(z_min)
                z_gs, _ = model.Encoder(
                    pt.tensor(self.gs_n[i][j]).double().unsqueeze(1)
                )
                z_gs = z_gs.squeeze()
                zgs.append(z_gs)
                dz = (
                    pt.linalg.norm(z_min - z_gs[: z_min.shape[0]], dim=1)
                    .detach()
                    .numpy()
                    / pt.linalg.norm(z_gs[: z_min.shape[0]]).detach().numpy()
                )
                dn_j.append(dz)
            self.z_gs.append(zgs)
            self.z_min.append(zmin)
            dn_i.append(dn_j)

        return dn_i


            
    def decoding(self, idx: List, jdx: List):
        results = []
        for i in idx:
            result=[]
            for j in jdx:
                # load the model
                model = pt.load(
                    "model_dft_pytorch/" + self.models_name[i][j], map_location="cpu"
                )
                model.eval()
                model = model.to(dtype=pt.double)

                # propose n from z
                x = model.proposal(pt.tensor(self.z_gs[i][j]).double())
                # print(x.shape)
                result.append(x)
            results.append(result)
        return results

    def gs_energy_computation(self, idx: List, jdx: List, v: pt.Tensor,batch:bool):
        results = []
        for i in idx:
            result=[]
            for j in jdx:
                # load the model
                model = pt.load(
                    "model_dft_pytorch/" + self.models_name[i][j], map_location="cpu"
                )
                model.eval()
                model = model.to(dtype=pt.double)

                energy = Energy(model, v=v, dx=self.dx, mu=0)
                # propose n from z
                if batch:
                    eng,_, _ = energy.batch_calculation(pt.tensor(self.z_gs[i][j],dtype=pt.double))
                else:
                    eng,_, _,_ = energy(self.gs_z)                    
                result.append(eng)
            results.append(result)
        return results

    def dn_vs_de(self,idx:List,jdx:List):
        for i in idx:
            for j in jdx:

                dn=np.sqrt( np.sum( (self.min_n[i][j]-self.gs_n[i][j])**2,axis=1) )/np.sqrt( np.sum( (self.gs_n[i][j])**2,axis=1) )
                de=(self.min_eng[i][j]-self.gs_eng[i][j])

                plt.scatter(de,dn,s=20,label=self.text[i][j])
        plt.xlabel(r'$\Delta e$',fontsize=20)
        plt.ylabel(r'$|\Delta n |$',fontsize=20)
        plt.legend(fontsize=15)
        plt.show()

def dataloader(
    type: str,
    model_name: str,
    cut: int,
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
            "gradient_descent_ensamble_numpy/min_density_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_n = data["min_density"]
        gs_n = data["gs_density"]
        z = data["z"]
        return min_n, gs_n, z

    elif type == "energy":

        data = np.load(
            "gradient_descent_ensamble_numpy/min_vs_gs_gradient_descent_"
            + session_name
            + ".npz",
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


def initial_ensamble_random(n_instances: int) -> pt.tensor:
    """This function creates the ensamble of the initial density profiles from a dataset.
    Those functions are average values of a number of subsets of the dataset.

    Argument:

    n: dataset in np.array
    n_istances: number of subsets of the dataset.

    Returns:

    the tensor of the inital values as [n_istances,resolution] in np.array


    """

    np.random.seed(42)
    pt.manual_seed(42)

    L = 14

    resolution = 256

    dx = L / resolution

    n_ensambles = pt.tensor([])

    for j in range(int(n_instances)):

        x = pt.linspace(0, L, resolution)

        # the minimum value of sigma
        # in order to avoid deltas
        min_sigma = (L / 2) * 0.2

        # generates 1, 2 or 3
        # gaussians
        if j < int(n_instances / 2):

            # localized profiles
            n_gauss = 1

        else:

            # delocalized profiles
            n_gauss = 1

        # generates n_gauss parameters
        params = pt.rand(n_gauss)

        # we need to create the rand
        # before the cycle because
        # the seed is fixed
        sigma_rand = pt.rand(n_gauss)
        shift_rand = pt.rand(n_gauss)

        # initialize the sample
        sample = pt.zeros(resolution)

        for i in range(n_gauss):

            # we define the sigma
            sigma = (L / 2) * sigma_rand[i] + min_sigma

            # the gaussian distribution
            # centered in L/2
            gauss = pt.exp((((-1 / (2 * sigma)) * ((L / 2) - x) ** 2)))

            # define a random shift of
            # the average value
            shift = int(resolution * shift_rand[i])
            gauss = pt.roll(gauss, shift)

            # sum the gauss to the
            # sample
            sample = sample + params[i] * gauss

        # normalize the sample
        norm = pt.trapezoid(sample, dx=dx)
        sample = sample / norm

        # reshape to append
        sample = sample.view(1, -1)

        if j == 0:
            # initial value
            n_ensambles = sample

        else:
            # append the other
            # values
            n_ensambles = pt.cat((n_ensambles, sample), dim=0)

    return n_ensambles
