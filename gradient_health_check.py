#%%
import time
from ast import increment_lineno
from numpy.lib.mixins import _inplace_binary_method
import torch as pt
import torch.nn as nn
import numpy as np
from src.model import Energy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy import fft, ifft
import random
from torch.distributions.normal import Normal


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


def smooth_grad(grad: pt.tensor, cut: int) -> pt.tensor:
    """This routine is a filter of the gradient function.

    Arguments:

    grad [pt.tensor]: the gradient of the functional respect to phi
    cut [int]: the cutoff in the momentum space (k-space)


    Returns:
        grad[pt.tensor]: the filtered gradient
    """

    grad = grad.detach().numpy()
    grad_fft = fft(grad, axis=1)
    grad_fft[:, cut:128] = 0
    grad_fft[:, -128 : -1 * cut] = 0

    new_grad = ifft(grad_fft, axis=1)
    grad = pt.tensor(np.real(new_grad), dtype=pt.double)

    return grad


class GradientDescent:
    def __init__(
        self,
        n_instances: int,
        loglr: int,
        cut: int,
        beta: float,
        beta_final: float,
        ann_step: int,
        annealing: bool,
        logdiffsoglia: int,
        n_ensambles: int,
        target_path: str,
        model_name: str,
        epochs: int,
        variable_lr: bool,
        final_lr: float,
        early_stopping: bool,
        L: int,
        resolution: int,
        latent_dimension: int,
        seed: int,
        num_threads: int,
        device: str,
    ):

        if self.annealing:
            self.beta = beta
            self.beta_ratio = pt.exp(
                (1 / epochs) * pt.log(pt.tensor(beta_final / beta))
            )
            self.ann_step = ann_step

        self.device = device
        self.num_threads = num_threads
        self.seed = seed

        self.early_stopping = early_stopping
        self.variable_lr = variable_lr

        # two version for different operations
        self.dx_torch = pt.tensor(L / resolution, dtype=pt.double, device=self.device)
        self.dx = L / resolution
        self.latent_dimension = latent_dimension

        self.n_instances = n_instances

        self.loglr = loglr
        if self.early_stopping:
            self.lr = (10 ** loglr) * pt.ones(n_ensambles, device=self.device)
        else:
            self.lr = pt.tensor(10 ** loglr, device=self.device)
        self.cut = cut

        self.epochs = epochs
        self.diffsoglia = 10 ** logdiffsoglia
        self.n_ensambles = n_ensambles

        self.n_target = np.load(target_path)["density"]
        self.v_target = np.load(target_path)["potential"]
        self.e_target = np.load(target_path)["energy"]

        self.model_name = model_name

        if self.variable_lr:
            self.ratio = pt.exp(
                (1 / epochs) * pt.log(pt.tensor(final_lr) / (10 ** loglr))
            )

        # the set of the loaded data
        self.min_engs = np.array([])
        self.min_ns = np.array([])
        self.min_hist = []
        self.min_exct_hist = []
        self.eng_model_ref = np.array([])
        self.grads = np.array([])

        self.min_engs = {}
        self.min_ns = {}
        self.min_hist = {}
        self.min_exct_hist = {}
        self.eng_model_ref = {}
        self.grads = {}
        self.min_z = {}

        self.epochs = epochs

    def run(self) -> None:
        """This function runs the entire process of gradient descent for each instance."""

        # select number of threads
        pt.set_num_threads(self.num_threads)

        # fix the seed
        # Initialize the seed
        pt.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # loading the model
        print("loading the model...")
        model = pt.load(
            "model_dft_pytorch/" + self.model_name,
            map_location=pt.device(self.device),
        )
        model = model.to(device=self.device)
        model.eval()

        # starting the cycle for each instance
        print("starting the cycle...")
        for idx in trange(0, self.n_instances):

            # initialize phi
            z = self.initialize_z()
            print(f"is leaf={z.is_leaf}")

            # compute the gradient descent
            # for a single target sample
            self.annealing(z=z, idx=idx, model=model)

    def _model_test(self, model: nn.Module, idx: int) -> None:
        """This routine is a test for the pytorch model (remember the problem of the different outcomes for the same input data)

        Arguments:

        model[nn.Module]: [the tested pytorch model]

        idx[int]: [the index of the idx-th sample]

        """
        pot = self.v_target[idx]
        e_ref = self.e_target[idx]

        n_ref = self.n_target[idx]
        n_ref = n_ref.reshape(1, 256)

        energy = Energy(model, pt.tensor(pot, device=self.device), self.dx)
        energy.eval()
        eng = energy(pt.tensor(n_ref, device=self.device))
        self.eng_model_ref = np.append(self.eng_model_ref, eng.detach().cpu().numpy())

        print(
            f"energy_model={eng[0].detach().cpu().numpy():.4f},exact_energy={e_ref:.4f}"
        )

    def initialize_z(self) -> pt.tensor:
        """This routine initialize the phis using the average decomposition of the dataset (up to now, the best initialization ever found)
        Returns:
            phi[pt.tensor]: [the initialized phis with non zero gradient]
        """
        # sqrt of the initial configuration
        z = pt.randn((self.n_ensambles, self.latent_dimension))
        # initialize in double and device
        z = z.to(dtype=pt.double)
        z = z.to(device=self.device)
        # make it a leaft
        z.requires_grad_(True)

        return z

    def _single_gradient_descent(
        self, z: pt.Tensor, idx: int, model: nn.Module
    ) -> tuple:
        """This routine compute the gradient descent for an energy functional
        with external potential given by the idx-th instance and kinetic energy functional determined by model.

        Args:
            z (pt.tensor): [the hidden variable]
            idx (int): [index of the instance]
            model (nn.Module): [model which describes the kinetic energy functional]

        Returns:
           eng[np.array] : [the energy values for different initial configurations]
           exact_eng[np.array] : [an estimation of the Von Weiszacker functional]
           phi[pt.tensor] : [the minimum configuration of the run for different initial states]
           history[np.array] : [the histories of the different gradient descents]
           exact_history[np.array] : [the histories of the estimation of the different gradient descents]
        """

        # initialize the single gradient descent

        n_ref = self.n_target[idx]
        pot = pt.tensor(self.v_target[idx], device=self.device)
        energy = Energy(model, pot, self.dx)
        energy = energy.to(device=self.device)

        history = pt.tensor([], device=self.device)
        # exact_history = np.array([])
        eng_old = pt.tensor(0, device=self.device)

        # refresh the lr every time
        if self.early_stopping:
            self.lr = (10 ** self.loglr) * pt.ones(self.n_ensambles, device=self.device)
        else:
            self.lr = pt.tensor(10 ** self.loglr, device=self.device)

        # start the gradient descent
        tqdm_bar = tqdm(range(self.epochs))
        for epoch in tqdm_bar:

            eng, z, n_z, grad = self.gradient_descent_step(energy=energy, z=z)
            diff_eng = pt.abs(eng.detach() - eng_old)

            if self.early_stopping:
                self.lr[diff_eng < self.diffsoglia] = 0

            if self.variable_lr:
                self.lr = self.lr * self.ratio  # ONLY WITH FIXED EPOCHS

            if epoch == 0:
                history = eng.detach().view(1, eng.shape[0])
            elif epoch % 100 == 0:
                history = pt.cat((history, eng.detach().view(1, eng.shape[0])), dim=0)

            eng_old = eng.detach()

            plt.plot(n_z[0], label="n")
            plt.legend()
            plt.show()
            plt.plot(z.detach().cpu().numpy()[0], label="z")
            plt.legend()
            plt.show()
            plt.plot(grad[0], label="grad")
            plt.legend()
            plt.show()

            tqdm_bar.set_description(f"eng={eng}")
            tqdm_bar.refresh()

    def gradient_descent_step(self, energy: nn.Module, z: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain
        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        eng, n = energy(z)
        eng.backward(pt.ones_like(eng))
        print(z.is_leaf)
        with pt.no_grad():
            grad = z.grad.clone()
            print(grad)
            z -= self.lr * (grad)
            z.grad.zero_()
        return (
            eng.clone().detach(),
            z,
            n.detach().cpu().numpy(),
            grad.detach().cpu().numpy(),
        )

    def _annealing_step(self, z: pt.Tensor, energy: nn.Module):

        count = 0
        for step in trange(self.ann_step):

            # # distribution
            distrib = Normal(
                pt.zeros(self.latent_dimension).double(),
                pt.ones(self.latent_dimension).double(),
            )

            # new propose
            new_z = distrib.rsample().to(device=self.device)

            # compute the transition
            # rate
            logp_new = distrib.log_prob(new_z.cpu())
            logp_old = distrib.log_prob(z.cpu())
            # print(f"logp_old={logp_old.sum().item()} ")
            # print(f"logp_new={logp_new.sum().item()} ")
            ratio = logp_old.view(-1).sum() - logp_new.sum()
            # print(f"log_trans={ratio} ")

            # boltzmann
            eng_new, _ = energy(new_z)
            # print(f"eng_new={eng_new.item()} ")
            eng, n_z = energy(z)
            # print(f"eng={eng.item()} \n")
            delta_e = (eng - eng_new) * self.beta
            ratio = delta_e  # ratio.to(device=self.device) + delta_e
            # print(f"log boltz={delta_e.item()} ")

            # norm
            # norm = pt.abs(pt.sum(n_z) * self.dx - 1)
            # print(norm)

            # random number
            w = pt.rand(1, device=self.device)
            # print(f"value={ratio.exp()}")
            if ratio.exp() > w:  # and norm < 0.01:
                z = new_z
                #    print("new sample!")
                count += 1

            # print("arikez \n")

        print(f"accept={count/self.ann_step} \n")

        return eng, z, n_z.detach().cpu().numpy()

    def annealing(self, z: pt.Tensor, idx: int, model: nn.Module) -> tuple:

        # initialize the single gradient descent

        n_ref = self.n_target[idx]
        pot = pt.tensor(self.v_target[idx], device=self.device)
        energy = Energy(model, pot, self.dx)
        energy = energy.to(device=self.device)

        history = pt.tensor([], device=self.device)
        # exact_history = np.array([])
        eng_old = pt.tensor(0, device=self.device)

        tqdm_bar = tqdm(range(self.epochs))
        for epoch in tqdm_bar:

            eng, z, n_z = self._annealing_step(energy=energy, z=z)
            diff_eng = pt.abs(eng.detach() - eng_old)

            if epoch == 0:
                history = eng.detach().view(1, eng.shape[0])
            elif epoch % 100 == 0:
                history = pt.cat((history, eng.detach().view(1, eng.shape[0])), dim=0)

            eng_old = eng.detach()

            plt.plot(n_z[0], label=eng[0].item())
            plt.plot(self.n_target[idx], label=self.e_target[idx])
            plt.legend()
            plt.show()

            self.beta = self.beta_ratio * self.beta
            tqdm_bar.set_description(f"eng={eng.item()}")
            tqdm_bar.refresh()


#%%
gd = GradientDescent(
    annealing=True,
    beta=1,
    beta_final=500,
    ann_step=1000,
    n_instances=1,
    loglr=3,
    cut=128,
    logdiffsoglia=-200,
    n_ensambles=1,
    target_path="data/final_dataset/data_test.npz",
    model_name="emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb",
    epochs=200,
    variable_lr=False,
    final_lr=1,
    early_stopping=False,
    L=14,
    resolution=256,
    latent_dimension=16,
    seed=42,
    num_threads=1,
    device="cuda",
)

gd.run()

# %%
