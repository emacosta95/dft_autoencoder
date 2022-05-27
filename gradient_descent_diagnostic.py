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
from src.training.utils import initial_ensamble_random

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


# %% THE GRADIENT DESCENT CLASS


class GradientDescentDiagnostic:
    def __init__(
        self,
        n_instances: int,
        loglr: int,
        cut: int,
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
        mu: float,
        init_path: str,
    ):

        self.device = device
        self.num_threads = num_threads
        self.seed = seed

        self.early_stopping = early_stopping
        self.variable_lr = variable_lr

        # two version for different operations
        self.dx_torch = pt.tensor(L / resolution, dtype=pt.double, device=self.device)
        self.dx = L / resolution
        self.latent_dimension = latent_dimension
        self.mu = mu

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
        self.init_path = init_path

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

        self.min_engs = {}
        self.min_ns = {}
        self.min_hist = {}
        self.min_exct_hist = {}
        self.eng_model_ref = {}
        self.min_z = {}
        self.total_history=None

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
            history=self._single_gradient_descent(z=z, idx=idx, model=model)
            if idx==0:
                self.total_history=history.unsqueeze(0).detach().numpy()
            else:
                self.total_history=np.append(self.total_history,history.unsqueeze(0).detach().numpy(),axis=0)

            np.save('local_minima_analysis.npz',self.total_history)

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

        energy = Energy(model, pt.tensor(pot, device=self.device), self.dx, self.mu)
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
        norm_check = 1

        # generate a likewise density profiles
        ns = np.load(self.init_path)["density"]
        ns = pt.tensor(ns, dtype=pt.double)
        for i in range(self.n_ensambles):

            idx = pt.randint(0, ns.shape[0], size=(1,))

            if i == 0:
                x_init = ns[idx].view(1, -1)
            else:
                x_init = pt.cat((x_init, ns[idx].view(1, -1)), dim=0)

        # loading the model
        print("loading the model...")
        model = pt.load(
            "model_dft_pytorch/" + self.model_name,
            map_location=pt.device(self.device),
        )
        model = model.to(device=self.device)
        model.eval()

        while norm_check > 0.01:
            # initial configuration from pseudo
            # density profiles
            z, _ = model.Encoder(x_init.unsqueeze(1).to(device=self.device))
            z = z.squeeze(1).detach()

            #provisional for low vb
            if self.n_instances==106:
                z=pt.randn((self.n_ensambles,self.latent_dimension))
            
            # initialize in double and device
            z = z.to(dtype=pt.double)
            z = z.to(device=self.device)
            # make it a leaft
            z.requires_grad_(True)

            n_z = model.proposal(z)
            norm = pt.abs(pt.sum(n_z, dim=1) * self.dx - 1)
            norm_check = pt.max(norm)
            print(norm_check)

        print(z.shape)

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

        pot = pt.tensor(self.v_target[idx], device=self.device)
        energy = Energy(model, pot, self.dx, self.mu)
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

            if self.mu != None:
                eng, z, n_z = self._step_soft(energy=energy, z=z)
            else:
                eng, z, n_z = self._step_hard(energy=energy, z=z)

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

            # if epoch % 1000 == 0:
            #     self.checkpoints(
            #         eng=eng,
            #         n_z=n_z,
            #         idx=idx,
            #         history=history,
            #         epoch=epoch,
            #         z=z,
            #     )
            #     for i in range(history.shape[-1]):
            #         plt.plot(history[:,i].detach().numpy())
            #     plt.show()

            idxmin = pt.argmin(eng)
            # tqdm_bar.set_description(
            #     f"eng={(eng[idxmin]).item():.3f},norm={np.sum(n_z[idxmin],axis=0)*self.dx:.3f}"
            # )
            tqdm_bar.refresh()
    
        return history

    def _step_soft(self, energy: nn.Module, z: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain
        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        # provisional change
        eng, _,_, n = energy(z)
        eng.backward(pt.ones_like(eng))

        with pt.no_grad():
            grad = z.grad
            z -= self.lr * (grad)
            z.grad.zero_()
        return (
            eng.clone().detach(),
            z,
            n.detach().cpu().numpy(),
        )

    def _step(self, energy: nn.Module, z: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain
        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        _, eng, n = energy.soft_constrain(z)
        eng.backward(pt.ones_like(eng))

        with pt.no_grad():
            grad = z.grad
            z -= self.lr * (grad)
            z.grad.zero_()
        return (
            eng.clone().detach(),
            z,
            n.detach().cpu().numpy(),
        )

    def _step_hard(self, energy: nn.Module, z: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain
        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        eng, norm, n = energy(z)
        eng.backward(pt.ones_like(eng), retain_graph=True)

        with pt.no_grad():
            grad_e = z.grad.clone()
            z.grad.zero_()

        norm.backward(pt.ones_like(norm))

        with pt.no_grad():
            grad_n = z.grad.clone()
            z.grad.zero_()

            # chemical potential with the low grad_e
            # approximation
            mu = pt.einsum("ai,ai->a", grad_n, grad_e) / pt.einsum(
                "ai,ai->a", grad_n, grad_n
            )
            z -= self.lr * (grad_e - mu[:, None] * grad_n)

        return (
            eng.clone().detach(),
            z,
            n.detach().cpu().numpy(),
        )

    def checkpoints(
        self,
        eng: np.array,
        n_z: np.array,
        idx: int,
        history: np.array,
        epoch: int,
        z: pt.tensor,
    ) -> None:
        """This function is a checkpoint save.

        Args:
        eng[np.array]: the set of energies for each initial configuration obtained after the gradient descent
        phi[pt.tensor]: the set of sqrt density profiles for each initial configuration obtained after the gradient descent
        idx[int]: the index of the instance
        history[np.array]: the history of the computed energies for each initial configuration
        epoch[int]: the current epoch in which the data are saved
        """

        # initialize the filename
        session_name = self.model_name

        name_istances = f"number_istances_{self.n_instances}"
        session_name = session_name + "_" + name_istances

        n_initial_name = f"n_ensamble_{self.n_ensambles}_different_initial"
        session_name = session_name + "_" + n_initial_name

        epochs_name = f"epochs_{epoch}"
        session_name = session_name + "_" + epochs_name

        lr_name = f"lr_{np.abs(self.loglr)}"
        session_name = session_name + "_" + lr_name

        if self.variable_lr:
            variable_name = "variable_lr"
            session_name = session_name + "_" + variable_name

        if self.early_stopping:
            diff_name = f"diff_soglia_{int(np.abs(np.log10(self.diffsoglia)))}"
            session_name = session_name + "_" + diff_name

        # considering the minimum value
        eng_min = pt.min(eng, axis=0)[0].cpu().numpy()
        idx_min = pt.argmin(eng, axis=0)

        # exact_eng_min = exact_eng.clone()[idx_min].cpu()

        n_z_min = n_z[idx_min]
        history_min = history[:, idx_min]
        z_min = z[idx_min].detach().cpu().numpy()

        # exact_history_min = exact_history[idx_min]
        # append to the values
        if idx == 0:
            self.min_engs[epoch] = eng_min
            self.min_hist[epoch] = history_min.cpu().numpy().reshape(1, -1)

        else:
            self.min_engs[epoch] = np.append(self.min_engs[epoch], eng_min)
            self.min_hist[epoch] = np.append(
                self.min_hist[epoch], history_min.cpu().numpy().reshape(1, -1)
            )

        # self.min_exct_hist.append(exact_history_min)

        if idx == 0:
            self.min_ns[epoch] = n_z_min
            self.min_z[epoch] = z_min
        else:
            self.min_ns[epoch] = np.vstack((self.min_ns[epoch], n_z_min))

        # save the numpy values
        if idx != 0:
            np.savez(
                "gradient_descent_ensamble_numpy/min_vs_gs_gradient_descent_"
                + session_name,
                min_energy=self.min_engs[epoch],
                gs_energy=self.e_target[0 : (self.min_engs[epoch].shape[0])],
            )
            np.savez(
                "gradient_descent_ensamble_numpy/min_density_" + session_name,
                min_density=self.min_ns[epoch],
                gs_density=self.n_target[0 : self.min_ns[epoch].shape[0]],
                z=self.min_z[epoch],
            )

#%%

n_instances=10
loglr=-1
cut=128
logdiffsoglia=-1
n_ensambles=20
target_path='data/final_dataset/data_test.npz'
model_name='normMSE_20_hc_13_ks_2_ps_16_ls_1e-06_vb'
epochs=20000
variable_lr=False
final_lr=0
early_stopping=False
L=14
resolution=256
latent_dimension=16
seed=42
num_threads=10
device='cpu'
mu=0
init_path='data/final_dataset/data_train.npz'



gd = GradientDescentDiagnostic(
    n_instances=n_instances,
    loglr=loglr,
    cut=128,
    logdiffsoglia=logdiffsoglia,
    n_ensambles=n_ensambles,
    target_path=target_path,
    model_name=model_name,
    epochs=epochs,
    variable_lr=variable_lr,
    final_lr=final_lr,
    early_stopping=early_stopping,
    L=L,
    resolution=resolution,
    latent_dimension=latent_dimension,
    seed=seed,
    num_threads=num_threads,
    device=device,
    mu=mu,
    init_path=init_path,
)

#%%

history=gd.run()
# %%
import matplotlib.pyplot as plt
import numpy as np

hist=np.load('history_vb_0.1.npy')

print(hist.shape)
# %%
plt.plot(hist[-3,0:10])
#plt.loglog()
plt.show()
# %%
