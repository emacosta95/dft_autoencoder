# %%
import time
from ast import increment_lineno
from numpy.lib.mixins import _inplace_binary_method
import torch as pt
import torch.nn as nn
import numpy as np
from src.model import Energy
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from scipy import fft, ifft
import random
from torch.distributions.normal import Normal
from src.training.utils import initial_ensamble_random
from src.gradient_descent import norm_check_ND, compute_the_norm

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
        logdiffsoglia: int,
        n_ensambles: int,
        target_path: str,
        model_name: str,
        epochs: int,
        variable_lr: bool,
        final_lr: float,
        early_stopping: bool,
        L: int,
        dimension: str,
        resolution: int,
        latent_dimension: int,
        seed: int,
        num_threads: int,
        device: str,
        init_path: str,
        Energy: nn.Module,
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

        self.dimension = dimension
        self.Energy = Energy
        self.n_instances = n_instances

        self.loglr = loglr
        if self.early_stopping:
            self.lr = (10**loglr) * pt.ones(n_ensambles, device=self.device)
        else:
            self.lr = pt.tensor(10**loglr, device=self.device)

        self.epochs = epochs
        self.diffsoglia = 10**logdiffsoglia
        self.n_ensambles = n_ensambles

        self.n_target = np.load(target_path)["density"]
        self.v_target = np.load(target_path)["potential"]
        self.e_target = np.load(target_path)["energy"]
        self.f_target = np.load(target_path)["F"]
        self.init_path = init_path

        self.model_name = model_name

        if self.variable_lr:
            self.ratio = pt.exp(
                (1 / epochs) * pt.log(pt.tensor(final_lr) / (10**loglr))
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

        # initialize energy (we can put this on a getter in a next version)
        energy = self.Energy(model=model, dx=self.dx, dimension=self.dimension)

        # starting the cycle for each instance
        print("starting the cycle...")
        for idx in trange(0, self.n_instances):
            # initialize phi
            z = self.initialize_z()
            print(f"is leaf={z.is_leaf}")

            # compute the gradient descent
            # for a single target sample
            z, n_z, eng = self._single_gradient_descent(z=z, idx=idx, energy=energy)

        return z, n_z, eng

    def _model_test(self, model: nn.Module, idx: int) -> None:
        """This routine is a test for the pytorch model (remember the problem of the different outcomes for the same input data)

        Arguments:

        model[nn.Module]: [the tested pytorch model]

        idx[int]: [the index of the idx-th sample]

        """
        pot = self.v_target[idx]
        e_ref = self.e_target[idx]

        n_ref = self.n_target[idx]

        energy = self.Energy(
            model, pt.tensor(pot, device=self.device), self.dx, self.mu
        )
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
                x_init = ns[idx].unsqueeze(0)
            else:
                x_init = pt.cat((x_init, ns[idx].unsqueeze(0)), dim=0)

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
            z, _ = model.Encoder(x_init.to(device=self.device))
            z = z.squeeze(1).detach()

            # initialize in double and device
            z = z.to(dtype=pt.double)
            z = z.to(device=self.device)
            # make it a leaft
            z.requires_grad_(True)

            n_z = model.proposal(z)
            norm = norm_check_ND((n_z), self.dx, dimension=self.dimension)
            norm_check = pt.max(norm)
            print(norm_check)

        return z

    def _single_gradient_descent(
        self, z: pt.Tensor, idx: int, energy: nn.Module
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
        energy = energy.to(device=self.device)

        history = pt.tensor([], device=self.device)
        # exact_history = np.array([])
        eng_old = pt.tensor(0, device=self.device)

        # refresh the lr every time
        if self.early_stopping:
            self.lr = (10**self.loglr) * pt.ones(self.n_ensambles, device=self.device)
        else:
            self.lr = pt.tensor(10**self.loglr, device=self.device)

        # start the gradient descent
        tqdm_bar = tqdm(range(self.epochs))
        n_old = 200
        for epoch in tqdm_bar:
            eng, z, n_z = self._step(energy=energy, z=z, v=pot)
            diff_eng = pt.abs(eng.detach() - eng_old)
            if self.early_stopping:
                self.lr[diff_eng < self.diffsoglia] = 0

            if self.variable_lr:
                self.lr = self.lr * self.ratio  # ONLY WITH FIXED EPOCHS

            # Meyer's Early stopping
            # print(n_z.shape)
            dn = compute_the_norm(
                (np.abs(n_z[0] - self.n_target[idx])),
                dx=self.dx,
                dimension="3D",
            )

            if epoch == 0:
                history = eng.detach().view(1, eng.shape[0])
            elif epoch % 100 == 0:
                history = pt.cat((history, eng.detach().view(1, eng.shape[0])), dim=0)

            dn_t = np.sum(np.abs(n_z[0] - n_old)) * (self.dx)
            if dn_t < 10**-6 * self.lr:
                self.lr = 0.0
            n_old = n_z[0]

            eng_old = eng.detach()

            if epoch % 1000 == 0:
                self.checkpoints(
                    eng=eng,
                    n_z=n_z,
                    idx=idx,
                    history=history,
                    epoch=epoch,
                    z=z,
                )

            idxmin = pt.argmin(eng)
            f_ml = eng[idxmin].item() - compute_the_norm(
                (self.v_target[idx] * n_z[idxmin]),
                dx=self.dx,
                dimension="3D",
            )
            tqdm_bar.set_description(
                f"df={f_ml-self.f_target[idx]:.5f} eng={(eng[idxmin]).item()-self.e_target[idx]:.5f},norm={compute_the_norm((n_z[idxmin]),dx=self.dx,dimension='3D'):.5f}, dn={dn:.7f} dn_t={dn_t:.9f}"
            )
            tqdm_bar.refresh()

        return z, n_z, eng

    def _step(self, energy: nn.Module, z: pt.Tensor, v: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain
        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        eng, n = energy(z, v=v)
        eng.backward(pt.ones_like(eng), retain_graph=True)
        with pt.no_grad():
            grad_e = z.grad.clone()
            z -= self.lr * (grad_e)
            z.grad.zero_()
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
                "data/gradient_descent_data/" + session_name + "_energy",
                min_energy=self.min_engs[epoch],
                gs_energy=self.e_target[0 : (self.min_engs[epoch].shape[0])],
            )
            np.savez(
                "data/gradient_descent_data/" + session_name + "_density",
                min_density=self.min_ns[epoch],
                gs_density=self.n_target[0 : self.min_ns[epoch].shape[0]],
                z=self.min_z[epoch],
            )


# important instances == 12,
idx_instance = 0
n_instances = 1
loglr = 0

logdiffsoglia = -2
n_ensambles = 1
target_path = "data/dataset_speckle_3d/test.npz"
# target_path='data/final_dataset/data_test.npz'
model_name = "3d_speckle/DFTVAEnorm3D_hidden_channels_vae_[60, 60, 60]_hidden_channels_dft_[60, 60, 60, 60]_kernel_size_[3, 3, 3]_pooling_size_[2, 2, 2]_latent_dimension_48_l1_0.0_l2_1e-07_36k"
# model_name = "meyer_case/cnn_softplus_for_gaussian_test_5_60_hc_13_ks_2_ps_5_ls_0.1_vb"
# model_name='speckle_case/normMSE_60_hc_13_ks_2_ps_16_ls_1e-06_vb'
epochs = 30000
variable_lr = False
final_lr = 10
early_stopping = False
L = 2
resolution = 18
latent_dimension = 32
seed = 42
num_threads = 10
device = "cpu"
mu = 0
init_path = "data/dataset_speckle_3d/test.npz"
# init_path='data/final_dataset/data_train.npz'


gd = GradientDescentDiagnostic(
    n_instances=n_instances,
    loglr=loglr,
    logdiffsoglia=logdiffsoglia,
    n_ensambles=n_ensambles,
    target_path=target_path,
    model_name=model_name,
    epochs=epochs,
    variable_lr=variable_lr,
    final_lr=final_lr,
    early_stopping=early_stopping,
    Energy=Energy,
    L=L,
    resolution=resolution,
    latent_dimension=latent_dimension,
    seed=seed,
    num_threads=num_threads,
    device=device,
    init_path=init_path,
    dimension="3D",
)

# %%

z, n_z, eng = gd.run()


# %% Let's try to import weights from a model to another
# Landscape creation
box_inf_0 = -3
box_sup_0 = 3
box_inf_1 = -3
box_sup_1 = 3
res = 128
dx = 1 / 256

z_0 = pt.linspace(box_inf_0, box_sup_0, res)
z_1 = pt.linspace(box_inf_1, box_sup_1, res)
model = pt.load("model_dft_pytorch/" + model_name, map_location="cpu")
model.eval()

x = pt.tensor(gd.n_target[idx_instance].reshape(1, 1, -1), dtype=pt.double)

z_exact = model.Encoder(x)[0]
x_recon = model.Decoder(z_exact).detach().numpy().reshape(-1)

z_exact = z_exact[0].detach().numpy()
print(z_exact.shape)

plt.plot(gd.n_target[0])
plt.plot(n_z[0])
plt.show()

# %%
v = np.load(target_path)["potential"]
v = pt.tensor(v, dtype=pt.double)

energy = Energy(model, dx, dimension="1D")

z = pt.zeros((res, res, 2), dtype=pt.double)
for i in trange(res):
    for j in range(res):
        z[i, j, 0] = z_0[i]
        z[i, j, 1] = z_1[j]

z = z.view(-1, 2)
batch = res
nbatch = int(z.shape[0] / batch)

for i in trange(nbatch):
    if i == 0:
        eng_imshow = (
            energy(
                z[i * batch : (i + 1) * batch],
                v[idx_instance][i * batch : (i + 1) * batch],
            )[0]
            .detach()
            .numpy()
        )
    else:
        eng_imshow = np.append(
            eng_imshow,
            energy(z[i * batch : (i + 1) * batch])[0].detach().numpy(),
            axis=0,
        )

eng_imshow = eng_imshow.reshape(res, res)


# %% plot the surface
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(figsize=(20, 20))


x, y = np.meshgrid(z_0, z_1)

print(gd.history_z.shape)
print(gd.history_z[0:10])
surf = ax.contour(
    x, y, eng_imshow, 200, cmap=cm.coolwarm, linewidth=0, antialiased=False
)

fig.colorbar(surf, shrink=3, aspect=2)
plt.tick_params(
    top=True,
    right=True,
    labeltop=False,
    labelright=False,
    direction="in",
    labelsize=30,
    width=5,
)
ax.scatter(gd.history_z[-1, 0], gd.history_z[-1, 1], color="blue", marker="*", s=5000)
ax.scatter(gd.history_z[:, 0], gd.history_z[:, 1], color="red", marker="o", s=100)
plt.scatter(z_exact[0], z_exact[1], marker="*", s=5000, color="green")
plt.show()


# %%
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": "3d"})
surf = ax.plot_surface(
    x, y, eng_imshow, cmap=cm.coolwarm, linewidth=0, antialiased=False
)
ax.view_init(elev=20.0, azim=200)
plt.show()
# %%
# latent space 1d

# %% Let's try to import weights from a model to another
# Landscape creation
a = -40
b = 40
res = 5000
dx = 1 / 256

z_0 = pt.linspace(a, b, res)
model = pt.load("model_dft_pytorch/" + model_name, map_location="cpu")
model.eval()

x = pt.tensor(gd.n_target[idx_instance].reshape(1, 1, -1), dtype=pt.double)

z_exact = model.Encoder(x)[0]
x_recon = model.Decoder(z_exact).detach().numpy().reshape(-1)

z_exact = z_exact[0].detach().numpy()
print(z_exact.shape)

plt.plot(gd.n_target[idx_instance])
plt.plot(x_recon)
plt.show()

v = np.load(target_path)["potential"]
v = pt.tensor(v, dtype=pt.double)

energy = Energy(model, v[idx_instance], dx, mu=0)

z = z_0.double()
z = z.view(-1, 1)
batch = res
nbatch = int(z.shape[0] / batch)

for i in trange(nbatch):
    if i == 0:
        eng = energy(z[i * batch : (i + 1) * batch])[0].detach().numpy()
    else:
        eng = np.append(
            eng_imshow,
            energy(z[i * batch : (i + 1) * batch])[0].detach().numpy(),
            axis=0,
        )

eng = eng.reshape(res)

# %%
plt.plot(z, eng)
plt.axvline(z_exact, linewidth=2, linestyle="--", color="red")
plt.axvline(gd.history_z[-1], label="stop", color="black")
plt.axvline(gd.history_z[0], label="start", color="green")

plt.show()

# %%
