from src.gradient_descent import GradientDescent
import argparse
import torch
import numpy as np
from src.training.utils import from_txt_to_bool
from src.model import Energy


parser = argparse.ArgumentParser()

parser.add_argument(
    "--latent_dimension",
    type=int,
    help="dimension of the latent space (default=8)",
    default=8,
)

parser.add_argument(
    "--n_instances", type=int, help="# of target samples (default=250)", default=250
)

parser.add_argument(
    "--n_ensambles",
    type=int,
    help="# of the initial configuration (default=1)",
    default=1,
)

parser.add_argument(
    "--logdiffsoglia",
    type=int,
    help="The logarithm of the threshold value for the early stopping (default=-4)",
    default=-4,
)


parser.add_argument(
    "--final_lr",
    type=float,
    help="learning rate at final epoch (dynamic) (default=10**-6)",
    default=10**-6,
)

parser.add_argument(
    "--early_stopping",
    type=str,
    help="if True set the early stopping option (default=False)",
    default="False",
)


parser.add_argument(
    "--variable_lr",
    type=str,
    help="True if the learning rate is dynamic (default=False)",
    default="False",
)

parser.add_argument(
    "--loglr",
    type=float,
    help="The logarithm of the learning rate (default=-1)",
    default=-1,
)

parser.add_argument(
    "--target_path",
    type=str,
    help="name of the target dataset (default='data/dataset_meyer/dataset_meyer_test_256_100.npz')",
    default="data/dataset_meyer/dataset_meyer_test_256_100.npz",
)

parser.add_argument(
    "--init_path",
    type=str,
    help="dataset path for the initial configurations (default='data/dataset_meyer/dataset_meyer_test_256_100.npz')",
    default="data/dataset_meyer/dataset_meyer_test_256_100.npz",
)

parser.add_argument(
    "--model_name",
    type=str,
    help="name of model (default='emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb')",
    default="meyer_case/DFTVAEnorm_hidden_channels_vae_[60, 60, 60, 60, 60]_hidden_channels_dft_[60, 60, 60]_kernel_size_13_pooling_size_2_latent_dimension_8_l1_0.0_l2_0.001",
)
parser.add_argument(
    "--epochs", type=int, help="# of epochs (default=15001)", default=15001
)

parser.add_argument("--L", type=int, help="size of the system (default=1)", default=1)
parser.add_argument(
    "--resolution", type=int, help="resolution of the system (default=256)", default=256
)

parser.add_argument(
    "--num_threads",
    type=int,
    help="number of threads for the torch process (default=10)",
    default=10,
)
parser.add_argument(
    "--device",
    type=str,
    help="the threshold difference for the early stopping (default=device available)",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
parser.add_argument(
    "--seed",
    type=int,
    help="seed for numpy and pytorch (default=42)",
    default=42,
)


args = parser.parse_args()

print(args)

torch.set_num_threads(args.num_threads)

gd = GradientDescent(
    n_instances=args.n_instances,
    loglr=args.loglr,
    logdiffsoglia=args.logdiffsoglia,
    n_ensambles=args.n_ensambles,
    target_path=args.target_path,
    model_name=args.model_name,
    epochs=args.epochs,
    variable_lr=from_txt_to_bool(args.variable_lr),
    final_lr=args.final_lr,
    early_stopping=from_txt_to_bool(args.early_stopping),
    L=args.L,
    resolution=args.resolution,
    latent_dimension=args.latent_dimension,
    seed=args.seed,
    num_threads=args.num_threads,
    device=args.device,
    init_path=args.init_path,
    Energy=Energy,
    dimension="1D",
)


gd.run()
