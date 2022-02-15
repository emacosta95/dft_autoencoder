from src.gradient_descent import GradientDescent
import argparse
import torch
import numpy as np
from src.training.utils import from_txt_to_bool


parser = argparse.ArgumentParser()

parser.add_argument(
    "--latent_dimension",
    type=int,
    help="dimension of the latent space (default=16)",
    default=16,
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
    default=10 ** -6,
)

parser.add_argument(
    "--early_stopping",
    type=str,
    help="if True set the early stopping option (default=False)",
    default="False",
)

parser.add_argument(
    "--mu",
    type=float,
    help="Chemical potential for the normalization, softconstrain (default=None)",
    default=None,
)


parser.add_argument(
    "--variable_lr",
    type=str,
    help="True if the learning rate is dynamic (default=False)",
    default="False",
)

parser.add_argument(
    "--loglr",
    type=int,
    help="The logarithm of the learning rate (default=-1)",
    default=-1,
)

parser.add_argument(
    "--target_path",
    type=str,
    help="name of the target dataset (default='data/final_dataset/data_test.npz')",
    default="data/final_dataset/data_test.npz",
)

parser.add_argument(
    "--init_path",
    type=str,
    help="dataset path for the initial configurations (default='data/final_dataset/data_train.npz')",
    default="data/final_dataset/data_train.npz",
)

parser.add_argument(
    "--model_name",
    type=str,
    help="name of model (default='emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb')",
    default="emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb",
)
parser.add_argument(
    "--epochs", type=int, help="# of epochs (default=15001)", default=15001
)

parser.add_argument("--L", type=int, help="size of the system (default=14)", default=14)
parser.add_argument(
    "--resolution", type=int, help="resolution of the system (default=256)", default=256
)

parser.add_argument(
    "--num_threads",
    type=int,
    help="number of threads for the torch process (default=1)",
    default=1,
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

gd = GradientDescent(
    n_instances=args.n_instances,
    loglr=args.loglr,
    cut=128,
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
    mu=args.mu,
    init_path=args.init_path,
)


gd.run()
