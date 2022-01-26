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
    "--loglr", type=int, help="value of the log(lr) (default=-3)", default=-3
)
parser.add_argument(
    "--cut", type=int, help="value of gradient cutoff (deprecated)", default=128
)
parser.add_argument(
    "--logdiffsoglia",
    type=int,
    help="value of the early stopping thrashold (default=-5)",
    default=-5,
)
parser.add_argument(
    "--n_ensambles",
    type=int,
    help="# of initial configuration (default=20)",
    default=20,
)
parser.add_argument(
    "--target_path",
    type=str,
    help="name of the target dataset (default='data/final_dataset/data_test.npz')",
    default="data/final_dataset/data_test.npz",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="name of model (default='emodel_2_layer_120_hc_13_ks_2_ps')",
    default="emodel_2_layer_120_hc_13_ks_2_ps",
)
parser.add_argument(
    "--epochs", type=int, help="# of epochs (default=9000)", default=9000
)
parser.add_argument(
    "--variable_lr",
    type=str,
    help="if it is true implement a dynamic learning rate (default=True)",
    default="True",
)
parser.add_argument(
    "--early_stopping",
    type=str,
    help="if it is true implement the early stopping (default=False)",
    default="False",
)
parser.add_argument("--L", type=int, help="size of the system (default=14)", default=14)
parser.add_argument(
    "--resolution", type=int, help="resolution of the system (default=256)", default=256
)
parser.add_argument(
    "--final_lr",
    type=float,
    help="resolution of the system (default=10**-6)",
    default=10 ** -6,
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


gd = GradientDescent(
    latent_dimension=args.latent_dimension,
    n_instances=args.n_instances,
    loglr=args.loglr,
    cut=args.cut,
    n_ensambles=args.n_ensambles,
    model_name=args.model_name,
    target_path=args.target_path,
    epochs=args.epochs,
    variable_lr=from_txt_to_bool(args.variable_lr),
    early_stopping=from_txt_to_bool(args.early_stopping),
    L=args.L,
    resolution=args.resolution,
    final_lr=args.final_lr,
    num_threads=args.num_threads,
    device=args.device,
    seed=args.seed,
    logdiffsoglia=args.logdiffsoglia,
)

gd.run()
