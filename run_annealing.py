from src.gradient_descent import SimulatedAnnealing
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
    "--beta",
    type=float,
    help="beta value (default=10)",
    default=10,
)


parser.add_argument(
    "--final_beta",
    type=float,
    help="final beta value (default=100)",
    default=100,
)

parser.add_argument(
    "--ann_step",
    type=int,
    help="step of the simulated annealing for each fixed beta (default=100)",
    default=100,
)

parser.add_argument(
    "--local",
    type=str,
    help="True if the proposal is sampled by a local algorithm (default=False)",
    default="False",
)

parser.add_argument(
    "--delta",
    type=float,
    help="The average distance of the new sample (default=0.1)",
    default=0.1,
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
    help="name of model (default='emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb')",
    default="emodel_20_hc_13_ks_2_ps_16_ls_0.001_vb",
)
parser.add_argument(
    "--epochs", type=int, help="# of epochs (default=9000)", default=9000
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


ann = SimulatedAnnealing(
    n_instances=args.n_instances,
    beta=args.beta,
    delta=args.delta,
    final_beta=args.final_beta,
    ann_step=args.ann_step,
    local=from_txt_to_bool(args.local),
    target_path=args.target_path,
    model_name=args.model_name,
    epochs=args.epochs,
    L=args.L,
    resolution=args.resolution,
    latent_dimension=args.latent_dimension,
    seed=args.seed,
    num_threads=args.num_threads,
    device=args.device,
)

ann.run()
