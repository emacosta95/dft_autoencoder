from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser.add_argument(
    "--load",
    type=bool,
    help="Loading or not the model",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--name", type=str, help="name of the model", default=None)

parser.add_argument(
    "--data_path",
    type=str,
    help="data path (default=data/dataset_meyer/dataset_meyer_test_256_15k_a_1-10_b_04-06_c_003-01.npz)",
    default="data/dataset_meyer/dataset_meyer_test_256_15k_a_1-10_b_04-06_c_003-01.npz",
)

parser.add_argument(
    "--num_threads",
    type=int,
    help="the number of threads for pytorch (default=1)",
    default=1,
)

parser.add_argument(
    "--seed",
    type=int,
    help="seed for pytorch and numpy (default=42)",
    default=42,
)


parser.add_argument(
    "--device",
    type=str,
    help="the threshold difference for the early stopping (default=device available)",
    default=("cuda" if pt.cuda.is_available() else "cpu"),
)

parser.add_argument(
    "--patiance",
    type=int,
    help="num of epochs tollerance for the early stopping (default=5)",
    default=5,
)

parser.add_argument(
    "--early_stopping",
    type=float,
    help="the threshold difference for the early stopping (default=10**-4)",
    default=10**-4,
)

parser.add_argument(
    "--lr",
    type=float,
    help="learning rate (default=0.001)",
    default=0.001,
)

parser.add_argument(
    "--bs",
    type=int,
    help="batch size (default=100)",
    default=100,
)


parser.add_argument(
    "--epochs",
    type=int,
    help="training epochs (default=1200)",
    default=1200,
)

parser.add_argument(
    "--regularization",
    type=float,
    help="the weight decay, useful to avoid overfitting",
    default=0.0,
)

parser.add_argument(
    "--l",
    type=float,
    help="size of the box (default=1)",
    default=1,
)

parser.add_argument(
    "--loss_parameter1",
    type=float,
    help="the amplitude of the kldiv (default=1e-06)",
    default=10**-6,
)

parser.add_argument(
    "--loss_parameter2",
    type=float,
    help="the convex parameter of the full loss (default=0.5)",
    default=0.5,
)

parser.add_argument(
    "--training_restriction",
    type=str,
    help="trains just a set of parameters, it could be either 'generative' or 'prediction'",
    default="generative",
)


model_parser = subparsers.add_parser("model", help="model hparameters")

model_parser.add_argument(
    "--generative",
    type=str,
    help="if the model is generative or not (default=True)",
    default="False",
)

model_parser.add_argument(
    "--ModelType",
    type=str,
    help="if the model is generative or not (default=True)",
    default="DFTVAEnorm2ndGEN",
)

model_parser.add_argument(
    "--input_channels", type=int, help="# input channels (default=1)", default=1
)
model_parser.add_argument(
    "--input_size",
    type=int,
    help="number of features of the input (default=256)",
    default=256,
)

model_parser.add_argument(
    "--latent_dimension",
    type=int,
    help="number of features of the input (default=2)",
    default=2,
)

model_parser.add_argument(
    "--hidden_channels_vae",
    type=int,
    nargs="+",
    help="list of hidden channels in the VAE Model (default=20)",
    default=[60, 60, 60, 60, 60],
)

model_parser.add_argument(
    "--hidden_channels_dft",
    type=int,
    nargs="+",
    help="list of hidden channels in the DFTModel (default=[40,40,40,40,40])",
    default=[40, 40, 40, 40, 40],
)


model_parser.add_argument(
    "--pooling_size",
    type=int,
    help="pooling size in the Avg Pooling (default=2)",
    default=2,
)

model_parser.add_argument(
    "--padding",
    type=int,
    help="padding dimension (default=2)",
    default=6,
)


model_parser.add_argument(
    "--kernel_size",
    type=int,
    help="kernel size (default=13)",
    default=13,
)

model_parser.add_argument(
    "--padding_mode",
    type=str,
    help="the padding mode of the model (default='zeros')",
    default="zeros",
)


model_parser.add_argument(
    "--activation",
    type=str,
    help="activation function (default='Softplus')",
    default="Softplus",
)

model_parser.add_argument(
    "--model_directory",
    type=str,
    help="name of the directory where the models are saved (default='cnn_for_gaussian')",
    default="meyer_case/",
)

model_parser.add_argument(
    "--other_information",
    type=str,
    help="other comments about either the training or the neural network ('')",
    default="",
)

parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

