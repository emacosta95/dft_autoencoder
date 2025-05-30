from pyexpat import model
import random
from types import ModuleType
import torch as pt
import numpy as np
from torch.nn.modules import pooling
from tqdm import tqdm, trange
from src.training.utils import (
    make_data_loader,
    get_optimizer,
    count_parameters,
    from_txt_to_bool,
)
from src.losses import DFTVAELoss
from src.training.utils import VaeLossMSE
from src.training.train_module import fit, fit2ndGEN
import torch.nn as nn
import argparse
import os
import importlib


# parser arguments

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


# pooling_size_dft=args.pooling_size_dft,
# kernel_size_dft=args.kernel_size_dft,
# hidden_channels_dft=args.hidden_channels_dft,
# padding_dft=args.padding_dft,


def main(args):
    # hyperparameters

    device = pt.device(args.device)
    input_channel = args.input_channels
    input_size = args.input_size

    # 256 for model test, 30 for the others
    output_size = 1
    pooling_size = args.pooling_size

    padding = args.padding  # 6
    padding_mode = args.padding_mode

    kernel_size = args.kernel_size  # 13

    if args.ModelType == "DFTVAEnorm3D":
        kernel_size = [args.kernel_size for i in range(3)]
        padding = [(args.kernel_size - 1) // 2 for i in range(3)]
        input_size = [args.input_size for i in range(3)]
        pooling_size = [args.pooling_size for i in range(3)]

    # Select the number of threads
    pt.set_num_threads(args.num_threads)

    # Initialize the seed
    pt.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pt.cuda.manual_seed(args.seed)
    pt.backends.cudnn.deterministic = (
        True  # Note that this Deterministic mode can have a performance impact
    )
    pt.backends.cudnn.benchmark = False

    # Set hyperparameters
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    patiance = args.patiance
    early_stopping = args.early_stopping

    # Set the model name
    name_loss_parameter2 = f"l2_{args.loss_parameter2}"
    name_loss_parameter1 = f"l1_{args.loss_parameter1}_"
    training_description = (
        name_loss_parameter1 + name_loss_parameter2 + args.other_information
    )

    model_directory = args.model_directory

    # Set the dataset path
    file_name = args.data_path

    if args.ModelType == "DFTVAEnorm2ndGEN":
        loss = DFTVAELoss(
            loss_parameter=args.loss_parameter1,
            variational_beta=args.loss_parameter2,
        )

    else:
        loss = {}
        loss["generative"] = VaeLossMSE(args.loss_parameter2)
        loss["prediction"] = nn.MSELoss()

    if args.load:
        print(f"loading the model {args.name}")
        if args.generative:
            if os.path.isfile(
                f"losses_dft_pytorch/{args.name}" + "_loss_valid_generative"
            ):
                history_valid = pt.load(
                    f"losses_dft_pytorch/{args.name}" + "_loss_valid_generative"
                )
                history_train = pt.load(
                    f"losses_dft_pytorch/{args.name}" + "_loss_train_generative"
                )
            else:
                history_valid = []
                history_train = []
        else:
            if os.path.isfile(f"losses_dft_pytorch/{args.name}" + "_loss_valid_dft"):
                history_valid = pt.load(
                    f"losses_dft_pytorch/{args.name}" + "_loss_valid_dft"
                )
                history_train = pt.load(
                    f"losses_dft_pytorch/{args.name}" + "_loss_train_dft"
                )
            else:
                history_valid = []
                history_train = []

        print(len(history_train), len(history_valid))
        model = pt.load(f"model_dft_pytorch/{args.name}", map_location=device)
        model_name = args.name
        # redefine the loss
        model.training_restriction = args.training_restriction
        # we should implement a getter for this
        if args.ModelType == "DFTVAEnorm" or args.ModelType == "DFTVAEnorm3D":
            model.loss = loss[args.training_restriction]
    else:
        history_valid = []
        history_train = []

        module = importlib.import_module("src.model_dft_autoencoder")
        model_class = getattr(module, args.ModelType)
        model = model_class(
            input_size=input_size,
            latent_dimension=args.latent_dimension,
            loss=loss,
            input_channels=input_channel,
            hidden_channels_vae=args.hidden_channels_vae,
            hidden_channels_dft=args.hidden_channels_dft,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            pooling_size=pooling_size,
            output_size=input_size,
            activation=args.activation,
            # only provisional
            dx=args.l / args.input_size,
            training_restriction=args.training_restriction,
        )

        model.name_checkpoint(training_description, args.model_directory)

    model = model.to(pt.double)
    model = model.to(device=device)

    print(model.model_name)
    print(model)
    print(count_parameters(model))
    print(args)

    train_dl, valid_dl = make_data_loader(
        file_name=file_name,
        bs=bs,
        split=0.8,
        generative=from_txt_to_bool(args.generative),
    )

    opt = get_optimizer(lr=lr, model=model, weight_decay=args.regularization)

    fit2ndGEN(
        # supervised=not (from_txt_to_bool(args.generative)),
        model=model,
        train_dl=train_dl,
        opt=opt,
        epochs=epochs,
        valid_dl=valid_dl,
        checkpoint=True,
        history_train=history_train,
        history_valid=history_valid,
        loss_func=nn.MSELoss(),
        patiance=patiance,
        early_stopping=early_stopping,
        device=device,
    )

    print(model)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
