from typing import List
from pyexpat import model
import random
import torch
import numpy as np
from torch.nn.modules import pooling
from src.model_vae import VarAE2D
from src.training.utils import (
    make_data_loader,
    get_optimizer,
    count_parameters,
    VaeLoss,
    from_txt_to_bool,
)
from src.training.train_module import fit
import torch.nn as nn
import argparse
import os


# parser arguments

parser = argparse.ArgumentParser()


parser.add_argument("--load", type=bool, help="Loading or not the model", default=False)
parser.add_argument("--name", type=str, help="name of the model", default=None)

parser.add_argument(
    "--data_path",
    type=str,
    help="",
    default="data/",
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
    help="device in which the script would run, either 'cuda' (gpu) or 'cpu'  (default=device available)",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

parser.add_argument(
    "--patiance",
    type=int,
    help="# of epochs tollerance for the early stopping (default=5)",
    default=5,
)

parser.add_argument(
    "--early_stopping",
    type=float,
    help="the threshold for the early stopping (default=10**-4)",
    default=10 ** -4,
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
    help="batch size (default=2000)",
    default=2000,
)

parser.add_argument(
    "--epochs",
    type=int,
    help="training epochs (default=2000)",
    default=2000,
)



parser.add_argument(
    "--input_channels",
    type=int,
    help="# channels of the input data (default=1)",
    default=1,
)
parser.add_argument(
    "--input_size",
    nargs='+',
    type=int,
    help="number of features of the input data (default=(16,16))",
    default=(16,16),
)

parser.add_argument(
    "--latent_dimension",
    type=int,
    help="dimension of the latent manifold (default=6)",
    default=6,
)

parser.add_argument(
    "--hidden_channels",
    type=int,
    help="channels (or filters) of the hidden layers (default=20)",
    default=20,
)

parser.add_argument(
    "--pooling_size",
    type=int,
    help="pooling size in the Avg Pooling (default=1)",
    default=1,
)

parser.add_argument(
    "--padding",
    nargs='+',
    type=int,
    help="padding dimension (default=[1,0])",
    default=[1,0],
)

parser.add_argument(
    "--hidden_neurons",
    nargs='+',
    type=int,
    help="number of hidden neurons per layer in the dense nn (default=[30,20,10])",
    default=[30,20,10],
)

parser.add_argument(
    "--dense",
    type=str,
    help="if True select the dense nn (default=False)",
    default=False,
)


parser.add_argument(
    "--kernel_size",
    nargs='+',
    type=int,
    help="kernel size (default=[3,1])",
    default=[3,1],
)

parser.add_argument(
    "--padding_mode",
    type=str,
    help="the padding mode of the model (default='circular')",
    default="circular",
)

parser.add_argument(
    "--loss_parameter",
    type=float,
    help="the amplitude of the kldiv (default=0.001)",
    default=0.001,
)

parser.add_argument(
    "--model_name",
    type=str,
    help="name of the model (default='vae_model')",
    default="vae_model",
)

parser.add_argument(
    "--ModelType",
    type=str,
    help="Type of the Model (default='VAETomography')",
    default="VAETomography",
)

parser.add_argument(
    "--nlayers",
    type=int,
    help="number of conv block (default=3)",
    default=3,
)


parser.add_argument(
    "--nconv",
    type=int,
    help="number of conv layers per block (default=3)",
    default=3,
)


def main(args):

    # Select the number of threads
    torch.set_num_threads(args.num_threads)

    # Initialize the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = (
        True  # Note that this Deterministic mode can have a performance impact
    )
    torch.backends.cudnn.benchmark = False

    hidden_neurons=args.hidden_neurons

    # Set the model name
    if not(from_txt_to_bool(args.dense)):
        model_name = args.model_name
        name_hc = f"_{args.hidden_channels}_hc"
        name_ks = f"_{args.kernel_size}_ks"
        name_pooling_size = f"_{args.pooling_size}_ps"
        name_latent_dimension = f"_{args.latent_dimension}_ls"
        name_loss_parameter = f"_{args.loss_parameter}_vb"
        model_name = (
            model_name
            + name_hc
            + name_ks
            + name_pooling_size
            + name_latent_dimension
            + name_loss_parameter
        )

    else:
        model_name=args.model_name
        name_nlayers = f"_{len(hidden_neurons)}_nlayers"
        name_hn = f"_{hidden_neurons}_hidden_neurons"
        name_latent_dimension = f"_{args.latent_dimension}_ls"
        name_loss_parameter = f"_{args.loss_parameter}_vb"
        model_name = (
            model_name
            + name_nlayers
            + name_hn
            + name_latent_dimension
            + name_loss_parameter
        )
    # init the loss function
    loss_func = VaeLoss(variational_beta=args.loss_parameter)

    # loading the state dict
    if args.load:
        print(f"loading the model {args.name}")
        model=torch.load('saved_models/'+args.name)
        model=model.to(torch.double)
        if os.path.isfile(f"loss/{args.name}" + "_loss_train"):
            print('history loading confirmed!')
            history_train = torch.load(f"loss/{args.name}" + "_loss_train")
            history_valid=[]
    else:

        history_valid = []
        history_train = []

        if args.ModelType=='VAE':
            model = VarAE2D(
                input_size=args.input_size,
                latent_dimension=args.latent_dimension,
                input_channels=args.input_channels,
                hidden_channels=args.hidden_channels,
                kernel_size=args.kernel_size,
                padding=args.padding,
                padding_mode=args.padding_mode,
                pooling_size=args.pooling_size,
            )
        


    model = model.to(torch.double)
    model = model.to(device=args.device)

    print(model)
    print(count_parameters(model))

    train_dl = make_data_loader(
        file_name=args.data_path,
        bs=args.bs,
        split=0.8,
        generative=True,
        l=args.input_size[0]
    )

    opt = get_optimizer(lr=args.lr, model=model)

    fit(
        model=model,
        train_dl=train_dl,
        opt=opt,
        epochs=args.epochs,
        checkpoint=True,
        name_checkpoint=model_name,
        history_train=history_train,
        history_valid=history_valid,
        loss_func=loss_func,
        patiance=args.patiance,
        early_stopping=args.early_stopping,
        device=args.device,
    )
    # print the summary of the model
    # at the end
    print(model)


if __name__ == "__main__":

    args = parser.parse_args()

    main(args)
