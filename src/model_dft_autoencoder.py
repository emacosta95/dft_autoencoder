from re import X
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchmetrics import R2Score
from zmq import device
from src.model_dft import PredictionHead, DFTModel
from src.model_vae import Encode, DecodeNorm, Encode3d, DecodeNorm3d, Encode3db


class DFTVAEnorm(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        loss: nn.Module,
        output_size: int,
        activation: nn.Module,
        hidden_channels_vae: List,
        hidden_channels_dft: List,
        dx: float,
        training_restriction: str,
    ):
        super().__init__()

        self.training_restriction = training_restriction
        self.loss = loss[self.training_restriction]

        self.Encoder = Encode(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels_vae,
            input_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            input_size=input_size,
            pooling_size=pooling_size,
            activation=activation,
        )
        self.Decoder = DecodeNorm(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels_vae,
            output_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            output_size=input_size,
            pooling_size=pooling_size,
            activation=activation,
            dx=dx,
        )
        self.DFTModel = DFTModel(
            input_size=input_size,
            input_channel=input_channels,
            hidden_channel=hidden_channels_dft,
            padding=int((kernel_size - 1) / 2),
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            output_size=1,
            activation=activation,
        )

        # data of the model
        self.hidden_channels_vae = hidden_channels_vae
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.hidden_channels_dft = hidden_channels_dft
        self.ModelType = "DFTVAEnorm"
        self.latent_dimension = latent_dimension

    def forward(self, z: torch.Tensor):
        x = self.Decoder(z)
        f = self.DFTModel(x)
        x = x.view(x.shape[0], -1)
        return x, f

    def proposal(self, z: torch.Tensor):
        x = self.Decoder(z)
        x = x.view(x.shape[0], -1)
        return x

    def functional(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        return self.DFTModel(x)

    def _latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu

    def train_generative_step(self, batch: Tuple, device: str):
        x = batch[0]
        x = x.unsqueeze(1).to(device=device)
        latent_mu, latent_logvar = self.Encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.Decoder(latent)

        loss, kldiv = self.loss(x_recon, x, latent_mu, latent_logvar)
        return loss, kldiv

    def train_dft_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.unsqueeze(1).to(device=device)
        y = y.to(device=device)
        x = self.DFTModel(x).squeeze()
        loss = self.loss(x, y)
        return loss

    def train_step(self, batch: Tuple, device: str):
        if self.training_restriction == "prediction":
            loss = self.train_dft_step(batch, device=device)
        elif self.training_restriction == "generative":
            loss, _ = self.train_generative_step(batch, device=device)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        if self.training_restriction == "prediction":
            loss = self.train_dft_step(batch, device=device)
        elif self.training_restriction == "generative":
            loss, _ = self.train_generative_step(batch, device=device)
        return loss, loss, torch.tensor(0.0)

    def freezing_parameters(self):
        if self.training_restriction == "prediction":
            for param in self.Decoder.parameters():
                param.requires_grad = False
            for param in self.Encoder.parameters():
                param.requires_grad = False
            for param in self.DFTModel.parameters():
                param.requires_grad = True

        elif self.training_restriction == "generative":
            for param in self.DFTModel.parameters():
                param.requires_grad = False
            for param in self.Decoder.parameters():
                param.requires_grad = True
            for param in self.Encoder.parameters():
                param.requires_grad = True

    def name_checkpoint(self, training_description: str, model_directory: str) -> None:
        name_hc = f"_hidden_channels_vae_{self.hidden_channels_vae}_"
        name_hidden_neurons = f"hidden_channels_dft_{self.hidden_channels_dft}_"
        name_ks = f"kernel_size_{self.kernel_size}_"
        name_pooling_size = f"pooling_size_{self.pooling_size}_"
        name_latent_dimension = f"latent_dimension_{self.latent_dimension}_"
        model_name = (
            self.ModelType
            + name_hc
            + name_hidden_neurons
            + name_ks
            + name_pooling_size
            + name_latent_dimension
            + training_description
        )
        self.model_name = model_directory + model_name


class DFTVAEnorm3D(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        kernel_size_dft: int,
        pooling_size: int,
        loss: nn.Module,
        output_size: int,
        activation: str,
        dx: float,
        training_restriction: str,
    ):
        super().__init__()

        self.training_restriction = training_restriction
        self.loss = loss[self.training_restriction]

        if training_restriction == "prediction":
            self.loss_dft = loss
        elif training_restriction == "generative":
            self.loss_generative = loss

        self.Encoder = Encode3d(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels,
            input_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            linear_input_size=input_size,
            pooling_size=pooling_size,
            activation=activation,
        )
        self.Decoder = DecodeNorm3d(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels,
            output_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            output_size=input_size,
            pooling_size=pooling_size,
            activation=activation,
            dx=dx,
        )
        self.DFTModel = Pilati_model_3d_3_layer(
            linear_input_size=input_size,
            input_channel=input_channels,
            hidden_channel=hidden_channels,
            padding=int((kernel_size_dft - 1) / 2),
            padding_mode=padding_mode,
            kernel_size=kernel_size_dft,
            pooling_size=pooling_size,
            output_size=output_size,
            activation=activation,
        )

    def forward(self, z: torch.Tensor):
        x = self.Decoder(z)
        f = self.DFTModel(x)
        x = x.squeeze(1)
        return x, f

    def proposal(self, z: torch.Tensor):
        x = self.Decoder(z)
        x = x.squeeze(1)
        return x

    def functional(self, x: torch.Tensor):
        # x = x.unsqueeze(1)
        return self.DFTModel(x)

    def _latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu

    def train_generative_step(self, batch: Tuple, device: str):
        x = batch[0]
        x = x.unsqueeze(1).to(device=device)
        latent_mu, latent_logvar = self.Encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.Decoder(latent)
        loss, kldiv = self.loss_generative(x_recon, x, latent_mu, latent_logvar)
        return loss, kldiv

    def train_dft_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.unsqueeze(1).to(device=device)
        y = y.to(device=device)
        x = self.DFTModel(x).squeeze()
        loss = self.loss_dft(x, y)
        return loss

    def train_step(self, batch: Tuple, device: str):
        if self.training_restriction == "prediction":
            loss = self.train_dft_step(batch, device=device)
        elif self.training_restriction == "generative":
            loss = self.train_generative_step(batch, device=device)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        if self.training_restriction == "prediction":
            loss = self.train_dft_step(batch, device=device)
        elif self.training_restriction == "generative":
            loss = self.train_generative_step(batch, device=device)
        return loss

    def freezing_parameters(self):
        if self.training_restriction == "prediction":
            for param in self.Decoder.parameters():
                param.requires_grad = False
            for param in self.Encoder.parameters():
                param.requires_grad = False
            for param in self.DFTModel.parameters():
                param.requires_grad = True

        elif self.training_restriction == "generative":
            for param in self.DFTModel.parameters():
                param.requires_grad = False
            for param in self.Decoder.parameters():
                param.requires_grad = True
            for param in self.Encoder.parameters():
                param.requires_grad = True


class DFTVAEnorm2ndGEN(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels_vae: List,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        loss: nn.Module,
        output_size: int,
        activation: nn.Module,
        hidden_channels_dft: List,
        dx: float,
        training_restriction: str,
    ):
        super().__init__()

        self.model_name = None

        self.training_restriction = training_restriction
        self.loss = loss

        self.Encoder = Encode(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels_vae,
            input_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            input_size=input_size,
            pooling_size=pooling_size,
            activation=activation,
        )
        self.Decoder = DecodeNorm(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels_vae,
            output_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            output_size=output_size,
            pooling_size=pooling_size,
            activation=activation,
            dx=dx,
        )
        self.DFTModel = PredictionHead(
            hidden_neurons=hidden_channels_dft,
            latent_space=latent_dimension,
            activation=activation,
        )

        # data of the model
        self.hidden_channels = hidden_channels_vae
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.hidden_neurons = hidden_channels_dft
        self.ModelType = "DFTVAEnorm2ndGEN"
        self.latent_dimension = latent_dimension

    def forward(self, z: torch.Tensor):
        x = self.Decoder(z)
        f = self.DFTModel(z)
        x = x.view(x.shape[0], -1)
        return x, f

    def proposal(self, z: torch.Tensor):
        x = self.Decoder(z)
        x = x.view(x.shape[0], -1)
        return x

    def functional(self, z: torch.Tensor):
        return self.DFTModel(z)

    def _latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        # generative step
        y = y.to(device=device).view(-1)
        x = x.unsqueeze(1).to(device=device)
        latent_mu, latent_logvar = self.Encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.Decoder(latent)
        # prediction
        y_pred = self.DFTModel(latent).view(-1)
        loss, _, _ = self.loss(x, y, x_recon, y_pred, latent_mu, latent_logvar)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        # generative step
        y = y.to(device=device).view(-1)
        x = x.unsqueeze(1).to(device=device)
        latent_mu, latent_logvar = self.Encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.Decoder(latent)
        # prediction
        y_pred = self.DFTModel(latent).view(-1)
        loss, l1, l2 = self.loss(x, y, x_recon, y_pred, latent_mu, latent_logvar)
        return (loss, l1, l2)

    def freezing_parameters(self):
        if self.training_restriction == "prediction":
            for param in self.Decoder.parameters():
                param.requires_grad = False
            for param in self.Encoder.parameters():
                param.requires_grad = False
            for param in self.DFTModel.parameters():
                param.requires_grad = True

        elif self.training_restriction == "generative":
            for param in self.DFTModel.parameters():
                param.requires_grad = False
            for param in self.Decoder.parameters():
                param.requires_grad = True
            for param in self.Encoder.parameters():
                param.requires_grad = True

    def name_checkpoint(self, training_description: str, model_directory: str) -> None:
        name_hc = f"_hidden_channels_vae_{self.hidden_channels}_"
        name_hidden_neurons = f"hidden_neurons_dft_{self.hidden_neurons}_"
        name_ks = f"kernel_size_{self.kernel_size}_"
        name_pooling_size = f"pooling_size_{self.pooling_size}_"
        name_latent_dimension = f"latent_dimension_{self.latent_dimension}_"
        model_name = (
            self.ModelType
            + name_hc
            + name_hidden_neurons
            + name_ks
            + name_pooling_size
            + name_latent_dimension
            + training_description
        )

        self.model_name = model_directory + model_name
