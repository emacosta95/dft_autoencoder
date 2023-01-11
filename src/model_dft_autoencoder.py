from re import X
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchmetrics import R2Score
from zmq import device
from src.model_dft import Pilati_model_3_layer, Pilati_model_3d_3_layer
from src.model_vae import Encode, DecodeNorm, Encode3d, DecodeNorm3d


class DFTVAEnorm(nn.Module):
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
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
        activation: nn.Module,
        dx: float,
    ):

        super().__init__()

        self.loss_generative = loss_generative
        self.loss_dft = loss_dft

        self.Encoder = Encode(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels,
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
        self.DFTModel = Pilati_model_3_layer(
            input_size=input_size,
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
        loss, kldiv = self.loss_generative(x_recon, x, latent_mu, latent_logvar)
        return loss, kldiv

    def fit_dft_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.unsqueeze(1).to(device=device)
        y = y.to(device=device)
        x = self.DFTModel(x).squeeze()
        loss = self.loss_dft(x, y)
        return loss

    def r2_computation(self, batch: Tuple, device: str, r2):
        x, y = batch
        x = x.unsqueeze(1).to(device=device)
        x = self.DFTModel(x).to(device=device).squeeze()
        r2.update(x.cpu().detach(), y.cpu().detach())
        return r2


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
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
        activation: str,
        dx: float,
    ):

        super().__init__()

        self.loss_generative = loss_generative
        self.loss_dft = loss_dft

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
        x = x.view(x.shape[0], -1)
        return x, f

    def proposal(self, z: torch.Tensor):
        x = self.Decoder(z)
        x = x.view(x.shape[0], -1)
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

    def fit_dft_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.unsqueeze(1).to(device=device)
        y = y.to(device=device)
        x = self.DFTModel(x).squeeze()
        loss = self.loss_dft(x, y)
        return loss

    def r2_computation(self, batch: Tuple, device: str, r2):
        x, y = batch
        x = x.unsqueeze(1).to(device=device)
        x = self.DFTModel(x).to(device=device).squeeze()
        r2.update(x.cpu().detach(), y.cpu().detach())
        return r2
