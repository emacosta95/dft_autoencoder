from re import X
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchmetrics import R2Score
from zmq import device


class Encode(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        hidden_channels: int,
        latent_dimension: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
    ):

        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.block_1 = nn.Sequential(
            # nn.BatchNorm1d(input_channels),
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(hidden_channels),
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=2 * hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(2 * hidden_channels),
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * hidden_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(4 * hidden_channels),
        )
        self.final_mu = nn.Sequential(
            nn.Linear(
                4 * hidden_channels * int(input_size / (pooling_size ** 3)),
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )
        self.final_logsigma = nn.Sequential(
            nn.Linear(
                4 * hidden_channels * int(input_size / (pooling_size ** 3)),
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.shape[0], -1)

        x_mu = self.final_mu(x)
        x_logstd = self.final_logsigma(x)
        return x_mu, x_logstd


class Encode3d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        linear_input_size: int,
        hidden_channels: int,
        latent_dimension: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
    ):

        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.block_1 = nn.Sequential(
            # nn.BatchNorm1d(input_channels),
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
            nn.BatchNorm3d(hidden_channels),
        )
        self.block_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=2 * hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
            nn.BatchNorm3d(2 * hidden_channels),
        )
        self.block_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=2 * hidden_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
            nn.BatchNorm3d(4 * hidden_channels),
        )
        self.final_mu = nn.Sequential(
            nn.Linear(
                4 * hidden_channels * int(linear_input_size / (pooling_size ** 3)) ** 3,
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )
        self.final_logsigma = nn.Sequential(
            nn.Linear(
                4 * hidden_channels * int(linear_input_size / (pooling_size ** 3)) ** 3,
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.shape[0], -1)

        x_mu = self.final_mu(x)
        x_logstd = self.final_logsigma(x)
        return x_mu, x_logstd


class DecodeNorm3d(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        output_channels: int,
        output_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
        dx: float,
    ):
        super().__init__()

        self.output_size = output_size
        self.pooling_size = pooling_size
        self.dx = dx

        activation = getattr(torch.nn, activation)()

        self.recon_block = nn.Sequential(
            nn.Linear(
                latent_dimension,
                hidden_channels * int(output_size / (pooling_size) ** 3) ** 3,
            ),
        )
        self.block_conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            activation,
            nn.BatchNorm3d(hidden_channels),
        )
        self.block_conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            activation,
            nn.BatchNorm3d(hidden_channels),
        )
        self.block_conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=kernel_size + 3,
                stride=2,
                padding=padding,
            ),
        )
        self.hidden_channel = hidden_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.recon_block(z)
        z = z.view(
            -1,
            self.hidden_channel,
            int(self.output_size / (self.pooling_size ** 3)),
            int(self.output_size / (self.pooling_size ** 3)),
            int(self.output_size / (self.pooling_size ** 3)),
        )
        z = self.block_conv1(z)
        z = self.block_conv2(z)
        z = self.block_conv3(z)
        z = torch.sigmoid(z)
        # normalization
        # condition
        norm = torch.sum(z, dim=2) * self.dx
        z = z / norm[:, :, None]
        return z


class DecodeNorm(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        output_channels: int,
        output_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
        dx: float,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.output_size = output_size
        self.pooling_size = pooling_size
        self.dx = dx

        self.recon_block = nn.Sequential(
            nn.Linear(
                latent_dimension,
                int(output_size / (pooling_size) ** 3) * hidden_channels,
            ),
        )
        self.block_conv1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            activation,
            nn.BatchNorm1d(hidden_channels),
        )
        self.block_conv2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            activation,
            nn.BatchNorm1d(hidden_channels),
        )
        self.block_conv3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
        )
        self.hidden_channel = hidden_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.recon_block(z)
        z = z.view(
            -1, self.hidden_channel, int(self.output_size / (self.pooling_size ** 3))
        )
        z = self.block_conv1(z)
        z = self.block_conv2(z)
        z = self.block_conv3(z)
        z = torch.sigmoid(z)
        # normalization
        # condition
        norm = torch.sum(z, dim=2) * self.dx
        z = z / norm[:, :, None]

        return z


class VarAE(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channel: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
    ):

        super().__init__()

        self.encoder = Encode(
            input_channels=input_channels,
            input_size=input_size,
            hidden_channel=hidden_channel,
            latent_dimension=latent_dimension,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
        )

        self.decoder = DecodeNorm(
            output_channels=input_channels,
            output_size=input_size,
            hidden_channel=hidden_channel,
            latent_dimension=latent_dimension,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
        )

    def forward(self, x):

        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)

        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):

        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu
