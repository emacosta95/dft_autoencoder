from typing import Dict, Tuple
import torch
import torch.nn as nn


class Encode(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        hidden_channel: int,
        latent_dimension: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
    ):

        super().__init__()

        self.block_1 = nn.Sequential(
            # nn.BatchNorm1d(input_channels),
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(hidden_channel),
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(hidden_channel),
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(hidden_channel),
        )
        self.final_mu = nn.Sequential(
            nn.Linear(
                hidden_channel * int(input_size / (pooling_size ** 3)), latent_dimension
            ),
        )
        self.final_logsigma = nn.Sequential(
            nn.Linear(
                hidden_channel * int(input_size / (pooling_size ** 3)), latent_dimension
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.shape[0], -1)

        x_mu = self.final_mu(x)
        x_logstd = self.final_logsigma(x)
        return x_mu, x_logstd


class Decode(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channel: int,
        output_channels: int,
        output_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
    ):
        super().__init__()
        self.recon_block = nn.Sequential(
            nn.Linear(
                latent_dimension,
                int(output_size / (pooling_size) ** 3) * hidden_channel,
            ),
        )
        self.block_conv1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channel),
        )
        self.block_conv2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channel),
        )
        self.block_conv3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channel,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
        )
        self.hidden_channel = hidden_channel

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.recon_block(z)
        z = z.view(z.shape[0], self.hidden_channel, -1)
        z = self.block_conv1(z)
        z = self.block_conv3(z)
        z = torch.sigmoid(z)
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

        self.decoder = Decode(
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
            return pt.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu


class DFTVAE(nn.Module, Encode, Decode):
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
        self.Encode = Encode()
        self.Decode = Decode()
