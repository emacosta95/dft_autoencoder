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
        activation:nn.Module
    ):

        super().__init__()

        self.block_1 = nn.Sequential(
            #nn.BatchNorm1d(input_channels),
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
                out_channels=2*hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(2*hidden_channels),
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=2*hidden_channels,
                out_channels=4*hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
            nn.BatchNorm1d(4*hidden_channels),
        )
        self.final_mu = nn.Sequential(
            nn.Linear(
                4*hidden_channels * int(input_size / (pooling_size ** 3)),
                100,
            ),
            activation,
            nn.Linear(100,50),
            activation,
            nn.Linear(50,latent_dimension)
         )
        self.final_logsigma = nn.Sequential(
            nn.Linear(
                4*hidden_channels * int(input_size / (pooling_size ** 3)),
                100,
            ),
            activation,
            nn.Linear(100,50),
            activation,
            nn.Linear(50,latent_dimension)
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
        hidden_channels: int,
        output_channels: int,
        output_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
    ):
        super().__init__()

        self.output_size = output_size
        self.pooling_size = pooling_size

        self.recon_block = nn.Sequential(
            nn.Linear(latent_dimension,int(output_size / (pooling_size) ** 3) * hidden_channels*4)
        )
        self.block_conv1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=4*hidden_channels,
                out_channels=2*hidden_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(2*hidden_channels),
        )
        self.block_conv2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=2*hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
            ),
            nn.ReLU(),
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
            -1, 4*self.hidden_channel, int(self.output_size / (self.pooling_size ** 3))
        )
        z = self.block_conv1(z)
        z = self.block_conv2(z)
        z = self.block_conv3(z)
        z=z.cos()
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
        activation:nn.Module,
        dx: float,
    ):
        super().__init__()

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
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu

class DFT_model_5_layer(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channel: int,
        hidden_channel: int,
        output_size: int,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        pooling_size: int,
    ):
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.AvgPool1d(kernel_size=pooling_size),
        )
        self.model_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.AvgPool1d(kernel_size=pooling_size),
        )
        self.model_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.AvgPool1d(kernel_size=pooling_size),
        )

        self.model_4 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.AvgPool1d(kernel_size=pooling_size),
        )

        self.model_5 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.Softplus(),
            nn.AvgPool1d(kernel_size=pooling_size),
        )

        self.flat = nn.Flatten()

        self.final_dense = nn.Sequential(
            nn.Linear(hidden_channel * int(256 / pooling_size ** 5), 20),
            nn.Softplus(),
            nn.Linear(20,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_1(x)
        x = self.model_2(x)
        x = self.model_3(x)
        x = self.model_4(x)
        x = self.model_5(x)
        x = self.flat(x)
        x = self.final_dense(x)

        return x



class Pilati_model_3_layer(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channel: int,
        hidden_channel: int,
        output_size: int,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        pooling_size: int,
        activation:nn.Module
    ):
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
        )
        self.model_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
        )
        self.model_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation,
            nn.AvgPool1d(kernel_size=pooling_size),
        )

        self.flat = nn.Flatten()

        self.final_dense = nn.Sequential(
            nn.Linear(hidden_channel * int(256 / pooling_size ** 3), 20),
            activation,
            nn.Linear(20,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_1(x)
        x = self.model_2(x)
        x = self.model_3(x)
        x = self.flat(x)
        x = self.final_dense(x)

        return x


class DFTVAE(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
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
        )
        self.Decoder = Decode(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels,
            output_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            output_size=input_size,
            pooling_size=pooling_size,
        )
        self.DFTModel = Pilati_model_3_layer(
            input_size=input_size,
            input_channel=input_channels,
            hidden_channel=hidden_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            output_size=output_size,
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
        return loss, kldiv.item()

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


class DFTVAEIsing(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
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
        )
        self.Decoder = Decode(
            latent_dimension=latent_dimension,
            hidden_channels=hidden_channels,
            output_channels=input_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            output_size=input_size,
            pooling_size=pooling_size,
        )
        self.DFTModel = Pilati_model_3_layer(
            input_size=input_size,
            input_channel=input_channels,
            hidden_channel=hidden_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            output_size=output_size,
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
        return loss, kldiv.item()

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
        pooling_size: int,
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
        activation:nn.Module,
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
            activation=activation
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
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            output_size=output_size,
            activation=activation
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
        return loss, kldiv.item()

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


class DFTVAEnormHeavy(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
        pooling_size_dft:int,
        kernel_size_dft:int,
        hidden_channels_dft:int,
        padding_dft:int,
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
            activation=nn.Softplus(),
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
            activation=nn.Softplus(),
            dx=dx,
        )
        self.DFTModel = DFT_model_5_layer(
            input_size=input_size,
            input_channel=input_channels,
            hidden_channel=hidden_channels_dft,
            padding=padding_dft,
            padding_mode=padding_mode,
            kernel_size=kernel_size_dft,
            pooling_size=pooling_size_dft,
            output_size=output_size,
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
        return loss, kldiv.item()

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



class Energy(nn.Module):
    def __init__(self, F_universal: nn.Module, v: torch.Tensor, dx: float, mu: float):
        super().__init__()
        self.model = F_universal
        self.v = v
        self.dx = dx
        self.mu = mu

    def forward(self, z: torch.Tensor):
        """Value of the Energy function given the potential

        Returns:
            [pt.tensor]: [The energy values of the different samples. shape=(n_istances)]
        """
        # self.Func.eval()
        x, eng_1 = self.model(z)
        eng_1 = eng_1.view(x.shape[0])
        eng_2 = torch.einsum("ai,i->a", x, self.v) * self.dx
        norm = torch.sum(x, axis=1) * self.dx
        eng_2_trapz = torch.trapz(x * self.v[None, :], dim=1, dx=self.dx)
        # eng_2 = pt.trapezoid(eng_2, dx=self.dx, dim=1)
        return eng_1 + eng_2, eng_1 + eng_2_trapz, norm, x

    def soft_constrain(self, z: torch.Tensor):
        eng,eng_exact, norm, x = self.forward(z)
        cons = self.mu * (norm - 1) ** 2
        return eng + cons, eng_exact, x

    def batch_calculation(self, z: torch.Tensor):
        x, eng_1 = self.model(z)
        eng_1 = eng_1.view(x.shape[0])
        eng_2 = self.dx*torch.einsum('ai,ai->a',self.v,x)
        eng_2_trapz=torch.trapz(self.v*x,dx=self.dx,dim=1)
        return eng_1 + eng_2,eng_1+eng_2_trapz, x

    def ml_calculation(self,x:torch.Tensor):

        eng_1 = self.model.DFTModel(x.unsqueeze(1)).squeeze()
        eng_2 = self.dx*torch.einsum('ai,ai->a',self.v,x)
        eng_2_trapz=torch.trapz(self.v*x,dx=self.dx,dim=1)
        return eng_1 + eng_2,eng_1+eng_2_trapz, x

class DenseVAE(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_neurons:List,
        input_size: int, 
    ):

        super().__init__()
        
        self.latent_dimension=latent_dimension
        self.encoder = nn.ModuleList([])
        self.input_size=input_size
        for i in range(len(hidden_neurons)):
            if i==0:
                self.encoder.append(nn.Linear(input_size,hidden_neurons[i]))
            else:
                self.encoder.append(nn.Linear(hidden_neurons[i-1],hidden_neurons[i]))
            self.encoder.append(nn.ReLU())
        self.final_mu=nn.Linear(hidden_neurons[-1],latent_dimension)
        self.latent_logvar=nn.Linear(hidden_neurons[-1],latent_dimension)    
        self.encoder=nn.Sequential(*self.encoder)

        self.decoder=nn.ModuleList([])
        for i in range(len(hidden_neurons)):
            if i==0:
                self.decoder.append(nn.Linear(latent_dimension,hidden_neurons[-1-i]))
            else:
                self.decoder.append(nn.Linear(hidden_neurons[-i],hidden_neurons[-1-i]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(hidden_neurons[0],input_size))
        #self.decoder.append(nn.Sigmoid())
        self.decoder=nn.Sequential(*self.decoder)

    def forward(self, x):
        #x=x.squeeze(1)
        x = self.encoder(x)
        latent_mu=self.final_mu(x)
        latent_logvar=self.latent_logvar(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        #x_recon=x_recon.unsqueeze(1)
        x_recon=x_recon.cos()
        return x_recon, latent_mu, latent_logvar


class DFTVAEDense(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        loss_generative: nn.Module,
        loss_dft: nn.Module,
        output_size: int,
        hidden_neurons:List
    ):

        super().__init__()

        self.loss_generative = loss_generative
        self.loss_dft = loss_dft

        self.VAE=DenseVAE(latent_dimension=latent_dimension,hidden_neurons=hidden_neurons,input_size=input_size)

        self.DFTModel = Pilati_model_3_layer(
            input_size=input_size,
            input_channel=input_channels,
            hidden_channel=hidden_channels,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            output_size=output_size,
        )

    def forward(self, z: torch.Tensor):
        x = self.VAE.decoder(z)
        f = self.DFTModel(x)
        x = x.view(x.shape[0], -1)
        return x, f

    def proposal(self, z: torch.Tensor):
        x = self.VAE.decoder(z)
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
        x = x.to(device=device)
        w=self.VAE.encoder(x)
        latent_mu=self.VAE.final_mu(w)
        latent_logvar=self.VAE.latent_logvar(w)
        #latent_mu, latent_logvar = self.VAE.encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.VAE.decoder(latent)
        #print(x_recon.shape)
        loss, kldiv = self.loss_generative(x_recon, x, latent_mu, latent_logvar)
        return loss, kldiv.item()

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
    
    def reconstruct(self,x):
        #x = x.to(device=device)
        w=self.VAE.encoder(x)
        latent_mu=self.VAE.final_mu(w)
        latent_logvar=self.VAE.latent_logvar(w)
        #latent_mu, latent_logvar = self.VAE.encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.VAE.decoder(latent)
        return x_recon

