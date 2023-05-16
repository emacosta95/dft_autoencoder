from re import X
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchmetrics import R2Score
from zmq import device


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
        activation: torch.nn.Module,
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
            nn.AvgPool1d(kernel_size=pooling_size),
        )

        self.flat = nn.Flatten()

        self.final_dense = nn.Sequential(
            nn.Linear(hidden_channel * int(input_size / pooling_size ** 3), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.model_1(x)
        x = self.model_2(x)
        x = self.model_3(x)
        x = self.flat(x)
        x = self.final_dense(x)
        return x


class Pilati_model_3d_3_layer(nn.Module):
    def __init__(
        self,
        linear_input_size: int,
        input_channel: int,
        hidden_channel: int,
        output_size: int,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        pooling_size: int,
        activation: str,
    ):

        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.model_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
        )

        self.model_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
        )

        self.model_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
        )

        self.flat = nn.Flatten()

        self.final_dense = nn.Sequential(
            nn.Linear(
                hidden_channel * int(linear_input_size / pooling_size ** 3) ** 3, 1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.model_1(x)
        x = self.model_2(x)
        x = self.model_3(x)
        x = self.flat(x)
        x = self.final_dense(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, hidden_neurons: List, latent_space: int, activation: str):
        super().__init__()
        activation = getattr(torch.nn, activation)()
        self.block = nn.Sequential()
        self.block.add_module(f"{0} layer", nn.Linear(latent_space, hidden_neurons[0]))
        self.block.add_module(f"{0} act", activation)
        for i in range(1, len(hidden_neurons) - 2):
            self.block.add_module(
                f"{i} layer", nn.Linear(hidden_neurons[i], hidden_neurons[i + 1])
            )
            self.block.add_module(f"{i} act", activation)
        self.block.add_module(f"{-1} layer", nn.Linear(hidden_neurons[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.block(x)
        return x
