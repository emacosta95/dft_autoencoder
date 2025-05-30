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
            nn.Linear(hidden_channel * int(input_size / pooling_size**3), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_1(x)
        x = self.model_2(x)
        x = self.model_3(x)
        x = self.flat(x)
        x = self.final_dense(x)
        return x


class DFTModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channel: int,
        hidden_channel: List,
        output_size: int,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        pooling_size: int,
        activation: torch.nn.Module,
    ):
        super().__init__()

        self.conv_list = nn.ModuleList([])

        self.conv_list.add_module(
            "block_0",
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channel,
                    out_channels=hidden_channel[0],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.Softplus(),
                nn.AvgPool1d(kernel_size=pooling_size),
            ),
        )

        for i in range(len(hidden_channel) - 1):
            self.conv_list.add_module(
                f"block_{i+1}",
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=hidden_channel[i],
                        out_channels=hidden_channel[i + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode,
                    ),
                    nn.Softplus(),
                    nn.AvgPool1d(kernel_size=pooling_size),
                ),
            )

        self.flat = nn.Flatten()

        self.final_dense = nn.Sequential(
            nn.Linear(
                hidden_channel[-1]
                * int(input_size / pooling_size ** len(hidden_channel)),
                output_size,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_list:
            x = conv(x)
        x = self.flat(x)
        x = self.final_dense(x)
        return x


class DFTModel3D(nn.Module):
    def __init__(
        self,
        input_size: List,
        input_channel: int,
        hidden_channel: List,
        output_size: int,
        kernel_size: Tuple,
        padding: List,
        padding_mode: str,
        pooling_size: Tuple,
        activation: torch.nn.Module,
    ):
        super().__init__()

        self.conv_list = nn.ModuleList([])

        self.conv_list.add_module(
            "block_0",
            nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channel,
                    out_channels=hidden_channel[0],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.Softplus(),
                nn.BatchNorm3d(hidden_channel[0]),
                # nn.Conv3d(
                #     in_channels=hidden_channel[0],
                #     out_channels=hidden_channel[0],
                #     kernel_size=kernel_size,
                #     stride=1,
                #     padding=padding,
                #     padding_mode=padding_mode,
                # ),
                # nn.Softplus(),
                nn.AvgPool3d(kernel_size=pooling_size),
            ),
        )

        for i in range(len(hidden_channel) - 1):
            self.conv_list.add_module(
                f"block_{i+1}",
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=hidden_channel[i],
                        out_channels=hidden_channel[i + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode,
                    ),
                    nn.Softplus(),
                    nn.BatchNorm3d(hidden_channel[i + 1]),
                    # # nn.Conv3d(
                    # #     in_channels=hidden_channel[i + 1],
                    # #     out_channels=hidden_channel[i + 1],
                    # #     kernel_size=kernel_size,
                    # #     stride=1,
                    # #     padding=padding,
                    # #     padding_mode=padding_mode,
                    # # ),
                    # # nn.Softplus(),
                    nn.AvgPool3d(kernel_size=pooling_size),
                ),
            )

        self.flat = nn.Flatten()

        self.final_dense = nn.Sequential(
            nn.Linear(
                hidden_channel[-1]
                * int(
                    (input_size[0] // pooling_size[0] ** len(hidden_channel))
                    * (input_size[1] // pooling_size[1] ** len(hidden_channel))
                    * (input_size[2] // pooling_size[2] ** len(hidden_channel))
                ),
                output_size,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_list:
            x = conv(x)
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
                hidden_channel * int(linear_input_size / pooling_size**3) ** 3, 1
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
        self.block.add_module(f"{0} act", nn.Softplus())
        for i in range(1, len(hidden_neurons) - 1):
            self.block.add_module(
                f"{i} layer", nn.Linear(hidden_neurons[i], hidden_neurons[i + 1])
            )
            self.block.add_module(f"{i} act", nn.Softplus())
        self.block.add_module(f"{-1} layer", nn.Linear(hidden_neurons[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x
