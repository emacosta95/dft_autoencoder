import torch
import torch.nn as nn
from src.training.utils import VaeLossMSE


class DFTVAELoss(nn.Module):
    def __init__(self, loss_parameter: float, variational_beta: float) -> None:
        super().__init__()

        self.loss_dft = nn.MSELoss(reduction="sum")
        self.loss_vae = VaeLossMSE(variational_beta=variational_beta)
        self.loss_parameter = loss_parameter

    def forward(self, x, y, x_recon, y_pred, mu, logvar):
        l1 = self.loss_dft(y, y_pred)
        l2, _ = self.loss_vae(x, x_recon, mu, logvar)
        return (
            l1 * self.loss_parameter + (1 - self.loss_parameter) * l2,
            l1,
            l2,
        )
