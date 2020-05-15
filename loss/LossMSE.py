import torch
import math
from prototype.Module import Module


class LossMse(Module):
    """Loss MSE"""

    def __call__(self, inputs, targets):
        loss = self.forward(inputs, targets)
        dloss = self.backward(inputs, targets)
        return loss, dloss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return ((inputs - targets).pow(2).sum(1)).mean().item()

    def backward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 2 * (inputs - targets)
