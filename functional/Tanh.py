import torch
from prototype.Module import Module


class Tanh(Module):
    """Tanh activation layer"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.requires_grad:
            self.cache = inputs

        return inputs.tanh()

    def backward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtanh = - self.cache.tanh().pow(2) + 1
        return dtanh * inputs
