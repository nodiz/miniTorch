import torch
from prototype.Module import Module


class ReLu(Module):
    """ReLu activation layer"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor):
        if self.requires_grad:
            self.cache = inputs

        zeros = torch.zeros_like(inputs)
        return inputs.where(inputs > 0, zeros)

    def backward(self, inputs: torch.Tensor):
        zeros = torch.zeros_like(inputs)
        return inputs.where(self.cache > 0, zeros)


class LeakyReLu(Module):
    """LeakyReLu activation layer"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        if self.requires_grad:
            self.cache = inputs

        return inputs.where(inputs > 0, inputs/10)

    def backward(self, inputs):
        ones = torch.ones_like(inputs)
        return inputs.where(inputs > 0, inputs/10)
