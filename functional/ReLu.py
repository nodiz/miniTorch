import torch
from prototype.Module import Module


class ReLu(Module):
    """ReLu activation layer"""

    def forward(self, inputs):
        zeros = torch.zeros_like(inputs)
        return inputs.where(inputs > 0, zeros)

    def backward(self, inputs):
        zeros = torch.zeros_like(inputs)
        ones = torch.ones_like(inputs)
        return ones.where(inputs > 0, zeros)


class LeakyReLu(Module):
    """LeakyReLu activation layer"""

    def forward(self, inputs):
        return inputs.where(inputs > 0, inputs/10)

    def backward(self, inputs):
        ones = torch.ones_like(inputs)
        return inputs.where(inputs > 0, ones, ones/10)
