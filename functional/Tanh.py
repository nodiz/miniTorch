import torch
import math
from prototype.Module import Module


def tanh(inputs):
    #return (math.exp(inputs) - math.exp(-inputs)) / (math.exp(inputs) + math.exp(-inputs))
    return inputs.tanh()


def dtanh(inputs):
    return


class Tanh(Module):
    """Tanh activation layer"""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.tanh()

    def backward(self, inputs: torch.Tensor) -> torch.Tensor:
        return - inputs.tanh().pow(2) + 1
