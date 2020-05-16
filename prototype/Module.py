import torch
from parameters.Parameters import Parameters


class Module:
    def __init__(self, *args):
        self.requires_grad = True
        self.cache = None

    def __call__(self, *arg):
        return self.forward(*arg)

    def forward(self, *inputs):
        """forward should get iterable input, and returns a tensor or a tuple of tensors."""
        raise NotImplementedError

    def backward(self, *inputs):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        raise NotImplementedError

    def zero_grad(self):
        self.cache = None
        for param in self.param():
            param.zero_grad()

    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [var for var in vars(self).values() if isinstance(var, Parameters)]

    def train(self):
        self.requires_grad = True

    def eval(self):
        self.requires_grad = False

    @property
    def eval_mode(self):
        return not self.requires_grad

    def init(self):
        return
