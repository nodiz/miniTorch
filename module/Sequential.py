import torch
from prototype.Module import Module
from collections import OrderedDict


class Sequential(Module):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self._modules = OrderedDict()
        for i, layer in enumerate(layers):
            name = layer[0]
            module = layer[1]
            self.appendModule(name,module)

    def __repr__(self):
        str = ""
        for i, name, module in enumerate(self._modules.items()):
            str += f"{i}: ({name})"
        return str

    def appendModule(self, name, module):
        self._modules[name] = module

    def children(self):
        for name, module in self._modules.items():
            if module is not None:
                yield module

    def inverse_children(self):
        for module in [module for module in self.children()][::-1]:
            if module is not None:
                yield module

    def forward(self, inputs):
        for module in self.children():
            inputs = module.forward(inputs)
        return inputs

    def backward(self, inputs: torch.Tensor):
        for module in self.inverse_children():
            inputs = module.backward(inputs.type(torch.float))
        return inputs

    def optim_sgd_step(self, eta):
        for module in self.children():
            module.optim_sgd_step(eta)

    def train(self):
        for module in self.children():
            module.requires_grad = True

    def eval(self):
        for module in self.children():
            module.requires_grad = False

    def zero_grad(self):
        for module in self.children():
            module.zero_grad()