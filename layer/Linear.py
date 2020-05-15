import torch
from prototype.Module import Module
from utils.utils import xavier_init
from utils.utils import kaiming_init


class Linear(Module):
    def __init__(self, size_input, size_output, use_bias=True, activ='ReLu'):
        super().__init__()

        self.size_input = size_input
        self.size_output = size_output
        self.batch_size = 0

        if activ.lower() == 'relu':
            self.weights = kaiming_init(size_input, size_output)
        elif activ.lower() == 'tanh':
            self.weights = xavier_init(size_input, size_output)
        else:
            raise NotImplementedError

        self.grad_weights = torch.zeros_like(self.weights)

        if use_bias:
            self.bias = torch.zeros(1, self.size_output) \
                .normal_(mean=0, std=1)
            self.grad_bias = torch.zeros_like(self.bias)
        else:
            self.bias, self.grad_bias = None, None

    def __repr__(self):
        return f"<Linear ({self.size_input}, {self.size_output})>"

    def forward(self, inputs):
        """perform forward pass"""
        if self.requires_grad:
            self.cache = inputs
        else:
            self.cache = None
        self.batch_size = inputs.shape[0]

        output = inputs.matmul(self.weights)
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, inputs):
        """update internal gradients
            inputs: batch_size x out_size
            grad_bias 1 x out_size
            grad_weights in_size x out_size"""

        # https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html

        self.grad_weights += self.cache.t().matmul(inputs)
        
        if self.bias is not None:
            self.grad_bias += torch.ones(1,self.batch_size).matmul(inputs)
        
        grad = inputs.matmul(self.weights.t())
        
        return grad

    def zero_grad(self):
        """reset the gradient for weights and biases"""
        self.grad_weights = torch.zeros_like(self.weights)
        if self.bias is not None:
            self.grad_bias = torch.zeros_like(self.bias)

    def optim_sgd_step(self, eta):
        """update weights and biases"""
        self.weights = self.weights - eta * self.grad_weights

        if self.bias is not None:
            self.bias = self.bias - eta * self.grad_bias
