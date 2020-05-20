import torch

from parameters.Parameters import Parameters
from prototype.Module import Module
from utils.utils import kaiming_init
from utils.utils import xavier_init


class Linear(Module):
    """Fully connected layer"""
    
    def __init__(self, size_input, size_output, use_bias=True, activ='ReLu'):
        super().__init__()

        self.size_input = size_input
        self.size_output = size_output
        self.activ = activ
        self.use_bias = use_bias
        self.batch_size = 0

        self.weights = Parameters(size_input, size_output)
        if activ.lower() == 'relu':
            self.weights.data = kaiming_init(size_input, size_output)
        elif activ.lower() == 'tanh':
            self.weights.data = xavier_init(size_input, size_output)
        else:
            raise NotImplementedError

        if use_bias:
            self.bias = Parameters(1, size_output)
            self.bias.data.normal_(mean=0, std=0.1)
        else:
            self.bias = None

    def __repr__(self):
        return f"<Linear ({self.size_input}, {self.size_output})>"

    def forward(self, inputs: torch.Tensor):
        """Perform forward pass"""
        if self.requires_grad:
            self.cache = inputs
        else:
            self.cache = None
        self.batch_size = inputs.shape[0]

        output = inputs.matmul(self.weights.data)
        if self.bias is not None:
            output += self.bias.data

        return output

    def backward(self, grad):
        """ update internal gradients
            grad: batch_size x out_size
            grad_bias 1 x out_size
            grad_weights in_size x out_size"""

        # https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html

        self.weights.update_grad(self.cache.t().matmul(grad))

        if self.bias is not None:
            self.bias.update_grad(torch.ones(1, self.batch_size).matmul(grad))

        grad = grad.matmul(self.weights.data.t())

        return grad

    def zero_grad(self):
        """reset the gradient for weights and biases"""
        self.weights.zero_grad()
        if self.bias is not None:
            self.bias.zero_grad()

    def init(self):
        """Reinitialise parameters --> need to reinit the optim also"""
        self.__init__(
            size_input=self.size_input,
            size_output=self.size_output,
            activ=self.activ,
            use_bias=self.use_bias,
        )