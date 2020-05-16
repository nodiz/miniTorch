import torch


class Optim:
    def __init__(self, parameters, lr):
        """Optimizer, receive iterable as parameter"""
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Perform optimizer step"""
        for param in self.parameters:
            param.update_data(self.update(param))

    def update(self, data):
        """Calculate optimizer step"""
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class SGD(Optim):
    def __init__(self, parameters, lr, momentum=0):
        """Optimizer, receive iterable as parameter"""
        super().__init__(parameters, lr)
        self.momentum = momentum

    def update(self, data):
        v = data.cache.pop('v', torch.zeros_like(data.grad))
        v = self.momentum * v - self.lr * data.grad
        data.cache['v'] = v
        return v


class BengioSGD(Optim):
    def __init__(self, parameters, lr, momentum=0):
        """Optimizer, receive iterable as parameter"""
        super().__init__(parameters, lr)
        self.momentum = momentum

    def update(self, data):
        v = data.cache.pop('v', torch.zeros_like(data.grad))
        update = self.momentum ** 2 * v - (1 - self.momentum) * self.lr * data.grad
        v = self.momentum * v - self.lr * data.grad
        data.cache['v'] = v
        return update


class Adam(Optim):  # TODO currently not working
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-08):
        """Optimizer, receive iterable as parameter"""
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

    def update(self, data):
        m = data.cache.pop('m', torch.zeros_like(data.grad))
        v = data.cache.pop('v', torch.zeros_like(data.grad))
        t = data.cache.pop('t', 0)
        m = self.beta1 * m + (1 - self.beta1) * data.grad
        v = self.beta2 * v + (1 - self.beta2) * data.grad.pow(2)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        update = - self.lr * m_hat / (v_hat.pow(0.5) + self.eps)
        data.cache['v'] = v
        data.cache['m'] = m
        data.cache['t'] = t + 1
        return update
