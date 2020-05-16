import torch


class Parameters:
    def __init__(self, *args, **kwargs):
        """parameters definition, data and grad couple"""
        self.data = torch.empty(args)
        self.grad = torch.zeros_like(self.data)
        self._cache = {}
        self.requires_grad = kwargs.pop("requires_grad", True)

    def __call__(self):
        return self.data, self.grad

    def zero_grad(self):
        self.grad = torch.zeros_like(self.data)
        return self.grad

    def update_data(self, update):
        self.data += update
        return self.data

    def update_grad(self, update):
        self.grad += update
        return self.grad

    def train(self):
        self.requires_grad = True

    def eval(self):
        self.requires_grad = False

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache
