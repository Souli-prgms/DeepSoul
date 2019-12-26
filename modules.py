import numpy as np
import math

import functional


class Module:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise Exception('not implemented')

    def backward(self, out_grad=None):
        return self.bwd(out_grad, self.out, *self.args)

    def parameters(self):
        return self.params

    def gradients(self):
        return self.grads


class Relu(Module):
    def forward(self, inp):
        return np.where(inp > 0, inp, 0)

    def bwd(self, out_grad, out, inp):
        return out_grad * np.where(inp > 0, 1, 0)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        kaiming = math.sqrt(1.0 / in_features)
        self.params['w'] = np.random.rand(in_features, out_features) * kaiming # weights
        self.params['b'] = np.zeros(out_features) # biases

    def forward(self, x):
        return x @ self.params['w'] + self.params['b']

    def bwd(self, out_grad, out, inp):
        self.grads['w'] = (inp.T @ out_grad) / inp.shape[0]
        self.grads['b'] = out_grad.sum(0) / inp.shape[0]

        return out_grad @ self.params['w'].T


class CrossEntropy(Module):
    def forward(self, inp, targ):
        return functional.cross_entropy(inp, targ)

    def bwd(self, out_grad, out, inp, targ):
        return inp - targ


class Conv2d(Module):
    def __init__(self, nb_filters, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.params['f'] = np.random.randn(kernel_size, kernel_size, nb_filters) / (kernel_size ** 2)
        self.kernel_size, self.padding, self.stride = kernel_size, padding, stride

    def forward(self, x):
        return functional.conv2d(x, self.params['f'], self.kernel_size, self.padding, self.stride)

    def bwd(self, out_grad, out, inp):
        self.grads['f'], inp_grad = functional.conv2d_backprop(inp, out_grad, self.params['f'], self.kernel_size, self.padding, self.stride)
        return inp_grad


class Maxpool2d(Module):
    def forward(self, x):
        return functional.maxpool2d(x)

    def bwd(self, out_grad, out, inp):
        return functional.maxpool2d_backprop(inp, out_grad)


class Flatten(Module):
    def forward(self, x):
        bs, w, h, c = x.shape
        return np.reshape(x, (bs, w * h * c))

    def bwd(self, out_grad, out, inp):
        return np.reshape(out_grad, inp.shape)


