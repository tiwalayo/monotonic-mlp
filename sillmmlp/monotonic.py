import math

import torch
import torch.nn as nn
import numpy as np


class Monotonic(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = latent
        self.size = latent.shape

    def getweights(self):
        return torch.exp(self.latent)


class MonotonicGroup(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / math.sqrt(in_features)
        self.latent_weights = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-stdv, stdv))
        self.latent_bias = nn.Parameter(torch.Tensor(out_features).uniform_(-stdv, stdv))

        self.weights = Monotonic(self.latent_weights)
        self.bias = Monotonic(self.latent_bias)

    def forward(self, x):
        return F.linear(x, self.weights.getweights(), self.bias.getweights())


class MonotonicLinear(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return [g.forward(x) for g in self.groups]


class MonotonicMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(tuple(torch.max(i, dim=1)[0].unsqueeze(1) for i in x), dim=1)


class MonotonicMin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.min(x, dim=1)[0].unsqueeze(1)