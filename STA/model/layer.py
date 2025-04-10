#!/usr/bin/env python

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from dgl.nn.pytorch import GATConv


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    From
    Xiong L, Tian K, Li Y, et al. Online single-cell data integration through projecting heterogeneous datasets
    into a common cell-embedding space[J]. Nature Communications, 2022, 13(1): 6118.
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """

        :param num_features:
        :param n_domain:
        :param eps:
        :param momentum:
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for _ in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, data):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        return out


def reparameterize(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, input_dim, h_dim=16, multiple=8, num_heads=2):
        """

        :param input_dim:
        :param h_dim:
        :param multiple:
        :param num_heads:
        """
        super().__init__()
        input_dim = input_dim
        input_dim2 = h_dim * multiple
        h_dim = h_dim
        # encode
        self.conv1 = GATConv(input_dim, input_dim2, num_heads=1, bias=True, activation=None)
        self.conv2 = GATConv(input_dim2, input_dim2, num_heads=num_heads, bias=True, activation=None)

        # reparameterize
        self.conv_mean = GATConv(input_dim2, h_dim, num_heads=num_heads, bias=True, activation=None)
        self.conv_var = GATConv(input_dim2, h_dim, num_heads=num_heads, bias=True, activation=None)

        # reset_parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv_mean.reset_parameters()
        self.conv_var.reset_parameters()

    def forward(self, x, block=None):
        """
        """
        o = self.conv1(block, x).mean(-2).view(x.shape[0], -1)
        o = self.conv2(block, o).mean(-2).view(x.shape[0], -1) + o
        mean = self.conv_mean(block, o).mean(-2).view(x.shape[0], -1)
        var = torch.exp(self.conv_var(block, o).mean(-2).view(x.shape[0], -1))
        z = reparameterize(mean, var)

        return z, mean, var


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, h_dim, input_dim, multiple=8, n_domain=None, num_heads=2):
        """

        :param h_dim:
        :param input_dim:
        :param multiple:
        :param n_domain:
        """
        super().__init__()
        input_dim2 = h_dim * multiple
        h_dim = h_dim
        # encode
        self.conv1 = GATConv(h_dim, input_dim2, num_heads=num_heads, bias=True, activation=None)
        self.norm1 = DSBatchNorm(input_dim2, n_domain)
        self.act1 = nn.LeakyReLU()
        self.conv2 = GATConv(input_dim2, input_dim, num_heads=1, bias=True, activation=None)
        self.act2 = nn.ReLU()
        self.act3 = nn.Sigmoid()

        # reset_parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm1.reset_parameters()

    def forward(self, x, y=None, block=None):
        o = self.conv1(block, x).mean(-2).view(x.shape[0], -1)
        o = self.act1(self.norm1(o, y))  # LeakyReLU
        o = self.conv2(block, o).mean(-2).view(x.shape[0], -1)
        o = self.act2(o)
        o_sig = self.act3(o)
        return o, o_sig
