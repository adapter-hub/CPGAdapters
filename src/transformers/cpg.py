import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CpgModule(nn.Module):

    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.cpg_params = []

    def add_params(self, name, shape):
        params = nn.Parameter(torch.Tensor(*shape, self.context_dim))
        self.register_parameter(name, params)
        self.cpg_params.append(params)

    def eval_params(self, name, context_embedding):
        return torch.matmul(self.__getattr__(name), context_embedding)

    def reset_params(self):
        std = 1.0 / self.context_dim
        for params in self.cpg_params:
            nn.init.normal_(params, b=std)


class Linear(CpgModule):

    def __init__(self, context_dim, in_features, out_features, bias=True):
        super().__init__(context_dim)
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.add_params('weight', (in_features, out_features))
        if bias:
            self.add_params('bias', (out_features,))
        self.reset_params()

    def forward(self, input, context_embedding):
        weight = self.eval_params('weight', context_embedding)
        if self.has_bias:
            bias = self.eval_params('bias', context_embedding)
        else:
            bias = None
        return F.linear(input, weight, bias)


class LayerNorm(CpgModule):

    def __init__(self, context_dim, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(context_dim)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.add_params('weight', normalized_shape)
            self.add_params('bias', normalized_shape)
        self.reset_params()

    def forward(self, input, context_embedding):
        if self.elementwise_affine:
            weight = self.eval_params('weight', context_embedding)
            bias = self.eval_params('bias', context_embedding)
        else:
            weight = None
            bias = None

        return F.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps)


