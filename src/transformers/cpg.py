import logging
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Property(nn.Module):

    def __init__(self, name, dim, values):
        super().__init__()
        logging.info('Initialising CPG property "%s" with dim %d and values %s' % (
                name, dim, ', '.join(values)))
        self.name = name
        self.dim = dim
        std = 1.0 / self.dim
        for value in values:
            name = value + '_embedding'
            embedding = nn.Parameter(torch.Tensor(self.dim))
            nn.init.normal_(embedding, std=std)
            self.register_parameter(name, embedding)

    def forward(self, value):
        name = value + '_embedding'
        return self.__getattr__(name)


class Environment(nn.Module):

    def __init__(self, properties):
        super().__init__()
        self.properties = properties
        self.dim = 0
        for p in properties:
            self.dim += p.dim
            self.add_module(p.name, p)

    def forward(self, context):
        property_embeddings = []
        for p in self.properties:
            if p.name in context:
                value = context[p.name]
                property_embeddings.append(p(value))

        return torch.cat(property_embeddings, dim=0)


class CpgModuleConfig:

    def __init__(self, cpg_config, include_layer=True):
        self.context_dim = cpg_config['language_embedding_dim']
        if cpg_config['layer_embedding_dim'] and include_layer:
            self.context_dim += cpg_config['layer_embedding_dim']
        # could have other attributes such as non-linearity to be applied after
        # matmul


class CpgModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def add_param(self, name, shape):
        if self.config:
            params = nn.Parameter(
                    torch.Tensor(*shape, self.config.context_dim))
        else:
            params = nn.Parameter(torch.Tensor(*shape))
        self.register_parameter(name, params)

    def eval_param(self, name, context_embedding=None):
        if self.config:
            if context_embedding is None:
                raise ValueError(
                        'Tried to evaluate contextually generated parameter %s '
                        'without supplying context embedding' % name)
            else:
                return torch.matmul(self.__getattr__(name), context_embedding)
        else:
            return self.__getattr__(name)

    def init_params(self):
        if not self.config:
            raise ValueError(
                    'init_params was called on non-contextually generated '
                    'module')
        std = 1.0 / self.config.context_dim
        for param in self.parameters():
            nn.init.normal_(param, std=std)


class Linear(CpgModule):

    def __init__(self, in_features, out_features, bias=True, config=None):
        super().__init__(config)
        self.in_features = in_features
        self.out_features = out_features

        self.add_param('weight', (out_features, in_features))
        if bias:
            self.add_param('bias', (out_features,))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.config:
            self.init_params()
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, context_embedding=None):
        weight = self.eval_param('weight', context_embedding=context_embedding)
        bias = self.eval_param('bias', context_embedding=context_embedding)
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Sequential(nn.Module):

    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input, context_embedding=None):
        for module in self._modules.values():
            if isinstance(module, CpgModule):
                input = module(input, context_embedding=context_embedding)
            else:
                input = module(input)
        return input

