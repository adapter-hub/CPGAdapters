import logging
import math

# import lang2vec.lang2vec as l2v
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


# class UrielMlpProperty(nn.Module):
#
#     def __init__(self, name, dim, languages, dropout=0.1):
#         super().__init__()
#         logging.info('Initialising URIEL embedding "%s" with dim %d and languages %s' % (
#                 name, dim, ', '.join(languages)))
#         self.name = name
#         self.dim = dim
#         self.features = l2v.get_features(
#                 languages, 'syntax_knn+phonology_knn+inventory_knn')
#         self.features = {
#                 lang: torch.Tensor(vec)
#                 for lang, vec in self.features.items()
#         }
#         self.n_features = None
#         for vec in self.features.values():
#             if self.n_features is None:
#                 self.n_features = len(vec)
#             else:
#                 assert len(vec) == self.n_features
#         self.layer_1 = nn.Linear(self.n_features, self.dim)
#         self.layer_2 = nn.Linear(self.dim, self.dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, language):
#         features = self.features[language].to(self.layer_1.weight.device)
#         embedding = self.layer_1(features)
#         embedding = F.relu(embedding)
#         embedding = self.layer_2(embedding)
#         embedding = F.relu(embedding)
#         embedding = self.dropout(embedding)
        return embedding


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
        self.down_dim = cpg_config['down_dim']
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

            self.cpg_down = self.config.down_dim

            self.down_dim = torch.nn.Linear(self.config.context_dim, self.cpg_down)
            self.dropout = torch.nn.Dropout(0.3)

            # params = nn.Parameter(
            #         torch.Tensor( *shape, self.config.context_dim))
            params = nn.Parameter(
                    torch.Tensor( *shape, self.cpg_down))
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
                # return torch.matmul(self.__getattr__(name), context_embedding)
                # return torch.matmul(context_embedding, self.__getattr__(name))
                context_embedding = self.down_dim(self.dropout(context_embedding))
                if len(self.__getattr__(name).shape)> 2:

                    # return torch.matmul(context_embedding, self.__getattr__(name)).transpose(0,1)
                    # return torch.matmul(context_embedding, self.__getattr__(name).transpose(1,2)).transpose(0,1)
                    return torch.matmul(self.__getattr__(name), context_embedding.T).permute(2,1,0)
                else:
                    return torch.matmul( self.__getattr__(name), context_embedding.T).T
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
#
# class Head(CpgModule):
#
#     def __init__(self, in_features, bias=True, config=None, max_length=None, max_label_length=None,layers=2, activation_function=None, hidden_dropout_prob=0.0):
#         class c:
#             context_dim = in_features
#         config = c()
#         super().__init__(config)
#         self.in_features = in_features
#         self.max_length = max_length
#         self.max_label_length = max_label_length
#         self.dropout = torch.nn.Dropout(0.3)
#         # self.label_embedder = nn.Linear(self.in_features , self.in_features)
#         self.label_embedder =
#
#         def gelu_new(x):
#             """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
#                 Also see https://arxiv.org/abs/1606.08415
#             """
#             return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
#         self.non_lin = gelu_new
#
#         pred_linear = []
#
#         for l in range(layers):
#             pred_linear.append(nn.Dropout(hidden_dropout_prob))
#             if l < layers - 1:
#                 pred_linear.append(nn.Linear(self.in_features, self.in_features))
#                 pred_linear.append(activation_function)
#
#         self.pred_linear = nn.Sequential(*pred_linear)
#
#         cpg_linear = []
#
#         for l in range(layers):
#             cpg_linear.append(nn.Dropout(hidden_dropout_prob))
#             if l < layers - 1:
#                 cpg_linear.append(nn.Linear(self.in_features, self.in_features))
#                 cpg_linear.append(activation_function)
#
#         self.cpg_linear = nn.Sequential(*cpg_linear)
#
#         self.add_param('weight', (1, in_features))
#         if bias:
#             self.add_param('bias', (1,))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.config:
#             self.init_params()
#         else:
#             nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#             if self.bias is not None:
#                 fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#                 bound = 1 / math.sqrt(fan_in)
#                 nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, pred_input, contextual_inputs, context_embedding=None, max_length=None, max_label_length=None,attention_mask=None):
#
#         if (max_label_length is None and self.max_label_length is None) or (max_length is None and self.max_length is None):
#             raise Exception("Max Lengths are not set for CPG Head, cannot compute parameters.")
#
#         max_length = max_length if max_length is not None else self.max_length
#         max_label_length = max_label_length if max_label_length is not None else self.max_label_length
#
#         input_length = contextual_inputs.shape[1]
#
#         weight = None
#         bias = None
#
#         contextual_inputs = self.non_lin(self.cpg_linear(contextual_inputs))
#
#         for i in range(max_length, input_length, max_label_length):
#             c = torch.sum(contextual_inputs[:, i + 1:i + max_label_length - 1, :] * attention_mask[:, i + 1:i + max_label_length - 1][:, :, None].repeat(1, 1, contextual_inputs.shape[-1]), 1) / torch.sum(attention_mask[:, i + 1:i + max_label_length - 1], -1)[:,None].repeat(1,contextual_inputs.shape[-1])
#             # c = torch.sum(contextual_inputs[:, i + 1:i + max_label_length - 1, :] * attention_mask[:, i + 1:i + max_label_length - 1][:, :, None].repeat(1, 1, contextual_inputs.shape[-1]), 1)
#
#
#             if weight is None:
#                 # torch.sum(contextual_inputs[:, i + 1:i + max_label_length - 1, :] * attention_mask[:,
#                 #                                                                     i + 1:i + max_label_length - 1][:,
#                 #                                                                     :, None].repeat(1, 1,
#                 #                                                                                     contextual_inputs.shape[
#                 #                                                                                         -1]), 1)
#                 # torch.sum(attention_mask[:, i + 1:i + max_label_length - 1], -1)
#                 weight = self.eval_param('weight', context_embedding=c)
#                 # weight = self.eval_param('weight', context_embedding=contextual_inputs[:,i+1,:])
#                 # weight = self.eval_param('weight', context_embedding=contextual_inputs[:,i,:])
#                 bias = self.eval_param('bias', context_embedding=c)
#                 # bias = self.eval_param('bias', context_embedding=contextual_inputs[:,i+1,:])
#                 # bias = self.eval_param('bias', context_embedding=contextual_inputs[:,i,:])
#             else:
#                 weight = torch.cat((weight, self.eval_param('weight', context_embedding=c)), 2)
#                 # weight = torch.cat((weight, self.eval_param('weight', context_embedding=contextual_inputs[:,i+1,:])), 2)
#                 # weight = torch.cat((weight, self.eval_param('weight', context_embedding=contextual_inputs[:,i,:])), 2)
#                 bias = torch.cat((bias, self.eval_param('bias', context_embedding=c)), 1)
#                 # bias = torch.cat((bias, self.eval_param('bias', context_embedding=contextual_inputs[:,i+1,:])), 1)
#                 # bias = torch.cat((bias, self.eval_param('bias', context_embedding=contextual_inputs[:,i,:])), 1)
#
#         output = self.pred_linear(pred_input)
#
#         return torch.matmul(output[:,None,:], weight).squeeze(1) +bias
#         # return F.linear(output, weight, bias)

class Linear(CpgModule):

    def __init__(self, in_features, out_features, bias=True, config=None, ):
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
        # return F.linear(input, weight, bias)
        if context_embedding is  None:
            return F.linear(input, weight, bias)
        else:
            if len(input.shape)>2:
                return torch.matmul(input, weight) + bias[:, None, :].repeat(1, input.shape[1], 1)
            else:
                return torch.matmul(input[:,None,:], weight).squeeze(1) +bias

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

