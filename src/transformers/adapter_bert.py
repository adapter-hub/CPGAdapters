import logging

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import math
from . import cpg

from .cpg import CpgModule
from .adapter_config import DEFAULT_ADAPTER_CONFIG, AdapterType
from .adapter_model_mixin import ModelAdaptersMixin, ModelWithHeadsAdaptersMixin
from .adapter_modeling import Activation_Function_Class, Adapter, BertFusion, GLOWCouplingBlock, NICECouplingBlock
from .adapter_utils import flatten_adapter_names, parse_adapter_names


logger = logging.getLogger(__name__)


def get_fusion_regularization_loss(model):
    if hasattr(model, "base_model"):
        model = model.base_model
    elif hasattr(model, "encoder"):
        pass
    else:
        raise Exception("Model not passed correctly, please pass a transformer model with an encoder")

    reg_loss = 0.0
    target = torch.zeros((model.config.hidden_size, model.config.hidden_size)).fill_diagonal_(1.0).to(model.device)
    for k, v in model.encoder.layer._modules.items():

        for _, layer_fusion in v.output.adapter_fusion_layer.items():
            if hasattr(layer_fusion, "value"):
                reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        for _, layer_fusion in v.attention.output.adapter_fusion_layer.items():
            if hasattr(layer_fusion, "value"):
                reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

    return reg_loss


class Head(CpgModule):

    def __init__(self, in_features, bias=True, model_config=None, max_length=None, max_label_length=None,layers=2, activation_function=None, hidden_dropout_prob=0.0):
        class c:
            if model_config.hp_dict['cpg_head_positions']:
                context_dim = in_features * 2
            else:
                context_dim = in_features

            down_dim = model_config.hp_dict['cpg_down_dim']

        config = c()
        super().__init__(config)
        self.in_features = in_features
        self.max_length = max_length
        self.max_label_length = max_label_length
        self.dropout = torch.nn.Dropout(0.3)
        self.model_config = model_config
        if self.model_config.hp_dict['add_label_noise']:
            if model_config.hp_dict['adapter_label_noise']:
                if model_config.hp_dict['position_label_noise']:
                    in_feat_noise = self.in_features*2
                else:
                    in_feat_noise = self.in_features
                self.label_embedder = Adapter(non_linearity="gelu",
                                              input_size=in_feat_noise,
                                              output_size=self.in_features,
                                              down_sample=48,)
            else:
                self.label_embedder = nn.Linear(self.in_features , self.in_features)


        def gelu_new(x):
            """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
            """
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        self.non_lin = gelu_new

        pred_linear = []

        for l in range(layers):
            pred_linear.append(nn.Dropout(hidden_dropout_prob))
            if l < layers - 1:
                # pred_linear.append(nn.Linear(self.in_features, self.in_features))
                pred_linear.append(cpg.Linear(self.in_features, self.in_features, config=c))
                pred_linear.append(activation_function)

        self.pred_linear = cpg.Sequential(*pred_linear)

        cpg_linear = []

        for l in range(layers):
            cpg_linear.append(nn.Dropout(hidden_dropout_prob))
            if l < layers - 1:
                cpg_linear.append(nn.Linear(self.in_features, self.in_features))
                cpg_linear.append(activation_function)

        self.cpg_linear = nn.Sequential(*cpg_linear)

        self.add_param('weight', (1, in_features))
        if bias:
            self.add_param('bias', (1,))
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

    def forward(self, pred_input, contextual_inputs, context_embedding=None, max_length=None, max_label_length=None,attention_mask=None):

        if (max_label_length is None and self.max_label_length is None) or (max_length is None and self.max_length is None):
            raise Exception("Max Lengths are not set for CPG Head, cannot compute parameters.")

        max_length = max_length if max_length is not None else self.max_length
        max_label_length = max_label_length if max_label_length is not None else self.max_label_length

        input_length = contextual_inputs.shape[1]

        weight = None
        bias = None

        contextual_inputs = self.non_lin(self.cpg_linear(contextual_inputs))

        for i in range(max_length, input_length, max_label_length):
            c = torch.sum(contextual_inputs[:, i + 1:i + max_label_length - 1, :] * attention_mask[:, i + 1:i + max_label_length - 1][:, :, None].repeat(1, 1, contextual_inputs.shape[-1]), 1) / torch.sum(attention_mask[:, i + 1:i + max_label_length - 1], -1)[:,None].repeat(1,contextual_inputs.shape[-1])
            # c = torch.sum(contextual_inputs[:, i + 1:i + max_label_length - 1, :] * attention_mask[:, i + 1:i + max_label_length - 1][:, :, None].repeat(1, 1, contextual_inputs.shape[-1]), 1)
            if self.model_config.hp_dict['cpg_head_positions']:
                c = torch.cat((c,contextual_inputs[:,0,:]), 1)

            if weight is None:
                # torch.sum(contextual_inputs[:, i + 1:i + max_label_length - 1, :] * attention_mask[:,
                #                                                                     i + 1:i + max_label_length - 1][:,
                #                                                                     :, None].repeat(1, 1,
                #                                                                                     contextual_inputs.shape[
                #                                                                                         -1]), 1)
                # torch.sum(attention_mask[:, i + 1:i + max_label_length - 1], -1)
                weight = self.eval_param('weight', context_embedding=c)
                # weight = self.eval_param('weight', context_embedding=contextual_inputs[:,i+1,:])
                # weight = self.eval_param('weight', context_embedding=contextual_inputs[:,i,:])
                bias = self.eval_param('bias', context_embedding=c)
                # bias = self.eval_param('bias', context_embedding=contextual_inputs[:,i+1,:])
                # bias = self.eval_param('bias', context_embedding=contextual_inputs[:,i,:])
            else:
                weight = torch.cat((weight, self.eval_param('weight', context_embedding=c)), 2)
                # weight = torch.cat((weight, self.eval_param('weight', context_embedding=contextual_inputs[:,i+1,:])), 2)
                # weight = torch.cat((weight, self.eval_param('weight', context_embedding=contextual_inputs[:,i,:])), 2)
                bias = torch.cat((bias, self.eval_param('bias', context_embedding=c)), 1)
                # bias = torch.cat((bias, self.eval_param('bias', context_embedding=contextual_inputs[:,i+1,:])), 1)
                # bias = torch.cat((bias, self.eval_param('bias', context_embedding=contextual_inputs[:,i,:])), 1)
        if self.model_config.hp_dict['cpg_head_positions']:
            c = torch.sum(contextual_inputs[:, max_length:, :] * attention_mask[:,max_length:][:, :, None].repeat(1, 1, contextual_inputs.shape[-1]), 1) / torch.sum(attention_mask[:, max_length:], -1)[:,None].repeat(1,contextual_inputs.shape[-1])
            c = torch.cat((c, contextual_inputs[:,0,:]), 1)
        else:
            c = contextual_inputs[:,0,:]
        # output = self.pred_linear[0](pred_input)
        # output = self.pred_linear[1](output,c)
        # output = self.pred_linear[2](output)
        # output = self.pred_linear[3](output)
        output = self.pred_linear(pred_input,c)

        return torch.matmul(output[:,None,:], weight).squeeze(1) +bias
        # return F.linear(output, weight, bias)


class BertSelfOutputAdaptersMixin:
    """Adds adapters to the BertSelfOutput module.
    """

    def _init_adapter_modules(self):
        self.attention_text_task_adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.attention_text_lang_adapters = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["mh_adapter"]:
            if adapter_config['cpg']:
                cpg_config = cpg.CpgModuleConfig(adapter_config['cpg'])
            else:
                cpg_config = None
            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // adapter_config["reduction_factor"],
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
                cpg_config=cpg_config
            )
            if adapter_type == AdapterType.text_task:
                self.attention_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.attention_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        if self.config.adapters.common_config_value(adapter_names, "mh_adapter"):
            self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both

        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
            for adapter_fusion_group in adapter_names:
                fusion_name = ",".join(adapter_fusion_group)
                if fusion_name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[fusion_name].parameters():
                        param.requires_grad = True

    def get_adapter_preparams(
        self, adapter_config, hidden_states, input_tensor,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        if adapter_config["original_ln_before"]:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and not self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter

        Returns: layer | None

        """
        if adapter_name in self.attention_text_lang_adapters:
            return self.attention_text_lang_adapters[adapter_name]
        if adapter_name in self.attention_text_task_adapters:
            return self.attention_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, input_tensor, adapter_stack, context_embedding=None):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters

        Returns: hidden_states

        """
        # We assume that all adapters have the same residual connection and layer norm setting as the first adapter in
        # the stack
        adapter_config = self.config.adapters.get(adapter_stack[0])

        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        if len(adapter_stack) == 1:

            adapter_layer = self.get_adapter_layer(adapter_stack[0])
            if adapter_layer is not None:
                hidden_states, _, _ = adapter_layer(
                        hidden_states, residual_input=residual, context_embedding=context_embedding)

            return hidden_states

        else:
            return self.adapter_fusion(hidden_states, adapter_stack, residual, query)

    def adapter_fusion(self, hidden_states, adapter_stack, residual, query):
        """
        If more than one adapter name is set for a stack layer, we fuse the adapters.
        For this, we pass through every adapter and learn an attention-like weighting of each adapter.
        The information stored in each of the adapters is thus fused together wrt the current example.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
            residual: residual of the previous layer
            query: query by which we attend over the adapters

        Returns: hidden_states

        """

        up_list = []

        for adapter_name in adapter_stack:
            adapter_layer = self.get_adapter_layer(adapter_name)
            if adapter_layer is not None:
                intermediate_output, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)
        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_name = ",".join(adapter_stack)

            hidden_states = self.adapter_fusion_layer[fusion_name](query, up_list, up_list, residual,)
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor, adapter_names=None,
                         cpg_environments=None, language=None):

        if adapter_names is not None:
            adapter_names = parse_adapter_names(adapter_names)
            flat_adapter_names = [item for sublist in adapter_names for item in sublist]

        if adapter_names is not None and (
            len(
                (set(self.attention_text_task_adapters.keys()) | set(self.attention_text_lang_adapters.keys()))
                & set(flat_adapter_names)
            )
            > 0
        ):
            for adapter_stack in adapter_names:
                if cpg_environments and adapter_stack[0] in cpg_environments:
                    assert language is not None
                    context = {'language': language}
                    context_embedding = cpg_environments[adapter_stack[0]](context)
                else:
                    context_embedding = None

                hidden_states = self.adapter_stack_layer(
                    hidden_states, input_tensor, adapter_stack, context_embedding=context_embedding
                )

            last_config = self.config.adapters.get(adapter_names[-1][-1])
            if last_config["original_ln_after"]:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertOutputAdaptersMixin:
    """Adds adapters to the BertOutput module.
    """

    def _init_adapter_modules(self):
        # self.bert_adapter_att = BertAdapterAttention(config)
        # self.bert_adapter_att = SimpleAdapterWeightingSentLvl(config)
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.layer_text_task_adapters = nn.ModuleDict(dict())
        self.layer_text_lang_adapters = nn.ModuleDict(dict())

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        if self.config.adapters.common_config_value(adapter_names, "output_adapter"):
            self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["output_adapter"]:
            if adapter_config.get('cpg', None):
                cpg_config = cpg.CpgModuleConfig(adapter_config['cpg'])
            else:
                cpg_config = None
            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // adapter_config["reduction_factor"],
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
                cpg_config=cpg_config
            )
            if adapter_type == AdapterType.text_task:
                self.layer_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.layer_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_fusion: bool):

        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
            for adapter_fusion_group in adapter_names:
                fusion_name = ",".join(adapter_fusion_group)
                if fusion_name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[fusion_name].parameters():
                        param.requires_grad = True

    def get_adapter_preparams(
        self, adapter_config, hidden_states, input_tensor,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        if adapter_config["original_ln_before"]:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and not self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter

        Returns: layer | None

        """
        if adapter_name in self.layer_text_lang_adapters:
            return self.layer_text_lang_adapters[adapter_name]
        if adapter_name in self.layer_text_task_adapters:
            return self.layer_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, input_tensor, adapter_stack, context_embedding=None, split_pos=None):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters

        Returns: hidden_states

        """
        # We assume that all adapters have the same residual connection and layer norm setting as the first adapter in
        # the stack

        if '|' in adapter_stack[0]:
            adapter_names = adapter_stack[0].split('|')
            adapter_config = self.config.adapters.get(adapter_names[0])

        else:
            adapter_names = None
            adapter_config = self.config.adapters.get(adapter_stack[0])

        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        if len(adapter_stack) == 1:
            if adapter_names is None:
                adapter_layer = self.get_adapter_layer(adapter_stack[0])
                adapter_layer2 = None
            else:
                adapter_layer = self.get_adapter_layer(adapter_names[0])
                adapter_layer2 = self.get_adapter_layer(adapter_names[1])

                hidden_states, hidden_states2 = hidden_states[:,:split_pos, :], hidden_states[:,split_pos:, :]
                residual, residual2 = residual[:,:split_pos, :], residual[:,split_pos:, :]


            if adapter_layer is not None:
                if hasattr(self.config, "hp_dict") and self.config.hp_dict['adapter_layer_context'] and context_embedding is not None:
                    context_embedding = torch.cat((context_embedding, self.previous_context_embedding), -1)

                hidden_states, _, _ = adapter_layer(
                        hidden_states, residual_input=residual, context_embedding=context_embedding)

            if adapter_layer2 is not None:
                hidden_states2, _, _ = adapter_layer(
                    hidden_states2, residual_input=residual2, context_embedding=context_embedding)

                hidden_states = torch.cat((hidden_states, hidden_states2), 1)

            return hidden_states

        else:
            return self.adapter_fusion(hidden_states, adapter_stack, residual, query)

    def adapter_fusion(self, hidden_states, adapter_stack, residual, query):
        """
        If more than one adapter name is set for a stack layer, we fuse the adapters.
        For this, we pass through every adapter and learn an attention-like weighting of each adapter.
        The information stored in each of the adapters is thus fused together wrt the current example.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
            residual: residual of the previous layer
            query: query by which we attend over the adapters

        Returns: hidden_states

        """
        up_list = []

        for adapter_name in adapter_stack:
            adapter_layer = self.get_adapter_layer(adapter_name)
            if adapter_layer is not None:
                intermediate_output, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)

        if len(up_list) > 0:

            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_name = ",".join(adapter_stack)

            hidden_states = self.adapter_fusion_layer[fusion_name](query, up_list, up_list, residual)
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor,
                         adapter_names=None,
                         cpg_environments=None,
                         language=None,
                         context_embedding=None,
                         split_pos=None
                         ):

        if adapter_names is not None:
            adapter_names = parse_adapter_names(adapter_names)


            # flat_adapter_names = [item for sublist in adapter_names for item in sublist]
            flat_adapter_names = [item.split('|')[0] for sublist in adapter_names for item in sublist]

        if adapter_names is not None and (
            len(
                (set(self.layer_text_lang_adapters.keys()) | set(self.layer_text_task_adapters.keys()))
                & set(flat_adapter_names)
            )
            > 0
        ):

            for adapter_stack in adapter_names:
                #logging.info('Initialising adapter_stack "%s" with cpg_environments %s' % (
                #        adapter_stack, str(cpg_environments)))
                # if cpg_environments and adapter_stack[0] in cpg_environments:
                #     assert language is not None
                #     context = {'language': language}
                #     context_embedding = cpg_environments[adapter_stack[0]](context)
                # else:
                #     context_embedding = None

                hidden_states = self.adapter_stack_layer(
                    hidden_states, input_tensor, adapter_stack, context_embedding=context_embedding,split_pos=split_pos
                )

            last_config = self.config.adapters.get(adapter_names[-1][-1].split('|')[0])
            if last_config["original_ln_after"]:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        self.previous_context_embedding = hidden_states[:,0,:]

        return hidden_states


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module.
    """

    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        self.attention.output.add_adapter(adapter_name, adapter_type)
        self.output.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention.output.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module.
    """

    def add_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        for layer in self.layer:
            layer.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class BertModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BertModel module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        self.cpg_environments = nn.ModuleDict(dict())
        self.invertible_lang_adapters = nn.ModuleDict(dict())

        # language adapters
        for language in self.config.adapters.adapter_list(AdapterType.text_lang):
            self.encoder.add_adapter(language, AdapterType.text_lang)
            self.add_invertible_lang_adapter(language)
        # task adapters
        for task in self.config.adapters.adapter_list(AdapterType.text_task):
            self.encoder.add_adapter(task, AdapterType.text_task)
        # fusion
        if hasattr(self.config, "fusion_models"):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def train_adapter(self, adapter_names: list):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.encoder.enable_adapters(adapter_names_flat, True, False)
        # unfreeze invertible adapters for invertible adapters
        for adapter_name in adapter_names_flat:
            if adapter_name in self.invertible_lang_adapters:
                for param in self.invertible_lang_adapters[adapter_name].parameters():
                    param.requires_grad = True
            if adapter_name in self.cpg_environments:
                for param in self.cpg_environments[adapter_name].parameters():
                    param.requires_grad = True
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_names)

    def train_fusion(self, adapter_names: list):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.encoder.enable_adapters(adapter_names_flat, False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_names)
        # TODO implement fusion for invertible adapters

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        if not AdapterType.has(adapter_type):
            raise ValueError("Invalid adapter type {}".format(adapter_type))
        if config is None:
            if not self.config.adapters.get_config(adapter_type):
                self.config.adapters.set_config(adapter_type, config or DEFAULT_ADAPTER_CONFIG)
            config = self.config.adapters.get_config(adapter_type)
        self.config.adapters.add(adapter_name, adapter_type, config=config)
        if adapter_type == AdapterType.text_lang and config.get('cpg', None):
            self.add_cpg_environment(adapter_name, config['cpg'])
        self.encoder.add_adapter(adapter_name, adapter_type)
        if adapter_type == AdapterType.text_lang and config['invertible_adapter']:
            self.add_invertible_lang_adapter(adapter_name)

    def add_cpg_environment(self, adapter_name, cpg_config):
        properties = []
        if cpg_config.get('use_typology', False):
            language_property = cpg.UrielMlpProperty(
                    'language', cpg_config['language_embedding_dim'], cpg_config['languages'])
        else:
            language_property = cpg.Property(
                    'language', cpg_config['language_embedding_dim'], cpg_config['languages'])
        properties.append(language_property)
        #if cpg_config['layer_embedding_dim']:
        #    layers = ['layer_%d' % layer for layer in range(self.config.num_hidden_layers)]
        #    properties.append(cpg.Property(
        #            'layer', cpg_config['layer_embedding_dim'], layers))
        environment = cpg.Environment(properties)
        self.cpg_environments[adapter_name] = environment
        return environment

    def add_invertible_lang_adapter(self, language):
        if language in self.invertible_lang_adapters:
            raise ValueError(f"Model already contains an adapter module for '{language}'.")
        inv_adap_config = self.config.adapters.get(language)["invertible_adapter"]
        if inv_adap_config["block_type"] == "nice":
            cpg_config = self.config.adapters.get(language).get('cpg', None)
            if cpg_config:
                cpg_config = cpg.CpgModuleConfig(cpg_config, include_layer=False)
            inv_adap = NICECouplingBlock(
                [[self.config.hidden_size]],
                non_linearity=inv_adap_config["non_linearity"],
                reduction_factor=inv_adap_config["reduction_factor"],
                cpg_config=cpg_config
            )
        elif inv_adap_config["block_type"] == "glow":
            inv_adap = GLOWCouplingBlock(
                [[self.config.hidden_size]],
                non_linearity=inv_adap_config["non_linearity"],
                reduction_factor=inv_adap_config["reduction_factor"],
            )
        else:
            raise ValueError(f"Invalid invertible adapter type '{inv_adap_config['block_type']}'.")
        self.invertible_lang_adapters[language] = inv_adap
        self.invertible_lang_adapters[language].apply(Adapter.init_bert_weights)

    def get_invertible_lang_adapter(self, language):
        if language in self.invertible_lang_adapters:
            return self.invertible_lang_adapters[language]
        else:
            return None

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        self.encoder.add_fusion_layer(adapter_names)


class BertModelHeadsMixin(ModelWithHeadsAdaptersMixin):
    """Adds heads to a Bert-based module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_head = None

    def _init_head_modules(self):
        if not hasattr(self.config, "prediction_heads"):
            self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        for head_name in self.config.prediction_heads:
            self._add_prediction_head_module(head_name)

    def set_active_adapters(self, adapter_names: list):
        """Sets the adapter modules to be used by default in every forward pass.
        This setting can be overriden by passing the `adapter_names` parameter in the `foward()` pass.
        If no adapter with the given name is found, no module of the respective type will be activated.
        In case the calling model class supports named prediction heads, this method will attempt to activate a prediction head with the name of the last adapter in the list of passed adapter names.

        Args:
            adapter_names (list): The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        self.base_model.set_active_adapters(adapter_names)
        # use last adapter name as name of prediction head
        if self.active_adapters:
            head_name = self.active_adapters[-1][-1]
            if head_name in self.config.prediction_heads:
                self.active_head = head_name
            else:
                logger.info("No prediction head for task_name '{}' available.".format(head_name))

    def add_classification_head(
        self, head_name, num_labels=2, layers=2, activation_function="tanh", overwrite_ok=False, multilabel=False
    ):
        """Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """
        if multilabel:
            head_type = "multilabel_classification"
        else:
            head_type = "classification"

        config = {
            "head_type": head_type,
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_cpg_head(self,
                     head_name, layers=2, activation_function="tanh"
                     , bias=True, max_length=None, max_label_length=None, overwrite_ok=False,
                     ):
        config = {
            "head_type": "cpg",
            "num_choices": None,
            "layers": layers,
            "activation_function": activation_function,
        }
        self.add_prediction_head(head_name,
                                 config,
                                 overwrite_ok,
                                 cpg=True,
                                 bias=bias,
                                 max_label_length=max_label_length,
                                 max_length=max_length)

    def add_multiple_choice_head(
        self, head_name, num_choices=2, layers=2, activation_function="tanh", overwrite_ok=False,
    ):
        """Adds a multiple choice head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_choices (int, optional): Number of choices. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        config = {
            "head_type": "multiple_choice",
            "num_choices": num_choices,
            "layers": layers,
            "activation_function": activation_function,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_tagging_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False,
    ):
        """Adds a token classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        config = {
            "head_type": "tagging",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False,
    ):
        config = {
            "head_type": "question_answering",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_prediction_head(
            self,
            head_name,
            config,
            overwrite_ok=False,
            cpg=False,
            bias=None,
            max_label_length=None,
            max_length=None,
    ):
        if head_name not in self.config.prediction_heads or overwrite_ok:
            self.config.prediction_heads[head_name] = config

            logger.info(f"Adding head '{head_name}' with config {config}.")
            if 'cpg' in head_name:
                cpg=True
                bias=True
            self._add_prediction_head_module(head_name,
                                             cpg=cpg,
                                             bias=bias,
                                             max_length=max_length,
                                             max_label_length=max_label_length)
            self.active_head = head_name

        else:
            raise ValueError(
                f"Model already contains a head with name '{head_name}'. Use overwrite_ok=True to force overwrite."
            )

    def _add_prediction_head_module(self,
                                    head_name,
                                    cpg=False,
                                    bias=None,
                                    max_label_length=None,
                                    max_length=None,
                                    ):
        head_config = self.config.prediction_heads.get(head_name)

        pred_head = []
        if not cpg:
            for l in range(head_config["layers"]):
                pred_head.append(nn.Dropout(self.config.hidden_dropout_prob))
                if l < head_config["layers"] - 1:
                    pred_head.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
                    pred_head.append(Activation_Function_Class(head_config["activation_function"]))
                else:
                    if "num_labels" in head_config:
                        pred_head.append(nn.Linear(self.config.hidden_size, head_config["num_labels"]))
                    else:  # used for multiple_choice head
                        pred_head.append(nn.Linear(self.config.hidden_size, 1))

        else:
            pred_head.append(Head(self.config.hidden_size,
                                     bias=bias,
                                     model_config=self.config,
                                     max_length=max_length,
                                     max_label_length=max_label_length,
                                     layers=head_config["layers"],
                                     activation_function=Activation_Function_Class(head_config["activation_function"]),
                                     hidden_dropout_prob=self.config.hidden_dropout_prob
                                     ) )

        self.heads[head_name] = nn.Sequential(*pred_head)

        self.heads[head_name].apply(self._init_weights)
        self.heads[head_name].train(self.training)  # make sure training mode is consistent

    def forward_head(self, outputs,
                     head_name=None,
                     attention_mask=None,
                     labels=None,
                     contextual_inputs=None,
                     attention_mask_=None,
                     max_label_length=None,
                     max_length=None
                     ):
        head_name = head_name or self.active_head
        if not head_name:
            logger.debug("No prediction head is used.")
            return outputs

        if head_name not in self.config.prediction_heads:
            raise ValueError("Unknown head_name '{}'".format(head_name))

        head = self.config.prediction_heads[head_name]

        sequence_output = outputs[0]

        if head["head_type"] == "classification" :
            logits = self.heads[head_name](sequence_output[:, 0])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if head["num_labels"] == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, head["num_labels"]), labels.view(-1))
                outputs = (loss,) + outputs

        elif head["head_type"] == "cpg":
            logits = self.heads[head_name][0](pred_input=sequence_output[:, 0],
                                              contextual_inputs=contextual_inputs,
                                              attention_mask=attention_mask_,
                                              max_label_length=max_label_length,
                                              max_length=max_length)

            outputs = (logits,) + outputs[2:]
            if labels is not None and not isinstance(labels[0], list):
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
                outputs = (loss,) + outputs

        elif head["head_type"] == "multilabel_classification":
            logits = self.heads[head_name](sequence_output[:, 0])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                if labels.dtype != torch.float32:
                    labels = labels.float()
                loss = loss_fct(logits, labels)
                outputs = (loss,) + outputs

        elif head["head_type"] == "multiple_choice":
            logits = self.heads[head_name](sequence_output[:, 0])
            logits = logits.view(-1, head["num_choices"])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                outputs = (loss,) + outputs

        elif head["head_type"] == "tagging":
            logits = self.heads[head_name](sequence_output)

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        elif head["head_type"] == "question_answering":
            logits = self.heads[head_name](sequence_output)

            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]
            if labels is not None:
                start_positions, end_positions = labels
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

        else:
            raise ValueError("Unknown head_type '{}'".format(head["head_type"]))

        return outputs  # (loss), logits, (hidden_states), (attentions)
