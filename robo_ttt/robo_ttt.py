from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, Tensor, tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.func import grad, vmap, functional_call

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch_einops_utils import pack_with_inverse, tree_flatten_with_inverse, tree_map_detach

# constants

Params = dict[str, Tensor]

# helpers

def exists(t):
    return t is not None

def default(t, d):
    return t if exists(t) else d

def add_dict(x, y):
    if not exists(x):
        return y

    return {k: x[k] + y[k] for k in x.keys()}

# ttt memory

class MemoryKeyValueBind(Module):
    def __init__(
        self,
        dim,
        memory_net: Module
    ):
        super().__init__()

        # query key values

        self.to_qkv = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim * 3)
        )

        self.split_qkv = Rearrange('b n (qkv d) -> qkv b n d', qkv = 3)

        # the memory

        self.memory = memory_net
        self.base_memory_params = dict(memory_net.named_parameters())

        def _retrieve(params, inputs):
            return functional_call(memory_net, params, (inputs,))

        self.retrieve = vmap(_retrieve, in_dims = (0, 0))

        def _store(params, inputs):
            keys, values = inputs
            retrieved = _retrieve(params, keys)
            return -F.mse_loss(retrieved, values)

        self.store = vmap(grad(_store, argnums = 0), in_dims = (0, 0))

        # learning rate

        self.learnable_lr = Parameter(tensor(1e-1))

    def forward(
        self,
        tokens: Tensor,
        prev_fast_weights: Params | None = None
    ) -> tuple[Tensor, Params]:
        batch = tokens.shape[0]

        # get the queries for retrieving, and keys and values for storing

        qkv = self.to_qkv(tokens)

        q, k, v = self.split_qkv(qkv)

        # constitute the params from accumulated fast weights and base params

        base_memory_params = {param_name: repeat(param, '... -> b ...', b = batch) for param_name, param in self.base_memory_params.items()}

        memory_params = add_dict(prev_fast_weights, base_memory_params)

        # get the next generated fast weights from the surprise

        delta_fast_weights = self.store(memory_params, (k, v))

        delta_fast_weights = {name: delta * self.learnable_lr for name, delta in delta_fast_weights.items()}

        # retrieve with queries

        retrieved = self.retrieve(memory_params, q)

        return retrieved, delta_fast_weights

# ttt wrapper

TTTWrapperIntermediates = namedtuple('TTTWrapperIntermediates', ('block_out', 'memory_out', 'next_fast_weights', 'delta_fast_weights'))

class TTTWrapper(Module):

    def __init__(
        self,
        dim,
        *,
        memory: Module,  # they use key/value binding method, but make it customizable
        block: Module,   # in paper, they wrap attention blocks
    ):
        super().__init__()
        self.block = block

        self.memory = memory
        self.memory_out_layerscale = nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(
        self,
        tokens,
        *args,
        prev_fast_weights: Params | None = None,
        detach_prev_fast_weights = False,
        detach_next_fast_weights = False,
        **kwargs
    ):

        # normal block out

        block_out = self.block(tokens, *args, **kwargs)

        # memory out

        # maybe detach prev memories, give some flexibility to the researcher, can detach prev or next

        if detach_prev_fast_weights:
            prev_fast_weights = tree_map_detach(prev_fast_weights)

        memory_out, delta_fast_weights = self.memory(tokens, prev_fast_weights)

        # block may return tuple

        (block_out, *rest), unflatten_tree = tree_flatten_with_inverse(block_out)

        # they propose to tanh gate the ttt recurrent memory output, small initted

        out = block_out + memory_out * self.memory_out_layerscale.tanh()

        # add the fast weights

        next_fast_weights = add_dict(prev_fast_weights, delta_fast_weights)

        # detaching, give some flexibility to the researcher

        if detach_next_fast_weights:
            next_fast_weights = tree_map_detach(next_fast_weights)

        # intermediates

        intermediates = TTTWrapperIntermediates(block_out, memory_out, next_fast_weights, delta_fast_weights)

        # bring back the caching from attention block

        out = unflatten_tree((out, *rest))

        return out, next_fast_weights, intermediates

# classes

class RoboTTT(Module):
    def __init__(
        self
    ):
        super().__init__()
