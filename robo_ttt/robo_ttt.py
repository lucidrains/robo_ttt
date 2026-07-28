from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn
from torch.nn import Module

import einx
from einops import rearrange
from einops.layers.torch import Rearrange

from torch_einops_utils import pack_with_inverse, tree_flatten_with_inverse, tree_map_detach

# helpers

def exists(t):
    return t is not None

def default(t, d):
    return t if exists(t) else d

def add_dict(x, y):
    if not exists(x):
        return y

    return {k: x[k] + y[k] for k in x.keys()}

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
        prev_fast_weights = None,
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
