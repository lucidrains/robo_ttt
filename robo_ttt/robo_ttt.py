from __future__ import annotations
from functools import partial
from collections import namedtuple
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor
from torch.nn import Module, Parameter, Linear
from torch.func import grad, vmap, functional_call

import einx
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange

from torch_einops_utils import pack_with_inverse, tree_flatten_with_inverse, tree_map_detach

# constants

Params = dict[str, Tensor]

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(t):
    return t is not None

def default(t, d):
    return t if exists(t) else d

def add_dict(x, y):
    if not exists(x):
        return y

    return {k: x[k] + y[k] for k in x.keys()}

def transpose(t):
    return t.transpose(-1, -2)

# muon updates

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    assert t.ndim > 2

    t, inv_pack = pack_with_inverse(t, '* i j')

    m, n = t.shape[-2:]
    should_transpose = m > n

    if should_transpose:
        t = transpose(t)

    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ transpose(t)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = transpose(t)

    return inv_pack(t)

# ttt memory

class MemoryKeyValueBind(Module):
    def __init__(
        self,
        dim,
        memory_net: Module,
        muon_update = False,
        muon_param_names: Sequence[str] | None = None
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

        self.learnable_lr = Parameter(tensor(1e-2))

        # maybe muon update

        self.muon_update = muon_update
        self.muon_param_names = set()

        if muon_update:
            self.muon_learnable_lr = Parameter(tensor(1e-1))

            self.muon_param_names = set(
                param_name for param_name, param in self.base_memory_params.items()
                if param.ndim >= 2 and (not exists(muon_param_names) or param_name in muon_param_names)
            )

    def forward(
        self,
        tokens: Tensor,
        prev_fast_weights: Params | None = None
    ) -> tuple[Tensor, Params]:

        batch, muon_update = tokens.shape[0], self.muon_update

        # get the queries for retrieving, and keys and values for storing

        qkv = self.to_qkv(tokens)

        q, k, v = self.split_qkv(qkv)

        # constitute the params from accumulated fast weights and base params

        base_memory_params = {param_name: repeat(param, '... -> b ...', b = batch) for param_name, param in self.base_memory_params.items()}

        memory_params = add_dict(prev_fast_weights, base_memory_params)

        # get the next generated fast weights from the surprise

        delta_fast_weights = self.store(memory_params, (k, v))

        # updates, standard gd or muon update

        lr = F.softplus(self.learnable_lr)

        if muon_update:
            muon_lr = F.softplus(self.muon_learnable_lr)

        updated_delta_fast_weights = {}

        for name, delta_weights in delta_fast_weights.items():

            if muon_update and name in self.muon_param_names:
                delta_weights = newtonschulz5(delta_weights) * muon_lr
            else:
                delta_weights = delta_weights * lr

            updated_delta_fast_weights[name] = delta_weights

        delta_fast_weights = updated_delta_fast_weights

        # retrieve with queries - section 2 eq (1) & (2) update then apply

        next_memory_params = add_dict(delta_fast_weights, memory_params)

        retrieved = self.retrieve(next_memory_params, q)

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

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_context = default(dim_context, dim)
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens,
        context = None,
        mask = None
    ):
        tokens = self.norm(tokens)

        context = default(context, tokens)

        q = self.to_q(tokens)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(self.split_heads, (q, *kv))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')
        sim = sim * self.scale

        attn = sim.softmax(dim = -1)

        agg = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(agg)
        return self.to_out(out)

# classes

class RoboTTT(Module):
    def __init__(
        self
    ):
        super().__init__()
