from __future__ import annotations
from functools import partial
from collections import namedtuple
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, stack
from torch.nn import Module, Parameter, Linear
from torch.func import grad, vmap, functional_call

from einx import multiply
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange, Reduce

from torch_einops_utils import pack_with_inverse, tree_map_detach

# constants

Params = dict[str, Tensor]

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(t):
    return t is not None

def default(t, d):
    return t if exists(t) else d

def divisible_by(num, den):
    return (num % den) == 0

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
        muon_param_names: Sequence[str] | None = None,
        learned_forget = False
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

        # forget gates

        self.learned_forget = learned_forget

        if learned_forget:
            self.to_forget_gate = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.RMSNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, 1),
                nn.Sigmoid(),
                Rearrange('b 1 -> b')
            )

    def forward(
        self,
        tokens: Tensor,
        prev_fast_weights: Params | None = None
    ) -> tuple[Tensor, Params]:

        batch, muon_update, should_forget = tokens.shape[0], self.muon_update, self.learned_forget

        # get the queries for retrieving, and keys and values for storing

        qkv = self.to_qkv(tokens)

        q, k, v = self.split_qkv(qkv)

        # constitute the params from accumulated fast weights and base params

        base_memory_params = {param_name: repeat(param, '... -> b ...', b = batch) for param_name, param in self.base_memory_params.items()}

        memory_params = add_dict(prev_fast_weights, base_memory_params)

        # get the next generated fast weights from the surprise

        delta_fast_weights = self.store(memory_params, (k, v))

        # updates, standard gd or muon update, and optional learned forget gates (many papers show forgetting / wd is important)

        lr = F.softplus(self.learnable_lr)

        if muon_update:
            muon_lr = F.softplus(self.muon_learnable_lr)

        if should_forget:
            forget_gate = self.to_forget_gate(tokens)

        updated_delta_fast_weights = {}

        for name, delta_weights in delta_fast_weights.items():

            if should_forget:
                delta_weights = multiply('b ..., b -> b ...', delta_weights, forget_gate)

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

TTTWrapperIntermediates = namedtuple('TTTWrapperIntermediates', ('memory_out', 'next_fast_weights', 'delta_fast_weights'))

class TTTWrapper(Module):

    def __init__(
        self,
        dim,
        *,
        memory: Module, # the memory module type
        tbptt_step_size: int | None = None
    ):
        super().__init__()
        self.memory = memory
        self.tbptt_step_size = tbptt_step_size
        self.memory_out_layerscale = nn.Parameter(torch.randn(dim) * 1e-4)

    def forward(
        self,
        action_chunks, # (b t n d) | (b n d) - let it accept one action chunk, or multiple action chunks
        prev_fast_weights: Params | None = None,
        tbptt_step_size: int | None = None
    ):
        tbptt_step_size = default(tbptt_step_size, self.tbptt_step_size)

        action_chunks, maybe_unsqueeze_time = pack_with_inverse(action_chunks, 'b * n d')

        # accumulate

        all_outputs = []

        for step, action_chunk in enumerate(rearrange(action_chunks, 'b t n d -> t b n d'), 1):

            # memory out

            memory_out, delta_fast_weights = self.memory(action_chunk, prev_fast_weights)

            # they propose to tanh gate the ttt recurrent memory output, small initted

            chunk_out = action_chunk + memory_out * self.memory_out_layerscale.tanh()

            all_outputs.append(chunk_out)

            # add the fast weights

            next_fast_weights = add_dict(prev_fast_weights, delta_fast_weights)

            # set for next chunk

            prev_fast_weights = next_fast_weights

            # truncated backprop through time (tbptt)

            if exists(tbptt_step_size) and divisible_by(step, tbptt_step_size):
                prev_fast_weights = tree_map_detach(prev_fast_weights)

        # intermediates

        intermediates = TTTWrapperIntermediates(memory_out, next_fast_weights, delta_fast_weights)

        out = rearrange(all_outputs, 't b n d -> b t n d')

        # maybe remove time

        out = maybe_unsqueeze_time(out)

        return out, next_fast_weights, intermediates

# classes

class RoboTTT(Module):
    def __init__(
        self,
        vla: Module,
        *,
        ttt_wrapper: TTTWrapper,
    ):
        super().__init__()
        self.vla = vla
        self.ttt_wrapper = ttt_wrapper
