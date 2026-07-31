from __future__ import annotations

from copy import deepcopy
from functools import partial
from collections import namedtuple
from collections.abc import Sequence, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor
from torch.distributions import Beta
from torch.nn import Module, ModuleList, Parameter
from torch.func import grad, vmap, functional_call

from einx import multiply
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from torch_einops_utils import pack_with_inverse, tree_map_detach, tree_map_tensor, tree_flatten_with_inverse, masked_mean
from torch_einops_utils.device import module_device

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# constants

Params = dict[str, Tensor]
ArgKey = int | str
ArgKeys = Sequence[ArgKey]
ModulePaths = Sequence[str] | str

# memory namedtuple holding fast weights parameters and total sequence step position

Memory = namedtuple('Memory', ('fast_weights', 'step'))

def normalize_memory(memory):
    if not exists(memory):
        return Memory(None, 0)
    if isinstance(memory, Memory):
        return memory
    return Memory(memory, 0)

# helpers

def exists(t):
    return t is not None

def default(t, d):
    return t if exists(t) else d

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(val, length = 1):
    if isinstance(val, tuple):
        return val
    if isinstance(val, list):
        return tuple(val)
    return (val,) * length

# args utils

def get_arg(args: tuple, kwargs: dict, arg_key: ArgKey):
    return args[arg_key] if isinstance(arg_key, int) else kwargs[arg_key]

def set_arg(args: tuple, kwargs: dict, arg_key: ArgKey, val):
    if isinstance(arg_key, int):
        return (*args[:arg_key], val, *args[arg_key + 1:]), kwargs
    return args, {**kwargs, arg_key: val}

def has_arg(args: tuple, kwargs: dict, arg_key: ArgKey):
    if not exists(arg_key):
        return False

    if isinstance(arg_key, int):
        return arg_key < len(args)

    return arg_key in kwargs

# tensor helpers

def add_dict(x, y):
    if not exists(x):
        return y
    if not exists(y):
        return x

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
        learned_forget = True,
        rotary_embed_qk = True,
        rope_kwargs: dict = dict(theta = 10000),
        select_tokens_slice = None # optional custom slice of tokens within chunk to bind to memory (e.g. slice(1, None) to omit joint state or register tokens)
    ):
        super().__init__()

        self.select_tokens_slice = select_tokens_slice

        # query key values

        self.to_qkv = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim * 3)
        )

        self.split_qkv = Rearrange('b ... (qkv d) -> qkv b ... d', qkv = 3)

        # position aware with rope

        self.rotary_embed_qk = rotary_embed_qk

        if rotary_embed_qk:
            self.rotary_emb = RotaryEmbedding(dim = dim, **rope_kwargs)

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
        tokens,
        prev_fast_weights = None
    ):

        prev_memory = normalize_memory(prev_fast_weights)
        fast_weights, step = prev_memory.fast_weights, prev_memory.step

        batch, muon_update, should_forget = tokens.shape[0], self.muon_update, self.learned_forget

        # get the queries for retrieving, and keys and values for storing

        qkv = self.to_qkv(tokens)

        q, k, v = self.split_qkv(qkv)

        # custom token sequence slicing if specified

        if exists(self.select_tokens_slice):
            q, k, v = map(lambda t: t[..., self.select_tokens_slice, :], (q, k, v))

        # position aware rotary embedding for queries and keys (Appendix A.1)

        if self.rotary_embed_qk:
            seq_dim = -2
            seq_len = 1

            if q.ndim == 4:
                seq_dim = 1
                seq_len = q.shape[1]

            seq_pos = self.rotary_emb.get_seq_pos(seq_len, offset = step, device = q.device)
            freqs = self.rotary_emb(seq_pos)

            if seq_dim == 1:
                freqs = rearrange(freqs, 't d -> t 1 d')

            q = apply_rotary_emb(freqs, q, seq_dim = seq_dim)
            k = apply_rotary_emb(freqs, k, seq_dim = seq_dim)

        # constitute the params from accumulated fast weights and base params

        base_memory_params = {param_name: repeat(param, '... -> b ...', b = batch) for param_name, param in self.base_memory_params.items()}

        memory_params = add_dict(fast_weights, base_memory_params)

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

        if exists(self.select_tokens_slice):
            full_retrieved = tokens.new_zeros(tokens.shape)
            full_retrieved[..., self.select_tokens_slice, :] = retrieved
            retrieved = full_retrieved

        return retrieved, delta_fast_weights

# fwpkm memory wrapper - Llion Jones paper at Sakana AI (https://arxiv.org/abs/2601.00671)

class fwPKMWrapper(Module):
    def __init__(
        self,
        fw_pkm: Module
    ):
        super().__init__()
        self.fw_pkm = fw_pkm

    def forward(
        self,
        tokens,
        prev_fast_weights = None
    ):
        from fast_weight_product_key_memory.fwPKM import Memories

        past_memories = None

        if exists(prev_fast_weights):
            fw = normalize_memory(prev_fast_weights).fast_weights
            if exists(fw):
                past_memories = Memories(fw['memory_values'], fw['keys']) if isinstance(fw, dict) else fw

        out, next_memories = self.fw_pkm(
            tokens,
            past_memories = past_memories,
            return_next_memories = True
        )

        mv, keys = next_memories.memory_values, next_memories.keys

        if exists(past_memories):
            mv = mv - past_memories.memory_values
            keys = keys - past_memories.keys

        delta_fast_weights = dict(memory_values = mv, keys = keys)

        return out, delta_fast_weights

# ttt wrapper

TTTWrapperIntermediates = namedtuple('TTTWrapperIntermediates', ('memory_out', 'next_fast_weights', 'delta_fast_weights'))

class TTTWrapper(Module):

    def __init__(
        self,
        dim,
        *,
        memory: Module, # the memory module type
        select_tokens_slice = None,
        tbptt_step_size = None
    ):
        super().__init__()
        self.memory = memory
        self.select_tokens_slice = select_tokens_slice
        self.tbptt_step_size = tbptt_step_size
        self.memory_out_layerscale = nn.Parameter(torch.randn(dim) * 1e-4)

    def forward(
        self,
        action_chunks, # (b t n d) | (b n d) - let it accept one action chunk, or multiple action chunks
        prev_fast_weights = None,
        tbptt_step_size = None
    ):
        tbptt_step_size = default(tbptt_step_size, self.tbptt_step_size)

        action_chunks, maybe_unsqueeze_time = pack_with_inverse(action_chunks, 'b * n d')

        curr_memory = normalize_memory(prev_fast_weights)

        # accumulate

        all_outputs = []

        for step, action_chunk in enumerate(rearrange(action_chunks, 'b t n d -> t b n d'), 1):

            # memory out

            input_chunk = action_chunk

            if exists(self.select_tokens_slice):
                input_chunk = action_chunk[..., self.select_tokens_slice, :]

            memory_out, delta_fast_weights = self.memory(input_chunk, curr_memory)

            if exists(self.select_tokens_slice):
                full_memory_out = action_chunk.new_zeros(action_chunk.shape)
                full_memory_out[..., self.select_tokens_slice, :] = memory_out
                memory_out = full_memory_out

            # they propose to tanh gate the ttt recurrent memory output, small initted

            chunk_out = action_chunk + memory_out * self.memory_out_layerscale.tanh()

            all_outputs.append(chunk_out)

            # add the fast weights and advance sequence step

            next_fast_weights = add_dict(curr_memory.fast_weights, delta_fast_weights)
            next_step = curr_memory.step + 1

            curr_memory = Memory(next_fast_weights, next_step)

            # truncated backprop through time (tbptt)

            if exists(tbptt_step_size) and divisible_by(step, tbptt_step_size):
                curr_memory = Memory(tree_map_detach(curr_memory.fast_weights), curr_memory.step)

        next_fast_weights = curr_memory

        intermediates = TTTWrapperIntermediates(memory_out, next_fast_weights, delta_fast_weights)

        out = rearrange(all_outputs, 't b n d -> b t n d')

        # maybe remove time

        out = maybe_unsqueeze_time(out)

        return out, next_fast_weights, intermediates

# sample action times using beta distribution following Kevin Black et al. (https://arxiv.org/abs/2410.24164)
# 0 noise -> 1 data

def sample_action_times(
    batch_times: int,
    *,
    s = 0.999,
    beta_a = 1.5,
    beta_b = 1.0
):
    u = Beta(beta_a, beta_b).sample((batch_times,))
    return s * (1. - u)

# main robo-ttt class

class RoboTTT(Module):
    def __init__(
        self,
        model: Module,
        *,
        ttt_wrapper: TTTWrapper | Module,
        ttt_module_paths: ModulePaths = (),
        batch_time_arg = 0,
        expand_time_args: ArgKeys = (),
        omit_time_args: ArgKeys = (),
        times_arg: ArgKey | None = 'time',
        unreduced_loss_arg: ArgKey | None = 'return_unreduced_loss',
        loss_arg: ArgKey = 0,
        sample_action_times_fn: Callable | None = None
    ):
        super().__init__()
        self.model = model
        self.ttt_wrapper = ttt_wrapper

        # arg keys

        self.batch_time_arg = batch_time_arg
        self.expand_time_args = set(expand_time_args)
        self.omit_time_args = set(omit_time_args)
        self.times_arg = times_arg
        self.unreduced_loss_arg = unreduced_loss_arg
        self.loss_arg = loss_arg
        self.sample_action_times_fn = sample_action_times_fn

        # ttt wrappers for designated submodules
        # ttt_module_paths can contain string paths or (memory_index, path) tuples to route specific wrappers to layers

        wrappers = cast_tuple(ttt_wrapper)
        raw_module_paths = cast_tuple(ttt_module_paths)

        resolved_wrappers = []
        resolved_paths = []

        for item in raw_module_paths:
            if not isinstance(item, tuple):
                item = (0 if len(wrappers) == 1 else len(resolved_paths), item)

            wrapper_idx, path = item

            resolved_wrappers.append(deepcopy(wrappers[wrapper_idx]))
            resolved_paths.append(path)

        self.ttt_wrappers = ModuleList(resolved_wrappers)
        self.ttt_module_paths = tuple(resolved_paths)
        self.submodules = [model.get_submodule(path) for path in self.ttt_module_paths]

        # default prev fast weights

        self.default_prev_fast_weights = (None,) * len(self.ttt_wrappers)

    @property
    def device(self):
        return module_device(self)

    def finetune_parameters(self):
        return self.ttt_wrappers.parameters()

    def _normalize_prev_fast_weights(self, prev_fast_weights):
        prev_fast_weights = cast_tuple(default(prev_fast_weights, self.default_prev_fast_weights))
        assert len(prev_fast_weights) == len(self.ttt_wrappers)
        return tuple(normalize_memory(fw) for fw in prev_fast_weights)

    def _forward_with_ttt(self, fn: Callable, args: tuple, kwargs: dict, time: int, prev_fast_weights):
        handles = []
        layer_fast_weights = list(prev_fast_weights)

        # helpers for packing and unpacking time into batch dimension

        pack_batch_time = partial(rearrange, pattern = 'b t ... -> (b t) ...')
        unpack_batch_time = partial(rearrange, pattern = '(b t) ... -> b t ...', t = time)

        # register forward hooks

        for idx, (submodule, ttt_wrapper) in enumerate(zip(self.submodules, self.ttt_wrappers)):
            def hook(module, input, output, ttt_wrapper = ttt_wrapper, idx = idx):
                assert output.ndim == 3, f'module output must be 3d tensor of shape (batch * time, action_chunk_len, dim), but got shape {output.shape} for module "{module.__class__.__name__}"'

                prev_fast_weight = layer_fast_weights[idx]

                # ttt memory forward

                action_chunks = unpack_batch_time(output)
                ttt_memory_out, next_fast_weight, _ = ttt_wrapper(action_chunks, prev_fast_weights = prev_fast_weight)
                layer_fast_weights[idx] = next_fast_weight

                # add to output

                return output + pack_batch_time(ttt_memory_out)

            handles.append(submodule.register_forward_hook(hook))

        # forward model

        out = fn(*args, **kwargs)

        # unregister hooks

        for handle in handles:
            handle.remove()

        return out, tuple(layer_fast_weights)

    def _get_batch_time(
        self,
        args: tuple,
        kwargs: dict
    ):
        target = get_arg(args, kwargs, self.batch_time_arg)
        assert is_tensor(target) and target.ndim >= 2, f'batch_time_arg target must be a tensor of at least 2 dimensions'
        batch, time = target.shape[:2]

        def pack_batch_time(t):
            if not is_tensor(t) or t.ndim < 2:
                return t
            return rearrange(t, 'b t ... -> (b t) ...')

        def unpack_batch_time(t):
            if not is_tensor(t) or t.ndim == 0:
                return t

            batch_times, *_ = t.shape

            if batch_times != batch * time:
                return t

            return rearrange(t, '(b t) ... -> b t ...', t = time)

        return batch, time, pack_batch_time, unpack_batch_time

    def _forward_with_time_packing(
        self,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        *,
        prev_fast_weights = None,
        auto_unsqueeze_time: bool = False,
        before_forward_fn: Callable | None = None
    ):
        # auto unsqueeze time dimension 1 for single-timestep sampling inputs if flag is set

        if auto_unsqueeze_time:
            non_temporal_keys = self.omit_time_args | self.expand_time_args
            unsqueeze_time = lambda t: rearrange(t, 'b ... -> b 1 ...')

            args = tuple(arg if idx in non_temporal_keys else tree_map_tensor(unsqueeze_time, arg) for idx, arg in enumerate(args))
            kwargs = {k: (v if k in non_temporal_keys else tree_map_tensor(unsqueeze_time, v)) for k, v in kwargs.items()}

        # normalize fast weights

        prev_fast_weights = self._normalize_prev_fast_weights(prev_fast_weights)

        # batch time

        batch, time, pack_batch_time, unpack_batch_time = self._get_batch_time(args, kwargs)

        # expand non-temporal args across time if needed

        for arg_key in self.expand_time_args:
            val = tree_map_tensor(lambda t: repeat(t, 'b ... -> (b t) ...', t = time), get_arg(args, kwargs, arg_key))
            args, kwargs = set_arg(args, kwargs, arg_key, val)

        # omit non-temporal args from time-packing

        omitted = {arg_key: get_arg(args, kwargs, arg_key) for arg_key in self.omit_time_args if has_arg(args, kwargs, arg_key)}

        # pack time into batch dimension

        packed_args, packed_kwargs = tree_map_tensor(pack_batch_time, (args, kwargs))

        for arg_key, val in omitted.items():
            packed_args, packed_kwargs = set_arg(packed_args, packed_kwargs, arg_key, val)

        if exists(before_forward_fn):
            packed_args, packed_kwargs = before_forward_fn(batch, time, args, kwargs, packed_args, packed_kwargs)

        # forward model with ttt hooks

        out, next_fast_weights = self._forward_with_ttt(fn, packed_args, packed_kwargs, time, prev_fast_weights)

        # unpack time

        out = tree_map_tensor(unpack_batch_time, out)

        if auto_unsqueeze_time:
            out = tree_map_tensor(lambda t: rearrange(t, 'b 1 ... -> b ...'), out)

        return out, next_fast_weights

    def sample(
        self,
        *args,
        sample_fn_name = None,
        prev_fast_weights = None,
        return_fast_weights = None,
        auto_unsqueeze_time = True,
        **kwargs
    ):
        return_fast_weights = default(return_fast_weights, exists(prev_fast_weights))

        # resolve sample fn

        sample_fn_name = default(sample_fn_name, 'sample' if hasattr(self.model, 'sample') else ('sample_actions' if hasattr(self.model, 'sample_actions') else '__call__'))
        sample_fn = getattr(self.model, sample_fn_name) if sample_fn_name != '__call__' else self.model

        out, next_fast_weights = self._forward_with_time_packing(
            sample_fn,
            args,
            kwargs,
            prev_fast_weights = prev_fast_weights,
            auto_unsqueeze_time = auto_unsqueeze_time
        )

        if not return_fast_weights:
            return out

        return out, next_fast_weights

    def forward(
        self,
        *args,
        loss_mask = None,
        prev_fast_weights = None,
        return_fast_weights = None,
        **kwargs
    ):
        return_fast_weights = default(return_fast_weights, exists(prev_fast_weights))

        def before_forward(batch, time, args, kwargs, packed_args, packed_kwargs):
            # sample action times if needed

            if exists(self.times_arg) and not has_arg(args, kwargs, self.times_arg) and not has_arg(packed_args, packed_kwargs, self.times_arg):
                sample_action_times_fn = default(self.sample_action_times_fn, sample_action_times)

                # they give each batch time chunk its own noise, as in diffusion forcing - "sequence action forcing" in paper (0 noise -> 1 data)

                noised_times = sample_action_times_fn(batch * time).to(self.device)
                packed_args, packed_kwargs = set_arg(packed_args, packed_kwargs, self.times_arg, noised_times)

            # unreduced loss for training if needed

            if exists(self.unreduced_loss_arg) and not has_arg(args, kwargs, self.unreduced_loss_arg) and not has_arg(packed_args, packed_kwargs, self.unreduced_loss_arg):
                packed_args, packed_kwargs = set_arg(packed_args, packed_kwargs, self.unreduced_loss_arg, True)

            return packed_args, packed_kwargs

        out, next_fast_weights = self._forward_with_time_packing(
            self.model,
            args,
            kwargs,
            prev_fast_weights = prev_fast_weights,
            auto_unsqueeze_time = False,
            before_forward_fn = before_forward
        )

        # reduce loss over extra dimensions past (batch, time) if needed

        flat_out, tree_unflatten = tree_flatten_with_inverse(out)
        loss = flat_out[self.loss_arg]

        if is_tensor(loss) and loss.ndim > 2:
            loss = reduce(loss, 'b t ... -> b t', 'mean')

        # loss masking for imitation / dagger distillation - section 3.3

        if exists(loss_mask):
            assert loss_mask.ndim == 2, f'loss_mask must be 2d tensor of shape (batch, time)'
            loss = masked_mean(loss, loss_mask)

        flat_out[self.loss_arg] = loss
        out = tree_unflatten(flat_out)

        if not return_fast_weights:
            return out

        return out, next_fast_weights
