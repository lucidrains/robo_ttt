from __future__ import annotations

from copy import deepcopy
from functools import partial
from collections import namedtuple
from collections.abc import Sequence, Callable
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor
from torch.distributions import Beta
from torch.nn import Module, ModuleList, Parameter
from torch.func import grad, vmap, functional_call

from einx import multiply
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from torch_einops_utils import pack_with_inverse, tree_map_detach, tree_map_tensor, tree_flatten_with_inverse, masked_mean, maybe_return
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

def safe_div(num, den, eps = 1e-8):
    if is_tensor(den):
        return num / den.clamp(min = eps)
    return num / max(den, eps)

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

def get_arg_or_none(args: tuple, kwargs: dict, arg_key: ArgKey):
    return get_arg(args, kwargs, arg_key) if has_arg(args, kwargs, arg_key) else None

def map_args(args: tuple, kwargs: dict, fn: Callable):
    # apply fn to each arg and kwarg, dispatching on key type (int positional, str keyword)

    args = tuple(fn(idx, arg) for idx, arg in enumerate(args))
    kwargs = {key: fn(key, val) for key, val in kwargs.items()}
    return args, kwargs

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
        fw_pkm: Module,
        *,
        use_delta_fast_weights = True
    ):
        super().__init__()
        self.fw_pkm = fw_pkm
        self.dim = getattr(fw_pkm, 'dim', None)
        self.use_delta_fast_weights = use_delta_fast_weights

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

        if self.use_delta_fast_weights and exists(past_memories):
            mv = mv - past_memories.memory_values
            keys = keys - past_memories.keys

        delta_fast_weights = dict(memory_values = mv, keys = keys)

        return out, delta_fast_weights

# fwa memory wrapper

class fWAWrapper(Module):
    def __init__(
        self,
        fwa: Module,
        *,
        use_delta_fast_weights = False
    ):
        super().__init__()
        self.fwa = fwa
        self.dim = getattr(fwa, 'dim', None)
        self.use_delta_fast_weights = use_delta_fast_weights

    def forward(
        self,
        tokens,
        prev_fast_weights = None
    ):
        from fast_weight_attention import FastWeightState

        past_state = None

        if exists(prev_fast_weights):
            fw = normalize_memory(prev_fast_weights).fast_weights
            if exists(fw):
                if isinstance(fw, dict):
                    token_count = fw.get('token_count', 0)
                    memory = {k: v for k, v in fw.items() if k != 'token_count'}
                    past_state = FastWeightState(memory, token_count)
                else:
                    past_state = FastWeightState(fw)

        out, next_state = self.fwa(
            tokens,
            past_mem = past_state,
            return_next_memories = True
        )

        next_memories = dict(next_state.memory)

        if self.use_delta_fast_weights and exists(past_state):
            next_memories = {k: v - past_state.memory[k] for k, v in next_memories.items()}
        else:
            next_memories['token_count'] = next_state.token_count

        return out, next_memories

# ttt wrapper

TTTWrapperIntermediates = namedtuple('TTTWrapperIntermediates', ('memory_out', 'next_fast_weights', 'delta_fast_weights'))

class TTTWrapper(Module):

    def __init__(
        self,
        dim,
        *,
        memory: Module, # the memory module type
        select_tokens_slice = None,
        tbptt_step_size = None,
        use_delta_fast_weights = None # whether the memory module returns the change in fast weights (delta) or the full next state; defaults to the memory module's convention
    ):
        super().__init__()
        self.dim = dim
        self.memory = memory
        self.select_tokens_slice = select_tokens_slice
        self.tbptt_step_size = tbptt_step_size
        self.memory_out_layerscale = nn.Parameter(torch.randn(dim) * 1e-4)
        self.use_delta_fast_weights = default(use_delta_fast_weights, getattr(self.memory, 'use_delta_fast_weights', True))

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

            if self.use_delta_fast_weights:
                next_fast_weights = add_dict(curr_memory.fast_weights, delta_fast_weights)
            else:
                next_fast_weights = delta_fast_weights

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

# next latent prediction (forward dynamics A -> S(t+1)) - Xu et al. (https://arxiv.org/abs/2609.09630)

def target_rms_normalized_mse(pred: Tensor, target: Tensor, eps = 1e-6):
    scale = target.shape[-1] ** -0.5
    target_rms = target.norm(dim = -1, keepdim = True) * scale
    return (safe_div(pred - target, target_rms, eps = eps) ** 2).mean()

class NextLatentPredictor(Module):
    def __init__(
        self,
        dim = None,
        *,
        dim_state = None,
        net: Module | None = None,
        action_tokens_slice = None,
        state_tokens_slice = None,
        loss_ratio = 0.1,
        action_loss_weight = 1.0,
        min_loss_ratio = 1e-6,
        max_loss_ratio = 0.1,
        eps = 1e-8
    ):
        super().__init__()

        self.action_tokens_slice = action_tokens_slice
        self.state_tokens_slice = state_tokens_slice

        self.loss_ratio = loss_ratio
        self.action_loss_weight = action_loss_weight
        self.min_loss_ratio = min_loss_ratio
        self.max_loss_ratio = max_loss_ratio
        self.eps = eps

        if exists(net):
            self.net = net
        else:
            assert exists(dim), 'dim must be passed if net is not provided'
            dim_state = default(dim_state, dim)
            dim_in = dim + dim_state

            self.net = nn.Sequential(
                nn.RMSNorm(dim_in),
                nn.Linear(dim_in, dim_state),
                nn.SiLU(),
                nn.Linear(dim_state, dim_state)
            )

    def forward(self, action_tokens, state_tokens):
        if exists(self.action_tokens_slice):
            action_tokens = action_tokens[..., self.action_tokens_slice, :]

        if exists(self.state_tokens_slice) and state_tokens.ndim == action_tokens.ndim:
            state_tokens = state_tokens[..., self.state_tokens_slice, :]

        pooled_action_tokens = reduce(action_tokens, '... n d -> ... d', 'mean')

        pooled_state_tokens = state_tokens
        if state_tokens.ndim == action_tokens.ndim:
            pooled_state_tokens = reduce(state_tokens, '... n d -> ... d', 'mean')

        return self.net(torch.cat((pooled_action_tokens, pooled_state_tokens), dim = -1))

    def loss(
        self,
        action_tokens,
        curr_state,
        next_state,
        action_loss
    ):
        pred = self(action_tokens, curr_state)

        target = next_state.detach()

        if exists(self.state_tokens_slice) and target.ndim > pred.ndim:
            target = target[..., self.state_tokens_slice, :]

        if target.ndim > pred.ndim:
            target = reduce(target, '... n d -> ... d', 'mean')

        next_latent_loss = target_rms_normalized_mse(pred, target)

        ratio = safe_div(
            self.loss_ratio * self.action_loss_weight * action_loss.detach().mean(),
            next_latent_loss.detach().mean(),
            eps = self.eps
        ).clamp(
            min = self.min_loss_ratio,
            max = self.max_loss_ratio
        )

        return self.action_loss_weight * action_loss + ratio * next_latent_loss

# main robo-ttt class

class RoboTTT(Module):
    def __init__(
        self,
        model: Module,
        *,
        ttt_wrapper: TTTWrapper | Module,
        ttt_module_paths: ModulePaths = (),
        action_tokens_module_path: str | int | None = None,
        batch_time_arg = 0,
        expand_time_args: ArgKeys = (),
        omit_time_args: ArgKeys = (),
        times_arg: ArgKey | None = 'time',
        unreduced_loss_arg: ArgKey | None = 'return_unreduced_loss',
        loss_arg: ArgKey = 0,
        sample_action_times_fn: Callable | None = None,
        predict_next_latent: bool = False,
        dim_state: int | None = None,
        next_latent_predictor_kwargs: dict = dict(),
        next_latent_predictor: NextLatentPredictor | bool | None = None,
        state_arg: ArgKey | None = None
    ):
        super().__init__()
        self.model = model
        self.ttt_wrapper = ttt_wrapper

        # arg keys

        self.batch_time_arg = batch_time_arg
        self.expand_time_args = set(expand_time_args)
        self.omit_time_args = set(omit_time_args)
        self.non_temporal_keys = self.expand_time_args | self.omit_time_args
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

        # action module index (which wrapper's output corresponds to action tokens)

        if isinstance(action_tokens_module_path, int):
            assert -len(self.ttt_wrappers) <= action_tokens_module_path < len(self.ttt_wrappers), f'action_tokens_module_path integer index {action_tokens_module_path} is out of range for {len(self.ttt_wrappers)} ttt wrappers'
            self.action_module_index = action_tokens_module_path
        elif exists(action_tokens_module_path):
            assert action_tokens_module_path in self.ttt_module_paths, f'"{action_tokens_module_path}" not found in ttt_module_paths: {self.ttt_module_paths}'
            self.action_module_index = self.ttt_module_paths.index(action_tokens_module_path)
        else:
            self.action_module_index = -1

        # next latent predictor

        self.next_latent_predictor = None

        if isinstance(next_latent_predictor, NextLatentPredictor):
            self.next_latent_predictor = next_latent_predictor
        elif isinstance(next_latent_predictor, Module):
            self.next_latent_predictor = NextLatentPredictor(net = next_latent_predictor, **next_latent_predictor_kwargs)
        elif predict_next_latent or next_latent_predictor is True:
            assert len(self.ttt_wrappers) > 0, 'at least one ttt wrapper is required for next latent prediction'

            action_wrapper = self.ttt_wrappers[self.action_module_index]
            layerscale = getattr(action_wrapper, 'memory_out_layerscale', None)
            dim = default(getattr(action_wrapper, 'dim', None), layerscale.shape[0] if exists(layerscale) else None)

            assert exists(dim), f'could not infer dimension from wrapper of type {action_wrapper.__class__.__name__}, please set .dim or pass in next_latent_predictor'

            self.next_latent_predictor = NextLatentPredictor(
                dim = dim,
                dim_state = dim_state,
                **next_latent_predictor_kwargs
            )

        self.state_arg = state_arg

        # default prev fast weights

        self.default_prev_fast_weights = (None,) * len(self.ttt_wrappers)

    @property
    def device(self):
        return module_device(self)

    def finetune_parameters(self):
        predictor_params = self.next_latent_predictor.parameters() if exists(self.next_latent_predictor) else ()
        return chain(self.ttt_wrappers.parameters(), predictor_params)

    def _normalize_prev_fast_weights(self, prev_fast_weights):
        prev_fast_weights = cast_tuple(default(prev_fast_weights, self.default_prev_fast_weights))
        assert len(prev_fast_weights) == len(self.ttt_wrappers)
        return tuple(normalize_memory(fw) for fw in prev_fast_weights)

    def _forward_with_ttt(self, fn: Callable, args: tuple, kwargs: dict, time: int, prev_fast_weights):
        handles = []
        layer_fast_weights = list(prev_fast_weights)
        all_action_chunks = [None] * len(self.ttt_wrappers)
        hook_call_counts = [0] * len(self.ttt_wrappers)

        # helpers for packing and unpacking time into batch dimension

        pack_batch_time = partial(rearrange, pattern = 'b t ... -> (b t) ...')
        unpack_batch_time = partial(rearrange, pattern = '(b t) ... -> b t ...', t = time)

        # register forward hooks

        for idx, (submodule, ttt_wrapper) in enumerate(zip(self.submodules, self.ttt_wrappers)):
            def hook(module, input, output, ttt_wrapper = ttt_wrapper, idx = idx):
                hook_call_counts[idx] += 1

                assert output.ndim == 3, f'module output must be 3d tensor of shape (batch * time, action_chunk_len, dim), but got shape {output.shape} for module "{module.__class__.__name__}"'

                prev_fast_weight = layer_fast_weights[idx]

                # ttt memory forward

                action_chunks = unpack_batch_time(output)
                ttt_memory_out, next_fast_weight, _ = ttt_wrapper(action_chunks, prev_fast_weights = prev_fast_weight)
                layer_fast_weights[idx] = next_fast_weight

                # policy consumes the module output modulated with the ttt memory output

                out = output + pack_batch_time(ttt_memory_out) if exists(ttt_memory_out) else output
                all_action_chunks[idx] = unpack_batch_time(out)

                return out

            handles.append(submodule.register_forward_hook(hook))

        # forward model

        try:
            out = fn(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        # ensure every ttt module was on the forward path, otherwise its memory was silently skipped

        for path, submodule, count in zip(self.ttt_module_paths, self.submodules, hook_call_counts):
            assert count > 0, f'ttt module "{path}" ({submodule.__class__.__name__}) was never invoked during the forward pass, so its ttt memory was silently skipped - if it is a container such as a ModuleList, hook the invoked child instead'

        action_chunks = all_action_chunks[self.action_module_index] if len(all_action_chunks) > 0 else None

        return out, tuple(layer_fast_weights), action_chunks

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
            unsqueeze_time = lambda t: rearrange(t, 'b ... -> b 1 ...')

            args, kwargs = map_args(args, kwargs, lambda key, val: val if key in self.non_temporal_keys else tree_map_tensor(unsqueeze_time, val))

        # normalize fast weights

        prev_fast_weights = self._normalize_prev_fast_weights(prev_fast_weights)

        # batch time

        batch, time, pack_batch_time, unpack_batch_time = self._get_batch_time(args, kwargs)

        # transform each arg once: repeat across time (expand), pass through (omit), or pack time into batch

        def transform_arg(key, val):
            if key in self.expand_time_args:
                return tree_map_tensor(lambda t: repeat(t, 'b ... -> (b t) ...', t = time), val)
            if key in self.omit_time_args:
                return val
            return tree_map_tensor(pack_batch_time, val)

        packed_args, packed_kwargs = map_args(args, kwargs, transform_arg)

        if exists(before_forward_fn):
            packed_args, packed_kwargs = before_forward_fn(batch, time, args, kwargs, packed_args, packed_kwargs)

        # forward model with ttt hooks

        out, next_fast_weights, action_chunks = self._forward_with_ttt(fn, packed_args, packed_kwargs, time, prev_fast_weights)

        # unpack time

        out = tree_map_tensor(unpack_batch_time, out)

        if auto_unsqueeze_time:
            out = tree_map_tensor(lambda t: rearrange(t, 'b 1 ... -> b ...'), out)
            action_chunks = rearrange(action_chunks, 'b 1 ... -> b ...') if exists(action_chunks) else None

        return out, next_fast_weights, action_chunks

    @maybe_return('fast_weights', 'pred_next_latent', primary = 'actions')
    def _sample(
        self,
        *args,
        sample_fn_name = None,
        prev_fast_weights = None,
        auto_unsqueeze_time = True,
        return_fast_weights = False,
        return_pred_next_latent = False,
        **kwargs
    ):
        # resolve sample fn

        sample_fn_name = default(sample_fn_name, 'sample' if hasattr(self.model, 'sample') else ('sample_actions' if hasattr(self.model, 'sample_actions') else '__call__'))
        sample_fn = getattr(self.model, sample_fn_name) if sample_fn_name != '__call__' else self.model

        out, next_fast_weights, action_chunks = self._forward_with_time_packing(
            sample_fn,
            args,
            kwargs,
            prev_fast_weights = prev_fast_weights,
            auto_unsqueeze_time = auto_unsqueeze_time
        )

        pred_next_latent = None

        if return_pred_next_latent and exists(self.next_latent_predictor):
            assert exists(action_chunks), 'action chunks were not captured, so the next latent prediction would be silently skipped - ensure action_tokens_module_path points to a module on the forward path'

            state = get_arg_or_none(args, kwargs, self.state_arg)

            assert is_tensor(state), 'state must be passed in to predict the next latent (see state_arg)'

            curr_action_chunks = action_chunks[:, -1] if action_chunks.ndim == 4 else action_chunks

            # when auto_unsqueeze_time is True (the default at sample time), a 3d state is (batch, tokens, dim) of the current observation
            # otherwise the state keeps an explicit time dimension, and the last timestep is taken

            state_has_time = is_tensor(state) and (state.ndim == 4 or (state.ndim == 3 and not auto_unsqueeze_time))

            if state_has_time:
                state = state[:, -1]

            pred_next_latent = self.next_latent_predictor(curr_action_chunks, state)

        return out, next_fast_weights, pred_next_latent

    def sample(
        self,
        *args,
        prev_fast_weights = None,
        return_fast_weights = None,
        **kwargs
    ):
        return_fast_weights = default(return_fast_weights, exists(prev_fast_weights))

        return self._sample(
            *args,
            prev_fast_weights = prev_fast_weights,
            return_fast_weights = return_fast_weights,
            **kwargs
        )

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

        out, next_fast_weights, action_chunks = self._forward_with_time_packing(
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

        # next latent prediction (forward dynamics A -> S(t+1))

        if exists(self.next_latent_predictor):
            assert exists(action_chunks), 'action chunks were not captured, so the next latent prediction loss would be silently skipped - ensure action_tokens_module_path points to a module on the forward path'

            if action_chunks.shape[1] > 1:
                state = get_arg_or_none(args, kwargs, self.state_arg)

                assert is_tensor(state), 'state must be passed in for next latent prediction (see state_arg)'

                action_tokens = action_chunks[:, :-1]
                curr_state, next_state = state[:, :-1], state[:, 1:]

                loss = self.next_latent_predictor.loss(
                    action_tokens,
                    curr_state,
                    next_state,
                    action_loss = loss
                )

        flat_out[self.loss_arg] = loss
        out = tree_unflatten(flat_out)

        if not return_fast_weights:
            return out

        return out, next_fast_weights
