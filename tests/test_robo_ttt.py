import pytest

import torch
from torch import nn

param = pytest.mark.parametrize

def exists(t):
    return t is not None

@param('muon_update', [False, True])
@param('muon_param_names', [None, ('0.weight',)])
@param('learned_forget', [False, True])
@param('rotary_embed', [False, True])
def test_memory_key_value_bind(
    muon_update,
    muon_param_names,
    learned_forget,
    rotary_embed
):
    from robo_ttt.robo_ttt import MemoryKeyValueBind, TTTWrapper

    dim = 16
    memory_network = nn.Sequential(
        nn.Linear(dim, 32),
        nn.GELU(),
        nn.Linear(32, dim)
    )

    memory = MemoryKeyValueBind(
        dim,
        memory_network,
        muon_update = muon_update,
        muon_param_names = muon_param_names,
        learned_forget = learned_forget,
        rotary_embed_qk = rotary_embed
    )

    tokens = torch.randn(2, 4, dim)

    # memory standalone

    output1, next_fast_weights1 = memory(tokens)
    assert output1.shape == (2, 4, dim)
    assert set(next_fast_weights1.keys()) == set(memory_network.state_dict().keys())

    output2, next_fast_weights2 = memory(tokens, prev_fast_weights = next_fast_weights1)
    assert output2.shape == (2, 4, dim)

    # ttt wrapper

    wrapper = TTTWrapper(dim, memory = memory)

    output1, next_fast_weights1, _ = wrapper(tokens)
    assert output1.shape == (2, 4, dim)
    assert next_fast_weights1.step == 1

    output2, next_fast_weights2, _ = wrapper(tokens, prev_fast_weights = next_fast_weights1)
    assert output2.shape == (2, 4, dim)
    assert next_fast_weights2.step == 2

    # multiple action chunks (with time dimension)

    multiple_chunks = torch.randn(2, 5, 4, dim)

    multi_output1, multi_next_fast_weights1, _ = wrapper(multiple_chunks)
    assert multi_output1.shape == (2, 5, 4, dim)
    assert multi_next_fast_weights1.step == 5

    multi_output2, multi_next_fast_weights2, _ = wrapper(multiple_chunks, prev_fast_weights = multi_next_fast_weights1)
    assert multi_output2.shape == (2, 5, 4, dim)
    assert multi_next_fast_weights2.step == 10

    # assert equivalence between sequential (one at a time) vs multi-chunk (all at once)

    seq_outputs = []
    curr_fast_weights = None

    for chunk in multiple_chunks.unbind(dim = 1):
        out_chunk, curr_fast_weights, _ = wrapper(chunk, prev_fast_weights = curr_fast_weights)
        seq_outputs.append(out_chunk)

    seq_outputs = torch.stack(seq_outputs, dim = 1)

    assert torch.allclose(multi_output1, seq_outputs, atol = 1e-5)

    assert multi_next_fast_weights1.step == curr_fast_weights.step == 5
    for k in multi_next_fast_weights1.fast_weights:
        assert torch.allclose(multi_next_fast_weights1.fast_weights[k], curr_fast_weights.fast_weights[k], atol = 1e-5)

def test_readme_basic_example():
    from robo_ttt import MemoryKeyValueBind, TTTWrapper

    memory_network = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 512)
    )

    memory = MemoryKeyValueBind(512, memory_network)
    ttt_wrapper = TTTWrapper(512, memory = memory)

    attended_action_chunks = torch.randn(2, 5, 4, 512)

    out, next_fast_weights, _ = ttt_wrapper(attended_action_chunks)

    assert out.shape == attended_action_chunks.shape

def test_readme_full_policy_example():
    from mimic_video import MimicVideo
    from robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    memory_network = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 512)
    )

    memory = MemoryKeyValueBind(512, memory_network)
    ttt_wrapper = TTTWrapper(512, memory = memory)

    policy = MimicVideo(
        dim = 512,
        dim_video_hidden = 512,
        depth = 2,
        dim_head = 64,
        heads = 8,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        policy,
        ttt_wrapper = ttt_wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',),
        times_arg = 'time'
    )

    video_hiddens = torch.rand(2, 3, 5, 512)
    joint_state = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 32, 4)
    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])

    loss_mask = torch.tensor([[True, False, True], [False, True, True]])

    loss = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens,
        actions = actions,
        joint_state = joint_state,
        loss_mask = loss_mask
    )

    loss.backward()

    init_video_hiddens = torch.randn(2, 5, 512)
    init_joint_state = torch.randn(2, 4)

    actions_t1, fast_weights1 = model.sample(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = init_video_hiddens,
        joint_state = init_joint_state,
        steps = 4,
        batch_size = 2,
        auto_unsqueeze_time = True,
        return_fast_weights = True
    )

    assert actions_t1.shape == (2, 32, 4)
    assert len(fast_weights1) == 1

def test_sequence_action_forcing():
    from mimic_video import MimicVideo
    from robo_ttt.robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper, sample_action_times

    times = sample_action_times(6)
    assert times.shape == (6,)

    dim = 16
    memory_network = nn.Sequential(
        nn.Linear(dim, 32),
        nn.GELU(),
        nn.Linear(32, dim)
    )

    memory = MemoryKeyValueBind(dim, memory_network)
    wrapper = TTTWrapper(dim, memory = memory)

    vam_model = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        vam_model,
        ttt_wrapper = wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',)
    )

    video_hiddens = torch.rand(2, 3, 5, dim)
    joint_state = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 32, 4)
    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])

    loss = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens,
        actions = actions,
        joint_state = joint_state,
        return_unreduced_loss = True
    )

    assert loss.shape == (2, 3)

@param('has_loss_mask', [False, True])
def test_robo_ttt_loss_mask(has_loss_mask):
    from mimic_video import MimicVideo
    from robo_ttt.robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    dim = 16
    memory_network = nn.Sequential(
        nn.Linear(dim, 32),
        nn.GELU(),
        nn.Linear(32, dim)
    )

    memory = MemoryKeyValueBind(dim, memory_network)
    wrapper = TTTWrapper(dim, memory = memory)

    vam_model = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        vam_model,
        ttt_wrapper = wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',)
    )

    video_hiddens = torch.rand(2, 3, 5, dim)
    joint_state = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 32, 4)
    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])

    loss_mask = torch.tensor([[True, False, True], [False, True, True]]) if has_loss_mask else None

    loss = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens,
        actions = actions,
        joint_state = joint_state,
        return_unreduced_loss = True,
        loss_mask = loss_mask
    )

    if has_loss_mask:
        assert loss.ndim == 0
    else:
        assert loss.shape == (2, 3)

@param('rotary_embed', [False, True])
def test_robo_ttt_time_sequence_equivalence(rotary_embed):
    from mimic_video import MimicVideo
    from robo_ttt.robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    dim = 16
    memory_network = nn.Sequential(
        nn.Linear(dim, 32),
        nn.GELU(),
        nn.Linear(32, dim)
    )

    memory = MemoryKeyValueBind(dim, memory_network, rotary_embed_qk = rotary_embed)
    wrapper = TTTWrapper(dim, memory = memory)

    vam_model = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        vam_model,
        ttt_wrapper = wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',),
        times_arg = 'time'
    )

    # full sequence of t = 5 timesteps (b = 2, t = 5)

    video_hiddens = torch.rand(2, 5, 5, dim)
    joint_state = torch.randn(2, 5, 4)
    actions = torch.randn(2, 5, 32, 4)
    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])
    time = torch.rand(2, 5)

    # 1. process full sequence all at once (t = 5)

    torch.manual_seed(42)

    out_full, next_fast_weights_full = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens,
        actions = actions,
        joint_state = joint_state,
        time = time,
        return_unreduced_loss = True,
        return_fast_weights = True
    )

    # 2. process chunk 1 (t = 2 timesteps)

    torch.manual_seed(42)

    out_chunk1, next_fast_weights_chunk1 = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens[:, :2],
        actions = actions[:, :2],
        joint_state = joint_state[:, :2],
        time = time[:, :2],
        return_unreduced_loss = True,
        return_fast_weights = True
    )

    # 3. process chunk 2 (t = 3 timesteps) passing prev_fast_weights = next_fast_weights_chunk1

    out_chunk2, next_fast_weights_chunk2 = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens[:, 2:],
        actions = actions[:, 2:],
        joint_state = joint_state[:, 2:],
        time = time[:, 2:],
        prev_fast_weights = next_fast_weights_chunk1,
        return_unreduced_loss = True,
        return_fast_weights = True
    )

    out_seq = torch.cat((out_chunk1, out_chunk2), dim = 1)

    # output equivalence

    assert torch.allclose(out_full, out_seq, atol = 1e-5)

    # fast weights equivalence across all hooked ttt layers

    assert len(next_fast_weights_full) == len(next_fast_weights_chunk2) == 1

    for layer_mem_full, layer_mem_chunk2 in zip(next_fast_weights_full, next_fast_weights_chunk2):
        assert layer_mem_full.step == layer_mem_chunk2.step == 5
        for k in layer_mem_full.fast_weights:
            assert torch.allclose(layer_mem_full.fast_weights[k], layer_mem_chunk2.fast_weights[k], atol = 1e-5)

def test_robo_ttt_sample():
    from mimic_video import MimicVideo
    from robo_ttt.robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    dim = 16
    memory = MemoryKeyValueBind(dim, nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim)))
    wrapper = TTTWrapper(dim, memory = memory)

    vam_model = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        vam_model,
        ttt_wrapper = wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',)
    )

    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])
    obs_t1_hiddens = torch.randn(2, 5, dim)
    obs_t1_joint = torch.randn(2, 4)

    # step 1: timestep 1 observation rollout

    actions_t1, fast_weights1 = model.sample(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = obs_t1_hiddens,
        joint_state = obs_t1_joint,
        steps = 2,
        batch_size = 2,
        return_fast_weights = True
    )

    assert actions_t1.shape == (2, 32, 4)
    assert len(fast_weights1) == 1

    # step 2: timestep 2 observation rollout passing prior fast weights fast_weights1

    obs_t2_hiddens = torch.randn(2, 5, dim)
    obs_t2_joint = torch.randn(2, 4)

    actions_t2, fast_weights2 = model.sample(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = obs_t2_hiddens,
        joint_state = obs_t2_joint,
        steps = 2,
        batch_size = 2,
        prev_fast_weights = fast_weights1,
        return_fast_weights = True
    )

    assert actions_t2.shape == (2, 32, 4)
    assert len(fast_weights2) == 1

def test_multistep_denoising_sample():
    from mimic_video import MimicVideo
    from robo_ttt.robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    dim = 16
    memory = MemoryKeyValueBind(dim, nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim)))
    wrapper = TTTWrapper(dim, memory = memory)

    vam_model = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        vam_model,
        ttt_wrapper = wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',)
    )

    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])
    video_hiddens = torch.randn(2, 5, dim)
    joint_state = torch.randn(2, 4)

    actions, fast_weights = model.sample(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens,
        joint_state = joint_state,
        steps = 4,
        batch_size = 2,
        return_fast_weights = True
    )

    assert actions.shape == (2, 32, 4)
    assert len(fast_weights) == 1

def test_pi_zero_robo_ttt():
    from pi_zero_pytorch import PiZero
    from robo_ttt.robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    dim = 16
    num_tokens = 64
    dim_action_input = 4
    dim_joint_state = 4
    dim_action = 4

    base_pizero = PiZero(
        dim = dim,
        num_tokens = num_tokens,
        dim_action_input = dim_action_input,
        dim_joint_state = dim_joint_state,
        dim_action = dim_action,
        depth = 2,
        dim_head = 8,
        heads = 2
    )

    memory = MemoryKeyValueBind(dim, nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim)))
    wrapper = TTTWrapper(dim, memory = memory)

    model = RoboTTT(
        base_pizero,
        ttt_wrapper = wrapper,
        ttt_module_paths = ('layers.0',),
        batch_time_arg = 'visual_tokens',
        expand_time_args = ('token_ids',),
        omit_time_args = ('token_ids',),
        times_arg = 'times'
    )

    # forward loss over sequence t = 3

    visual_tokens = torch.randn(2, 3, 4, 16)
    token_ids = torch.randint(0, 64, (2, 4))
    joint_state = torch.randn(2, 3, dim_joint_state)
    actions = torch.randn(2, 3, 4, dim_action)

    loss, fast_weights = model(
        images = None,
        visual_tokens = visual_tokens,
        token_ids = token_ids,
        joint_state = joint_state,
        actions = actions,
        return_fast_weights = True
    )

    assert exists(loss)
    assert len(fast_weights) == 1

@param('auto_unsqueeze_time', [False, True])
def test_sample_auto_unsqueeze_time(auto_unsqueeze_time):
    from mimic_video import MimicVideo
    from robo_ttt import RoboTTT, MemoryKeyValueBind, TTTWrapper

    dim = 16
    memory_network = nn.Sequential(
        nn.Linear(dim, 32),
        nn.GELU(),
        nn.Linear(32, dim)
    )

    memory = MemoryKeyValueBind(dim, memory_network)
    ttt_wrapper = TTTWrapper(dim, memory = memory)

    policy = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        policy,
        ttt_wrapper = ttt_wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',),
        times_arg = 'time'
    )

    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])

    if auto_unsqueeze_time:
        obs_video_hiddens = torch.randn(2, 5, dim)
        obs_joint_state = torch.randn(2, 4)

        actions_out, fast_weights = model.sample(
            prompt_token_ids = prompt_token_ids,
            video_hiddens = obs_video_hiddens,
            joint_state = obs_joint_state,
            steps = 2,
            batch_size = 2,
            auto_unsqueeze_time = True,
            return_fast_weights = True
        )

        assert actions_out.shape == (2, 32, 4)
    else:
        obs_video_hiddens = torch.randn(2, 1, 5, dim)
        obs_joint_state = torch.randn(2, 1, 4)

        actions_out, fast_weights = model.sample(
            prompt_token_ids = prompt_token_ids,
            video_hiddens = obs_video_hiddens,
            joint_state = obs_joint_state,
            steps = 2,
            batch_size = 2,
            auto_unsqueeze_time = False,
            return_fast_weights = True
        )

        assert actions_out.shape == (2, 1, 32, 4)

    assert len(fast_weights) == 1

def test_custom_wrapper_lstm():
    from einops import rearrange
    from torch_einops_utils import pack_with_inverse
    from mimic_video import MimicVideo
    from robo_ttt.robo_ttt import RoboTTT, Memory, normalize_memory

    class CustomLSTMTTTWrapper(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lstm = nn.LSTM(dim, dim, batch_first = True)

        def forward(self, action_chunks, prev_fast_weights = None):
            curr_memory = normalize_memory(prev_fast_weights)

            action_chunks, unpack_time = pack_with_inverse(action_chunks, 'b * n d')
            b, t, n, d = action_chunks.shape

            x = rearrange(action_chunks, 'b t n d -> b (t n) d')

            out, next_fast_weights = self.lstm(x, curr_memory.fast_weights)

            out = rearrange(out, 'b (t n) d -> b t n d', t = t, n = n)
            return unpack_time(out), Memory(next_fast_weights, curr_memory.step + t), None

    dim = 16
    custom_wrapper = CustomLSTMTTTWrapper(dim)

    policy = MimicVideo(
        dim = dim,
        dim_video_hidden = dim,
        depth = 2,
        dim_head = 8,
        heads = 2,
        dim_action = 4,
        dim_joint_state = 4
    )

    model = RoboTTT(
        policy,
        ttt_wrapper = custom_wrapper,
        ttt_module_paths = ('to_action_tokens',),
        batch_time_arg = 'video_hiddens',
        expand_time_args = ('prompt_token_ids',),
        times_arg = 'time'
    )

    video_hiddens = torch.rand(2, 3, 5, dim)
    joint_state = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 32, 4)
    prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])

    loss, fast_weights = model(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = video_hiddens,
        actions = actions,
        joint_state = joint_state,
        return_fast_weights = True
    )

    loss.sum().backward()

    obs_hiddens = torch.randn(2, 5, dim)
    obs_joint = torch.randn(2, 4)

    actions_t1, fast_weights1 = model.sample(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = obs_hiddens,
        joint_state = obs_joint,
        steps = 2,
        batch_size = 2,
        auto_unsqueeze_time = True,
        return_fast_weights = True
    )

    actions_t2, fast_weights2 = model.sample(
        prompt_token_ids = prompt_token_ids,
        video_hiddens = obs_hiddens,
        joint_state = obs_joint,
        steps = 2,
        batch_size = 2,
        prev_fast_weights = fast_weights1,
        auto_unsqueeze_time = True,
        return_fast_weights = True
    )

    assert actions_t1.shape == (2, 32, 4)
    assert actions_t2.shape == (2, 32, 4)
    assert len(fast_weights1) == 1 and len(fast_weights2) == 1
