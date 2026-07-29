import pytest

import torch
from torch import nn

param = pytest.mark.parametrize

def exists(t):
    return t is not None

def test_robo_ttt():
    from robo_ttt.robo_ttt import RoboTTT

    model = RoboTTT()
    assert True

@param('muon_update', [False, True])
@param('muon_param_names', [None, ('0.weight',)])
@param('learned_forget', [False, True])
def test_memory_key_value_bind(
    muon_update,
    muon_param_names,
    learned_forget
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
        learned_forget = learned_forget
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

    output2, next_fast_weights2, _ = wrapper(tokens, prev_fast_weights = next_fast_weights1)
    assert output2.shape == (2, 4, dim)

    # multiple action chunks (with time dimension)

    multiple_chunks = torch.randn(2, 5, 4, dim)

    multi_output1, multi_next_fast_weights1, _ = wrapper(multiple_chunks)
    assert multi_output1.shape == (2, 5, 4, dim)

    multi_output2, multi_next_fast_weights2, _ = wrapper(multiple_chunks, prev_fast_weights = multi_next_fast_weights1)
    assert multi_output2.shape == (2, 5, 4, dim)

    # assert equivalence between sequential (one at a time) vs multi-chunk (all at once)

    seq_outputs = []
    curr_fast_weights = None

    for chunk in multiple_chunks.unbind(dim = 1):
        out_chunk, curr_fast_weights, _ = wrapper(chunk, prev_fast_weights = curr_fast_weights)
        seq_outputs.append(out_chunk)

    seq_outputs = torch.stack(seq_outputs, dim = 1)

    assert torch.allclose(multi_output1, seq_outputs, atol = 1e-5)

    for k in multi_next_fast_weights1:
        assert torch.allclose(multi_next_fast_weights1[k], curr_fast_weights[k], atol = 1e-5)
