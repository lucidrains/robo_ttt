import pytest

import torch
from torch import nn

param = pytest.mark.parametrize

def test_robo_ttt():
    from robo_ttt.robo_ttt import RoboTTT

    model = RoboTTT()
    assert True

@param('muon_update', [False, True])
@param('muon_param_names', [None, ('0.weight',)])
def test_memory_key_value_bind(
    muon_update,
    muon_param_names
):
    from robo_ttt.robo_ttt import MemoryKeyValueBind, TTTWrapper, Attention

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
        muon_param_names = muon_param_names
    )

    tokens = torch.randn(2, 4, dim)

    # memory standalone

    output1, next_fast_weights1 = memory(tokens)
    assert output1.shape == (2, 4, dim)
    assert set(next_fast_weights1.keys()) == set(memory_network.state_dict().keys())

    output2, next_fast_weights2 = memory(tokens, prev_fast_weights = next_fast_weights1)
    assert output2.shape == (2, 4, dim)

    # ttt wrapper

    block = Attention(dim)
    wrapper = TTTWrapper(dim, memory = memory, block = block)

    output1, next_fast_weights1, _ = wrapper(tokens)
    assert output1.shape == (2, 4, dim)

    output2, next_fast_weights2, _ = wrapper(tokens, prev_fast_weights = next_fast_weights1)
    assert output2.shape == (2, 4, dim)
