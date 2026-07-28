import pytest

import torch
from torch import nn

def test_robo_ttt():
    from robo_ttt.robo_ttt import RoboTTT

    model = RoboTTT()
    assert True

def test_memory_key_value_bind():
    from robo_ttt.robo_ttt import MemoryKeyValueBind, TTTWrapper

    dim = 16
    net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim))
    mem = MemoryKeyValueBind(dim, net)

    tokens = torch.randn(2, 4, dim)

    # memory standalone

    out1, next_fast_weights1 = mem(tokens)
    assert out1.shape == (2, 4, dim)
    assert set(next_fast_weights1.keys()) == set(net.state_dict().keys())

    out2, next_fast_weights2 = mem(tokens, next_fast_weights1)
    assert out2.shape == (2, 4, dim)

    # ttt wrapper

    block = nn.Linear(dim, dim)
    wrapper = TTTWrapper(dim, memory = mem, block = block)

    out1, next_fw1, _ = wrapper(tokens)
    assert out1.shape == (2, 4, dim)

    out2, next_fw2, _ = wrapper(tokens, prev_fast_weights = next_fw1)
    assert out2.shape == (2, 4, dim)
