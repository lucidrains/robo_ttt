import torch
from torch.nn import Module

import einx
from einops import rearrange

from torch_einops_utils import pack_with_inverse

# helpers

def exists(t):
    return t is not None

def default(t, d):
    return t if exists(t) else d

# classes

