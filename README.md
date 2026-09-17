<img src="./fig2.png" width="400"></img>

## RoboTTT

Implementation of [RoboTTT](https://research.nvidia.com/labs/gear/robottt/) proposed by Yunfan Jiang et al. of Stanford and Nvidia.

## Install

```bash
pip install robo-ttt
```

## Usage

### Basic Usage

Using `MemoryKeyValueBind` and `TTTWrapper` standalone with a 2-layer MLP memory:

```python
import torch
from torch import nn

from robo_ttt import MemoryKeyValueBind, TTTWrapper

memory_network = nn.Sequential(
    nn.Linear(512, 1024),
    nn.GELU(),
    nn.Linear(1024, 512)
)

memory = MemoryKeyValueBind(512, memory_network)
ttt_wrapper = TTTWrapper(512, memory = memory)

# attended action chunks over time (batch = 2, time = 5, seq_len = 4, dim = 512)

attended_action_chunks = torch.randn(2, 5, 4, 512)

out, next_fast_weights, _ = ttt_wrapper(attended_action_chunks)

assert out.shape == attended_action_chunks.shape
```

You can also drop in [Fast Weight Product Key Memory](https://arxiv.org/abs/2601.00671) (Tianyu Zhao & Llion Jones) directly via `fwPKMWrapper`:

```python
import torch

from robo_ttt import fwPKMWrapper, TTTWrapper
from fast_weight_product_key_memory import fwPKM

pkm = fwPKM(
    dim = 512,
    num_memories = 256 * 256,
    dim_queries_keys = 512,
    dim_values = 512
)
memory = fwPKMWrapper(pkm)
ttt_wrapper = TTTWrapper(512, memory = memory)

attended_action_chunks = torch.randn(2, 5, 4, 512)
out, next_fast_weights, _ = ttt_wrapper(attended_action_chunks)
```

### Full Policy Wrapper with MimicVideo

Wrapping a policy model (e.g. `MimicVideo`) with `RoboTTT`:

```python
import torch
from torch import nn

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
    ttt_module_paths = ('to_action_tokens.0',),
    batch_time_arg = 'video_hiddens',
    expand_time_args = ('prompt_token_ids',),
    times_arg = 'time'
)

# inputs for sequence of t = 3 timesteps (batch = 2, time = 3)

video_hiddens = torch.rand(2, 3, 5, 512)
joint_state = torch.randn(2, 3, 4)
actions = torch.randn(2, 3, 32, 4)
prompt_token_ids = torch.tensor([[10, 20, 30, -1], [15, 25, -1, -1]])

# forward training pass with loss masking

loss_mask = torch.tensor([[True, False, True], [False, True, True]])

loss = model(
    prompt_token_ids = prompt_token_ids,
    video_hiddens = video_hiddens,
    actions = actions,
    joint_state = joint_state,
    loss_mask = loss_mask
)

loss.backward()

# sampling / rollout one timestep at a time (auto_unsqueeze_time defaults to True)

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
```

### Next Latent Forward Dynamics (A → Sₜ₊₁)

Forward dynamics modeling in latent representation space across consecutive action chunks, supervising action generation with next-state representation prediction without raw pixel reconstruction ([Xu et al., 2026](https://arxiv.org/abs/2609.09630)).

To enable forward dynamics, set `predict_next_latent = True` and designate `state_arg`:

```python
model = RoboTTT(
    policy,
    ttt_wrapper = ttt_wrapper,
    ttt_module_paths = ('to_action_tokens.0',),
    action_tokens_module_path = 'to_action_tokens.0', # which wrapper outputs action tokens (default -1)
    batch_time_arg = 'video_hiddens',
    expand_time_args = ('prompt_token_ids',),
    times_arg = 'time',
    predict_next_latent = True,  # automatically creates NextLatentPredictor matching action dim
    state_arg = 'video_hiddens', # argument containing observation state tokens
    # dim_state = 4              # optional state feature dim if different from action dim (e.g. 4 for joint_state)
)
```

You can customize the dynamics loss weighting and token slicing via `next_latent_predictor_kwargs`:

```python
model = RoboTTT(
    policy,
    ttt_wrapper = ttt_wrapper,
    ttt_module_paths = ('to_action_tokens.0',),
    batch_time_arg = 'video_hiddens',
    expand_time_args = ('prompt_token_ids',),
    times_arg = 'time',
    predict_next_latent = True,
    state_arg = 'video_hiddens',
    next_latent_predictor_kwargs = dict(
        loss_ratio = 0.1,                   # loss ratio target (default 0.1)
        action_loss_weight = 1.0,           # action loss weight (default 1.0)
        action_tokens_slice = slice(0, 32), # optional slice if tokens are concatenated
        state_tokens_slice = None,
    )
)
```

During rollout sampling, request the predicted next-latent representation:

```python
actions, fast_weights, pred_next_latent = model.sample(
    prompt_token_ids = prompt_token_ids,
    video_hiddens = init_video_hiddens,
    joint_state = init_joint_state,
    steps = 4,
    batch_size = 2,
    return_fast_weights = True,
    return_pred_next_latent = True
)
```

## Citations

```bibtex
@article{jiang2026robottt0,
    title   = {RoboTTT: Context Scaling for Robot Policies},
    author  = {Yunfan Jiang and Yevgen Chebotar and Ruijie Zheng and Fengyuan Hu and Yunhao Ge and Jimmy Wu and Tianyuan Dai and Scott Reed and Li Fei-Fei and Yuke Zhu and Linxi "Jim" Fan},
    year    = {2026},
    journal = {arXiv preprint arXiv: 2607.15275}
}
```

```bibtex
@misc{zhao2026fastweightproductkeymemory,
    title   = {Fast-weight Product Key Memory},
    author  = {Tianyu Zhao and Llion Jones},
    year    = {2026},
    eprint  = {2601.00671},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2601.00671},
}
```

```bibtex
@article{xu2026jepapolicy,
    title   = {JEPA Policy: Diffusion-Free Imitation Learning via Paired Action and Future Representation Prediction},
    author  = {Jie Xu and Kangjin Yu and Ziyi Jin and Junjie Gao and Liqing Chen and Yixian Li and Shuai Tian and Zhongpu Xia},
    year    = {2026},
    journal = {arXiv preprint arXiv: 2609.09630}
}
```

```bibtex
@inproceedings{schwarzer2021dataefficient,
    title   = {Data-Efficient Reinforcement Learning with Self-Predictive Representations},
    author  = {Max Schwarzer and Ankesh Anand and Rishab Goel and R Devon Hjelm and Aaron Courville and Philip Bachman},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year    = {2021},
    url     = {https://openreview.net/forum?id=Bke9u1HFwB}
}
```

```bibtex
@misc{teoh2026nextlatent,
    title   = {Next-Latent Prediction Transformers Learn Compact World Models},
    author  = {Jayden Teoh and Manan Tomar and Kwangjun Ahn and Edward S. Hu and Tim Pearce and Pratyusha Sharma and Akshay Krishnamurthy and Riashat Islam and Alex Lamb and John Langford},
    year    = {2026},
    eprint  = {2511.05963},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.05963}
}
```
