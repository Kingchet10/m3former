# Mixture-of-experts models

Megatron Core model components with an explicit PyTorch DDP/AdamW training loop (`train.py`). Tensor, pipeline and
expert parallelism are 1: every rank holds all experts, and DDP synchronizes all parameters, including the mixers of
the residual connection. This is not the Megatron training engine, and TP/PP/EP cannot be enabled in this recipe.

| File | Content |
|---|---|
| `architecture.py` | backbone and MoE configuration (Megatron Core `GPTModel`) |
| `residuals.py` | residual connections: Transformer, MUDDFormer, M3 (`m3-g<G>`) and block attention residuals |
| `packed_experts.py` | packed experts with `torch._grouped_mm` and an FP32 weight-gradient Triton kernel |
| `depth_mix_kernel.py`, `fla_attnres_op.py` | fused mixing kernels (same as in `dense/m3former/`) |
| `data.py` | reader of the `megatron_document_pairs_v1` corpus (see `docs/DATA.md`) |
| `train.py`, `recipe.py` | training loop, validation, final weights; cosine learning-rate schedule |

Recipes (`configs/`):

| | Small (`configs/small/`) | Large (`configs/large/`) |
|---|---|---|
| Layers x width | 24 x 1024 | 28 x 1536 |
| Attention heads | 16 | 24 |
| Experts, top-k, expert width | 64, 8, 352 | 64, 8, 512 |
| Total / active parameters | 1.83B / 0.38B | 4.59B / 0.89B |
| Active non-embedding parameters | 0.31B | 0.80B |
| Training tokens | 6.2B | 15.9B |
| M3 groups G | 64 | 96 |
| Methods | transformer, muddformer, m3-g64, attnres-block | transformer, m3-g96 |

Both train on 20 tokens per active non-embedding parameter, as the dense models do per non-embedding parameter, and
use a global batch of 128 x 2048 tokens, a peak learning rate of 3e-4 with 1% warmup and cosine decay to 10%,
AdamW (0.9, 0.95, weight decay 0.1), BF16 compute with FP32 weights, gradients and Adam states, untied embeddings,
and a load-balancing loss of 0.01 with a router z-loss of 0.001. Launch with `scripts/launch.py moe --config ...`
on one node with 8 GPUs; the trainer saves the final weights only.

The variant names of the original runs (`baseline`, `mudd`, `dd64`, `dd96`, `attnres_block4`) are still accepted, so
their checkpoints load unchanged.
