# Training and evaluation

## Environments

The dense and MoE models use different environments; these are the versions of the runs in the paper.

- **Dense:** Python 3.11 and `torch==2.12.0.dev20260408+cu128`, then `pip install -r dense/requirements.txt`.
- **MoE:** PyTorch 2.9.0, CUDA 12.8, Transformer Engine 2.11.0, FlashAttention 2.8.3, Triton 3.5.0
  (`moe/requirements.txt`).

Newer versions are likely to work but have not been tested.

## Methods

| Paper | `--method` | Code |
|---|---|---|
| Transformer | `transformer` | TorchTitan Llama3 model |
| MUDDFormer (multi-role mixing) | `muddformer` | `m3.M3Block`, mode `dense`, G = 1 |
| M3Former | `m3-g<G>` | `m3.M3Block`, mode `dense` |
| Multi-group mixing (ablation) | `multigroup-g<G>` | `m3.M3Block`, mode `dense1` |
| Block attention residuals | `attnres-block` | `block_attn_residuals.BlockAttnResBlock`, blocks of 4 layers |

G is the number of groups per role. The paper counts it in attention heads, G = H, 2H, 4H for groups of width 64,
32 and 16, where H = d/64 is 10, 16, 18, 20 and 26 from 100M to 1.4B.

## Dense models

```bash
python scripts/launch.py dense --size 700M --method m3-g80 --budget 20N --gpus 4 \
    --data-root /path/to/corpus --runs-root /path/to/runs
```

| Size | Layers x width | SwiGLU width | Peak learning rate |
|---|---|---|---|
| 100M | 16 x 640 | 1792 | 7.6e-4 |
| 300M | 24 x 1024 | 2816 | 4.55e-4 |
| 500M | 28 x 1152 | 3200 | 3.2e-4 |
| 700M | 32 x 1280 | 3520 | 2.6e-4 |
| 1.4B | 40 x 1664 | 4608 | 1.9e-4 |

All methods share the recipe and the learning rate of each size: AdamW (β = 0.9, 0.95, ε = 1e-8, weight decay 0.1),
gradient clipping at 1.0, sequences of 2,048 tokens, a global batch of 64 sequences, BF16 computation with FP32 master
weights, and per-block `torch.compile`. The schedule is warmup–stable–decay: linear warmup over 1% of the 20N run, then
a decay to 10% of the peak, 0.1 + 0.9 (1 - sqrt(p)), over the last 20% of each budget.

The mixing kernels are those of `dense/m3former/mix_kernel_v2.py` by default: they keep the hidden states of a
forward pass in one buffer, add every block's contribution to their gradients in place, and run the query, key and
value projections as one batched GEMM, which trains faster than the kernels of the runs of the paper
(`mix_kernel.py`, selected with `M3_MIX_KERNELS=v1`). Both compute the same mixtures in FP32; v2 rounds each
accumulated gradient of a hidden state to BF16 once per addition instead of twice. Activation checkpointing of M3
blocks is not supported with v2.

A model with N non-embedding parameters trains on 20N tokens. The 20N run saves complete checkpoints (model, Adam,
data loader, scheduler and RNG state) where the decay of the 5N and 10N budgets starts; the shorter runs restore them
and decay from there. Launch the 20N run first and the branches afterwards with the same `--runs-root` and
`--budget 5N` or `--budget 10N`; a branch checks that only the budget and schedule changed and that the restored
state matches the fingerprints saved with the checkpoint.

`dense/configs/paper_runs.json` lists the main runs of the paper and their GPU counts
(`python scripts/launch.py dense --paper`); `tests/test_launch.py` checks that the launcher reproduces their command
lines. The multi-group ablation uses `--method multigroup-g<G>` with the same recipe. At 1.4B, MUDDFormer diverged
under this recipe; the PrePostNorm variant of the paper (Section 5.3) is not part of this release.

## Mixture-of-experts models

```bash
python scripts/launch.py moe --config moe/configs/small/m3-g64.json --data-root /path/to/corpus --runs-root /path/to/runs
```

See [moe/README.md](../moe/README.md) for the architectures and the recipe (global batch 128 x 2048 tokens, cosine
schedule, 20 tokens per active non-embedding parameter).

## Evaluation

```bash
python scripts/export_dense_checkpoint.py RUN/checkpoint/step-N RUN/checkpoint/step-N-model
```

exports the weights of a dense checkpoint; MoE runs save their final weights directly.
[evaluation/README.md](../evaluation/README.md) describes the scoring of the ten benchmarks, and
[results/](../results/README.md) holds the downstream results of the paper.

## Tests

```bash
python tests/test_launch.py      # the launcher reproduces the 66 dense runs of the paper (CPU)
python tests/test_m3_gpu.py      # M3 on one GPU: exact initialization, parameter count, both kernel versions
```
