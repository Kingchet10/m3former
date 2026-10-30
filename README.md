# M3Former: Fine-Grained Cross-Layer Mixing for Transformers

Code for the paper *M3Former: Fine-Grained Cross-Layer Mixing for Transformers*.

A residual connection gives each Transformer block access to all earlier layers only through one fixed sum, and every
input of the block reads that same sum. **M3** (Multi-Role Multi-Group Mixing Residual) removes this bottleneck. The
query, key, value and residual inputs of a block (its four *roles*), and every *group* of channels within a role,
each read their own token-dependent mixture of all earlier hidden states. One small generator per block produces all
mixing coefficients from the current token, and a fused Triton kernel reads every earlier state once.

M3 starts as an exact residual connection: at initialization, M3Former computes exactly the standard Transformer. It
adds 1.4–3.9% parameters and 1.8–4.3% multiply–adds per token in the models below.

## Results

Validation loss at 20 training tokens per parameter (H200 runs; lower is better):

| Model | Transformer | Block AttnRes | MUDDFormer | **M3Former** | G of M3 |
|---|---|---|---|---|---|
| 100M | 2.7976 | 2.7507 | 2.7172 | **2.7126** | 10 |
| 300M | 2.4367 | 2.4168 | 2.3937 | **2.3825** | 16 |
| 500M | 2.3452 | 2.3447 | 2.3171 | **2.3039** | 18 |
| 700M | 2.2747 | 2.2692 | 2.2385 | **2.2309** | 80 |
| 1.4B | 2.1193 | 2.1053 | diverged | **2.0884** | 26 |
| MoE 1.8B-A0.38B | 2.2756 | 2.4089 | 2.2995 | **2.2654** | 64 |
| MoE 4.6B-A0.89B | 2.0881 | – | – | **2.0798** | 96 |

All methods share the recipe and the learning rate of each size. Downstream accuracy is in
[results/README.md](results/README.md).

## Repository

```text
dense/m3former/      M3Former and the baselines on TorchTitan: models, data, trainer, WSD schedule
  m3.py              M3 (four roles x G groups), MUDDFormer (G = 1) and multi-group mixing
  mix_kernel.py      fused Triton depth-mixing kernel (forward and backward)
  block_attn_residuals.py, attn_residuals.py, fla_attnres_op.py   block attention residuals baseline
  recipe.py          model sizes, methods and the training recipe of the paper
dense/torchtitan/    the subset of TorchTitan used for training (BSD-3-Clause)
dense/configs/       the dense runs of the paper
moe/                 mixture-of-experts models on Megatron Core with a PyTorch DDP/AdamW trainer
evaluation/          zero- and few-shot likelihood scoring of trained checkpoints
scripts/             data preparation, launcher, checkpoint export
third_party/         Megatron Core and the FLA attention-residual kernels used for training
results/             downstream results of the paper
tests/               launcher regression test and GPU checks of M3
```

### Methods and names

| Paper | `--method` | Code |
|---|---|---|
| Transformer | `transformer` | TorchTitan Llama3 model |
| MUDDFormer (multi-role mixing) | `muddformer` | `m3.M3Block`, mode `dense`, G = 1 |
| **M3Former** | `m3-g<G>` | `m3.M3Block`, mode `dense` |
| Multi-group mixing (ablation) | `multigroup-g<G>` | `m3.M3Block`, mode `dense1` |
| Block attention residuals | `attnres-block` | `block_attn_residuals.BlockAttnResBlock` (blocks of 4 layers) |

G is the number of groups per role. The paper counts it in attention heads: G = H, 2H, 4H for groups of width 64,
32, 16, where H = d/64 is 10, 16, 18, 20 and 26 from 100M to 1.4B.

## Setup

The dense and MoE models use different environments.

- **Dense:** Python 3.11, `torch==2.12.0.dev20260408+cu128`, `pip install -r dense/requirements.txt`.
- **MoE:** PyTorch 2.9.0, CUDA 12.8, Transformer Engine 2.11.0, FlashAttention 2.8.3, Triton 3.5.0 (`moe/requirements.txt`).

These are the versions of the runs in the paper. Newer versions are likely to work but have not been tested.

## Data

```bash
python scripts/prepare_data.py --output /path/to/corpus --tokenizer /path/to/tinyllama/tokenizer.model \
    --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-350BT --train-tokens 30e9
```

builds a FineWeb-Edu corpus tokenized with the 32K TinyLlama tokenizer, in the format the trainers read. See
[docs/DATA.md](docs/DATA.md).

## Training

A dense run on one node:

```bash
python scripts/launch.py dense --size 700M --method m3-g80 --budget 20N --gpus 4 \
    --data-root /path/to/corpus --runs-root /path/to/runs
```

The 20N run saves the checkpoints from which the 5N and 10N runs branch; launch them afterwards with the same
`--runs-root` and `--budget 5N` or `--budget 10N`. `python scripts/launch.py dense --paper` lists all dense runs of the
paper. `--dry-run` prints the command without starting it.

The recipe is AdamW (β = 0.9, 0.95, weight decay 0.1), a global batch of 64 × 2048 tokens, and a warmup–stable–decay
schedule: warmup over 1% of the 20N run, then a decay to 10% of the peak over the last 20% of each budget. The peak
learning rate is set per size and shared by all methods (`dense/m3former/recipe.py`). Training runs in BF16 with FP32
master weights and per-block `torch.compile`.

A MoE run (8 GPUs):

```bash
python scripts/launch.py moe --config moe/configs/small/m3-g64.json --data-root /path/to/corpus --runs-root /path/to/runs
```

The MoE recipes (`moe/configs/`) use a global batch of 128 × 2048 tokens and a cosine schedule; see
[moe/README.md](moe/README.md).

## Evaluation

```bash
python scripts/export_dense_checkpoint.py RUN/checkpoint/step-N RUN/checkpoint/step-N-model
```

exports the weights of a dense checkpoint. [evaluation/README.md](evaluation/README.md) describes the scoring of
the ten benchmarks.

## Tests

```bash
python tests/test_launch.py      # the launcher reproduces the 66 dense runs of the paper (CPU only)
python tests/test_m3_gpu.py      # M3 on one GPU: exact initialization, parameter count, fused kernel
```

## Notes

- 1.4B MUDDFormer diverged with the shared pre-norm recipe; its configuration is kept for completeness.
- The PrePostNorm variant of the paper (1.4B MUDDFormer and M3Former, Section 5.3) is not part of this release.
- `dense/configs/paper_runs.json` lists the main runs; the multi-group ablation runs use
  `--method multigroup-g<G>` with the same recipe.

## Citation

```bibtex
@article{han2026m3former,
  title   = {M3Former: Fine-Grained Cross-Layer Mixing for Transformers},
  author  = {Han, Jiasong and Chai, Yunkai and Zhu, Tong and Zhang, Hengyu and Fang, Wei and Wang, Jintao and Li, Guoqi and Cheng, Yu},
  year    = {2026}
}
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). Third-party code keeps its own
license: TorchTitan (BSD-3-Clause), Megatron Core and FLA; see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
