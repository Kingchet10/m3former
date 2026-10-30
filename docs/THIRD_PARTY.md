# Third-party code

| Code | Location | License |
|---|---|---|
| TorchTitan (subset used for training) | `dense/torchtitan/` | BSD-3-Clause, `licenses/TorchTitan-LICENSE` |
| Megatron Core (snapshot of the training runtime) | `third_party/Megatron-LM/` | `third_party/Megatron-LM/LICENSE` (copy in `licenses/Megatron-LM-LICENSE`) |
| FLA attention-residual kernels and their helpers | `third_party/fla-kernels/` | MIT, `licenses/FLA-LICENSE` |

The original copyright headers are kept in all third-party files, and their licenses apply to them. The files of
`dense/m3former/` that derive from TorchTitan keep its BSD-3-Clause header.

Changes to third-party code:

- **TorchTitan:** only the modules needed for training are included.
- **Megatron Core:** the snapshot contains optional router extensions of the original runtime. They are disabled by
  the MoE model construction (`moe/architecture.py`) and by `QWEN3_DEFAULT_MOE=0` and `QWEN3_EXPERT_DROPOUT_P=0`,
  which `scripts/launch.py` sets.
- **FLA:** only the attention-residual kernel and the helpers it needs are included. The eager imports of
  `fla/ops/__init__.py` and `fla/ops/utils/__init__.py` are narrowed to these modules; the kernels are unchanged.
