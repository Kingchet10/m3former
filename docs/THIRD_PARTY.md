# Third-party source

- **TorchTitan / original Routed Residuals checkout**: PyTorch/Meta BSD-3-Clause headers are retained in `dense/torchtitan`. License: `licenses/TorchTitan-LICENSE`. Original research repository: https://github.com/superajgklasjglaks/routed-residuals . Export is a local experimental derivative, not an upstream release.
- **Megatron Core**: the actual training runtime is vendored in `third_party/Megatron-LM/megatron/core`; license is `third_party/Megatron-LM/LICENSE`. This snapshot contains local optional router extensions, explicitly disabled by our construction. No verified upstream Git commit was present in this copied runtime; file hashes, not a guessed release tag, identify it.
- **FLA**: AttnRes kernel plus reachable helpers from the training snapshot, MIT license in `licenses/FLA-LICENSE`. Existing source headers are retained. `fla/ops/__init__.py` and `fla/ops/utils/__init__.py` are narrowed to kernel-only namespaces; kernel bodies are unchanged.

Original third-party licenses apply to their files. No blanket relicensing of third-party source is intended.
