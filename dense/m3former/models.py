# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama3-style decoders built from upstream TorchTitan components.

The Transformer baseline is the unmodified TorchTitan Llama3 model. M3Former, MUDDFormer and multi-group mixing
replace its blocks with `m3.M3Block`; block attention residuals use `block_attn_residuals.BlockAttnResBlock`.
"""

from dataclasses import dataclass

from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.llama3 import (
    _EMBEDDING_SKIP_INIT,
    _NORM_INIT,
    _build_llama3_layers,
    _output_linear_init,
)
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec


@dataclass(kw_only=True)
class Architecture:
    dim: int = 640
    layers: int = 16
    heads: int = 10
    kv_heads: int = 10
    ff_dim: int = 1792
    vocab_size: int = 32000
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    initialization_seed: int | None = None
    initialization_reference_width: int | None = None
    depth_groups: int = 0  # > 0: M3 (G > 1), MUDDFormer (G = 1) or multi-group mixing with G groups per role
    depth_mode: str = "dense"  # see m3.MODES: 'dense' (four roles), 'dense1' (multi-group), 'static'
    attnres: bool = False  # attention residuals
    attnres_block_size: int = 0  # > 0: block attention residuals over blocks of this many sublayers
    attnres_heads: int = 1


# Fields of the original research code that the runs of the paper saved in config.json. They must have these
# values (other values selected architectures that are not part of this release); the rest have no effect here.
_LEGACY_NEUTRAL = dict(ffn_kind="dense", activation="swiglu", attention_window=0, temporal="", r4_scales=0,
                       attention_kind="softmax", dynamics="", dendrite_kind="", residual_modes=0, biophys="",
                       sparse_kind="", depth_hidden_mult=1, residual_mult=1, depth_lns=False, depth_mix_norm="",
                       hc_streams=0, dca=False, output_compression=1)
_LEGACY_IGNORED = ("dpr_shifts", "ridge_rank", "ridge_kind", "branches", "anchor_width", "local_group_size",
                   "local_outputs", "local_mode", "projection_rank", "projection_blocks", "operator_alpha",
                   "operator_compile", "cf_depth", "cf_schedule", "placement", "dendrite_branches", "fanin",
                   "mask_seed", "wiring")


def load_architecture(saved):
    """Architecture from the 'architecture' entry of a run's config.json, including runs of the original code."""
    saved = dict(saved)
    for key, value in _LEGACY_NEUTRAL.items():
        if key in saved and saved.pop(key) != value:
            raise ValueError(f"Unsupported architecture field {key}")
    for key in _LEGACY_IGNORED:
        saved.pop(key, None)
    return Architecture(**saved)


def model_config(a):
    if min(a.dim, a.layers, a.heads, a.kv_heads, a.ff_dim, a.vocab_size) <= 0:
        raise ValueError("Architecture dimensions must be positive")
    if a.dim % a.heads or a.heads % a.kv_heads or (a.dim // a.heads) % 2:
        raise ValueError("Invalid attention head dimensions")
    if a.depth_groups and a.attnres:
        raise ValueError("Choose either M3-style mixing or attention residuals")
    config = Llama3Model.Config(
        dim=a.dim,
        vocab_size=a.vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=a.vocab_size,
            embedding_dim=a.dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=a.dim,
            out_features=a.vocab_size,
            param_init=_output_linear_init(a.dim),
        ),
        norm=RMSNorm.Config(normalized_shape=a.dim, param_init=_NORM_INIT),
        layers=_build_llama3_layers(
            n_layers=a.layers,
            dim=a.dim,
            n_heads=a.heads,
            n_kv_heads=a.kv_heads,
            hidden_dim=a.ff_dim,
            fuse_qkv=True,
            attn_backend="flex",
            rope=ComplexRoPE.Config(
                dim=a.dim // a.heads,
                max_seq_len=a.max_seq_len,
                theta=a.rope_theta,
                scaling="none",
            ),
        ),
    )
    if a.depth_groups:
        from dataclasses import fields

        from .m3 import M3Block, M3Model

        if a.dim % a.depth_groups:
            raise ValueError("The number of groups must divide the model width")
        for i, layer in enumerate(config.layers):
            config.layers[i] = M3Block.Config(
                **{v.name: getattr(layer, v.name) for v in fields(layer)},
                layer_id=i,
                n_layers=a.layers,
                groups=a.depth_groups,
                dim=a.dim,
                input_norm=RMSNorm.Config(normalized_shape=a.dim, param_init=_NORM_INIT),
                mode=a.depth_mode,
            )
        config = M3Model.Config(
            **{v.name: getattr(config, v.name) for v in fields(config)},
            depth_groups=a.depth_groups,
        )
    elif a.attnres:
        from dataclasses import fields

        from .attn_residuals import AttnResBlock, AttnResModel

        if a.attnres_block_size:
            from .block_attn_residuals import BlockAttnResBlock as AttnResBlock
            from .block_attn_residuals import BlockAttnResModel as AttnResModel
        for i, layer in enumerate(config.layers):
            config.layers[i] = AttnResBlock.Config(
                **{v.name: getattr(layer, v.name) for v in fields(layer)},
                layer_id=i,
                n_layers=a.layers,
                dim=a.dim,
                heads=a.attnres_heads,
            )
        config = AttnResModel.Config(
            **{v.name: getattr(config, v.name) for v in fields(config)},
            **({"attnres_block_size": a.attnres_block_size} if a.attnres_block_size else {}),
        )
    if a.initialization_seed is not None:
        from .paired_init import configure_paired_initializers

        reference = a.initialization_reference_width or a.ff_dim
        if reference < 1:
            raise ValueError("Initialization reference width must be positive")
        configure_paired_initializers(config, a.initialization_seed, reference)
    return config


def model_spec(a):
    return ModelSpec(
        name="llama3",
        flavor="m3former_dense",
        model=model_config(a),
        parallelize_fn=parallelize_llama,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def ffn_projection_parameters(a):
    return 3 * a.dim * a.ff_dim  # SwiGLU: w1, w3 (d -> ff) and w2 (ff -> d)


def depth_parameters(a):
    """Parameters that the residual connection adds to the Transformer."""
    if a.attnres:
        from .attn_residuals import attnres_parameters

        return attnres_parameters(a.dim, a.layers)
    if not a.depth_groups:
        return 0
    from .m3 import m3_parameters

    return m3_parameters(a.dim, a.layers, a.depth_groups, a.depth_mode)


def parameter_count(a):
    return (a.vocab_size * a.dim + a.layers * (4 * a.dim * a.dim + 2 * a.dim) + a.dim
            + a.layers * ffn_projection_parameters(a) + depth_parameters(a))


def nonembedding(a):
    return parameter_count(a) - a.vocab_size * a.dim


def parameter_report(a, model, seq_len):
    total = sum(p.numel() for p in model.parameters())
    embedding = a.vocab_size * a.dim
    # Dense matmuls (including the shared output projection) and full causal attention;
    # multiply-add counts as two FLOPs. The mixing of the residual connection is not included.
    per_layer_linear = 2 * a.dim * (
        a.dim + 2 * a.kv_heads * (a.dim // a.heads) + a.dim
    ) + 2 * ffn_projection_parameters(a)
    linear_forward = a.layers * per_layer_linear + 2 * a.vocab_size * a.dim
    pairs = seq_len * (seq_len + 1) // 2
    attention_forward = a.layers * 4 * a.dim * pairs / seq_len
    return {
        "total_parameters": total,
        "embedding_parameters": embedding,
        "non_embedding_parameters": total - embedding,
        "embedding_fraction": embedding / total,
        "linear_forward_flops_per_input_token": linear_forward,
        "full_context_training_flops_per_input_token": 3 * (linear_forward + attention_forward),
        "flops_scope": (
            "forward+backward dense matmuls and ideal triangular attention; "
            "2 FLOPs/MAC; full-context upper bound per input token; "
            "per-step document-aware counts are reported separately; "
            "excludes softmax,norm,activation,optimizer,recompute and communication"
        ),
        "kv_cache_bytes_per_sequence_bf16": 2 * a.layers * seq_len * a.kv_heads * (a.dim // a.heads) * 2,
        "ffn_projection_parameters": a.layers * ffn_projection_parameters(a),
        "residual_connection_parameters": depth_parameters(a),
    }
