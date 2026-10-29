# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small Llama3-style decoders built from upstream components."""

from dataclasses import dataclass
from math import prod

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

from .frontier_linear import RECIPES as FRONTIER_RECIPES
from .rrp_linear import RECIPES as RRP_RECIPES
from .p_operator import P_KINDS
from .e_operator import E_KINDS
from .cofr_operator import CF_KINDS
from .local_functions import LF_KINDS
from .s2_readout import KINDS as S2_KINDS
from .s2_readout import q_count
from .s3_linear import RECIPES as S3_RECIPES
from .structured_ops import KINDS, factor_shapes


@dataclass(kw_only=True)
class Architecture:
    dim: int = 384
    layers: int = 10
    heads: int = 6
    kv_heads: int = 6
    ff_dim: int = 1024
    vocab_size: int = 32768
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    ffn_kind: str = "dense"
    dpr_shifts: int = 0
    ridge_rank: int = 0
    ridge_kind: str = 'rational'
    branches: int = 2
    initialization_seed: int | None = None
    initialization_reference_width: int | None = None
    anchor_width: int = 640
    local_group_size: int = 4
    local_outputs: int = 2
    local_mode: str = "shared"
    projection_rank: int = 140
    projection_blocks: int = 4
    operator_alpha: float = 1.0
    operator_compile: bool = True
    output_compression: int = 1
    cf_depth: int = 3
    cf_schedule: bool = False
    activation: str = "swiglu"
    attention_window: int = 0
    temporal: str = ""
    r4_scales: int = 0
    attention_kind: str = "softmax"
    dynamics: str = ""
    placement: str = "gate"
    dendrite_kind: str = ""
    dendrite_branches: int = 4
    residual_modes: int = 0
    biophys: str = ""
    sparse_kind: str = ""
    fanin: float = 1.0
    mask_seed: int = 0
    wiring: str = "fixed"
    depth_groups: int = 0  # >0: depth dendrites (MUDD dense connections; groups = compartments per neuron type)
    depth_mode: str = "dense"  # 'dense' | 'dense1' (single-way) | ablations 'static', 'vemb'
    depth_hidden_mult: int = 1  # mixer hidden width multiplier (parameter-matched controls)
    residual_mult: int = 1  # residual stream width / dim (wide residual stream, depth-dendrite blocks only)
    depth_lns: bool = False  # LayerNorm Scaling (pre-norm outputs x 1/sqrt(layer)), depth-dendrite blocks only
    depth_mix_norm: str = ""  # "" | "l1": depth mixing weights normalised over sources
    hc_streams: int = 0  # Original dynamic Hyper-Connections
    attnres_block_size: int = 0
    attnres: bool = False  # Full Attention Residuals (Kimi 2026)
    attnres_heads: int = 1  # >1: Multi-Head Attention Residuals (MHAR) with this many routing heads
    dca: bool = False  # DeepCrossAttention (Heddes et al. 2025), our re-implementation


def architecture(flavor):
    if flavor == "30m":
        return Architecture()
    if flavor == "100m":
        return Architecture(dim=640, layers=16, heads=10, kv_heads=10, ff_dim=1792)
    if flavor == "10m":
        return Architecture(dim=256, layers=2, heads=4, kv_heads=4, ff_dim=704)
    raise ValueError(f"Unknown flavor: {flavor}")


def model_config(a):
    if a.ridge_rank and (a.ridge_rank not in (4,8,16) or a.ridge_kind not in ('rational','linear','silu') or a.ffn_kind!='lf_safe_rational' or a.dpr_shifts):
        raise ValueError('Invalid frozen RRM architecture')
    if a.dpr_shifts and (a.dpr_shifts not in (1,4) or a.ffn_kind!='lf_safe_rational' or a.ff_dim!=1024):
        raise ValueError('DPR is frozen to H1024 Safe-rational with 1/4 shifts')
    allowed_compression = (1,2,4,8) if a.ffn_kind == 'lf_safe_rational' else (1,2)
    if a.output_compression not in allowed_compression or (
        a.output_compression != 1 and a.ffn_kind not in (*E_KINDS, *P_KINDS, *CF_KINDS, *LF_KINDS)
    ):
        raise ValueError("Output compression is supported only by local P/E maps")
    if min(a.dim, a.layers, a.heads, a.kv_heads, a.ff_dim, a.vocab_size) <= 0:
        raise ValueError("Architecture dimensions must be positive")
    if a.dim % a.heads or a.heads % a.kv_heads or (a.dim // a.heads) % 2:
        raise ValueError("Invalid attention head dimensions")
    if a.ffn_kind not in (
        "dense",
        "s0",
        "s1_d",
        "s1_q",
        *KINDS,
        *S2_KINDS,
        *S3_RECIPES,
        *FRONTIER_RECIPES,
        *RRP_RECIPES,
        *P_KINDS,
        *E_KINDS,
        *CF_KINDS,
        *LF_KINDS,
    ):
        raise ValueError("Unknown FFN kind")
    if a.branches < 1 or (a.ffn_kind.startswith("s") and a.ff_dim % a.branches):
        raise ValueError("Invalid branch grouping")
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
    if a.ffn_kind in LF_KINDS:
        from .local_functions_ffn import LocalFunctionFeedForward
        from .local_functions import parameter_count
        parameter_count(a.ffn_kind[3:], a.local_group_size)
        outputs=2 if a.ffn_kind=='lf_safe_rational_4to2' else 1
        if outputs==2 and (a.local_outputs!=2 or a.dpr_shifts or a.ridge_rank):
            raise ValueError('Pure Safe-Rational4to2 requires two outputs, no DPR/RRM')
        if a.output_compression*outputs != a.local_group_size or a.ff_dim % a.local_group_size:
            raise ValueError('Local functions require compression B, H divisible by B')
        for layer_id,layer in enumerate(config.layers):
            original = layer.feed_forward
            original.w2.in_features = a.ff_dim // a.output_compression
            layer.feed_forward = LocalFunctionFeedForward.Config(
                w1=original.w1,w2=original.w2,w3=original.w3,
                kind=a.ffn_kind[3:],local_compile=a.operator_compile,dpr_shifts=a.dpr_shifts,group_size=a.local_group_size,
                ridge_rank=a.ridge_rank,ridge_kind=a.ridge_kind,ridge_seed=a.initialization_seed or 0,
                ridge_layer=layer_id,ridge_v_std=original.w1.param_init['weight'].keywords['std'],
                ridge_u_std=original.w2.param_init['weight'].keywords['std'])
    elif a.ffn_kind in CF_KINDS:
        from .cofr_ffn import ContinuedFractionFeedForward
        if a.output_compression != 2 or a.local_group_size != 4:
            raise ValueError("CF study uses G1 compression with B=4")
        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = a.ff_dim // 2
            layer.feed_forward = ContinuedFractionFeedForward.Config(
                w1=original.w1, w2=original.w2, w3=original.w3,
                depth=a.cf_depth, safe=a.ffn_kind == "local_safe_cf",
                scheduled=a.cf_schedule, local_compile=a.operator_compile,
            )
    elif a.ffn_kind in E_KINDS:
        from .e_ffn import EFeedForward
        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = a.ff_dim // a.output_compression
            layer.feed_forward = EFeedForward.Config(
                w1=original.w1, w2=original.w2, w3=original.w3,
                kind=a.ffn_kind, group_size=a.local_group_size,
                alpha=a.operator_alpha, local_compile=a.operator_compile,
                output_compression=a.output_compression,
            )
    elif a.ffn_kind in P_KINDS:
        from .p_ffn import PFeedForward

        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = a.ff_dim // a.output_compression
            layer.feed_forward = PFeedForward.Config(
                w1=original.w1,
                w2=original.w2,
                w3=original.w3,
                kind=a.ffn_kind,
                group_size=a.local_group_size,
                alpha=a.operator_alpha,
                local_compile=a.operator_compile,
                output_compression=a.output_compression,
            )
    elif a.ffn_kind in KINDS:
        from .structured_projection import StructuredProjection, factor_initializers

        factor_shapes(
            a.ffn_kind, a.ff_dim, a.dim, a.projection_rank, a.projection_blocks
        )
        for layer in config.layers:
            original = layer.feed_forward.w2
            # Preserve the upstream layer-dependent dense target variance.
            std = original.param_init["weight"].keywords["std"]
            layer.feed_forward.w2 = StructuredProjection.Config(
                in_features=a.ff_dim,
                out_features=a.dim,
                kind=a.ffn_kind,
                rank=a.projection_rank,
                blocks=a.projection_blocks,
                param_init=factor_initializers(
                    a.ffn_kind, a.ff_dim, a.projection_rank, a.projection_blocks, std
                ),
            )
    elif a.ffn_kind in RRP_RECIPES:
        from .rrp_ffn import ResidualRecyclingFeedForward
        from .rrp_linear import output_width

        if a.local_group_size != 16:
            raise ValueError("RRP uses consecutive supergroups of 16")
        width = output_width(a.anchor_width, a.ff_dim - a.anchor_width, a.local_outputs)
        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = width
            layer.feed_forward = ResidualRecyclingFeedForward.Config(
                w1=original.w1,
                w2=original.w2,
                w3=original.w3,
                anchor_width=a.anchor_width,
                local_outputs=a.local_outputs,
                local_mode=a.local_mode,
            )
    elif a.ffn_kind in FRONTIER_RECIPES:
        from .frontier_ffn import FrontierFeedForward

        dendritic = a.ff_dim - a.anchor_width
        if (
            a.anchor_width < 1
            or a.local_group_size < 2
            or dendritic < a.local_group_size
            or dendritic % a.local_group_size
        ):
            raise ValueError("Invalid frontier widths")
        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = (
                a.anchor_width + dendritic // a.local_group_size * a.local_outputs
            )
            layer.feed_forward = FrontierFeedForward.Config(
                w1=original.w1,
                w2=original.w2,
                w3=original.w3,
                anchor_width=a.anchor_width,
                local_outputs=a.local_outputs,
                local_group_size=a.local_group_size,
                local_mode=a.local_mode,
            )
    elif a.ffn_kind in S3_RECIPES:
        from .s3_ffn import S3FeedForward

        dendritic = a.ff_dim - a.anchor_width
        if a.anchor_width < 1 or dendritic < 4 or dendritic % 4:
            raise ValueError("Invalid S3 widths")
        if a.local_outputs not in (2, 3) or a.local_mode not in (
            "shared",
            "group_adaptive",
        ):
            raise ValueError("Invalid S3 local configuration")
        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = a.anchor_width + dendritic // 4 * a.local_outputs
            layer.feed_forward = S3FeedForward.Config(
                w1=original.w1,
                w2=original.w2,
                w3=original.w3,
                anchor_width=a.anchor_width,
                local_outputs=a.local_outputs,
                local_mode=a.local_mode,
            )
    elif a.ffn_kind in S2_KINDS:
        from .s2_ffn import S2FeedForward

        dendritic = a.ff_dim - a.anchor_width
        if a.anchor_width < 1 or dendritic < 4 or dendritic % 4:
            raise ValueError("Invalid S2 anchor/dendritic widths")
        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = a.anchor_width + dendritic // 2
            layer.feed_forward = S2FeedForward.Config(
                w1=original.w1,
                w2=original.w2,
                w3=original.w3,
                kind=a.ffn_kind,
                anchor_width=a.anchor_width,
            )
    elif a.ffn_kind != "dense":
        from .static_ffn import StaticFeedForward

        for layer in config.layers:
            original = layer.feed_forward
            original.w2.in_features = a.ff_dim // a.branches
            layer.feed_forward = StaticFeedForward.Config(
                w1=original.w1,
                w2=original.w2,
                w3=original.w3,
                kind=a.ffn_kind,
                branches=a.branches,
            )
    # Windows outside the frozen Step 1 set (0/16/32/128/512) take the Stage A path; for a plain baseline both
    # paths only wrap the model in WindowModel.
    if (a.dynamics or a.dendrite_kind or a.residual_modes or a.biophys or a.sparse_kind or a.depth_groups or a.attnres or a.hc_streams
            or a.dca or a.attention_window not in (0, 16, 32, 128, 512)):
        from .neuron_dynamics import extend_config_stagea
        config = extend_config_stagea(config, a)
    elif a.activation != "swiglu" or a.attention_window or a.temporal or a.r4_scales:
        from .neuron_complexity import extend_config
        config = extend_config(config, a)
    if a.attention_kind == 'gla':
        if a.attention_window or a.heads != a.kv_heads:
            raise ValueError('Step 2 GLA requires full range and equal Q/KV heads')
        from .neuron_gla import extend_gla_config
        config = extend_gla_config(config)
    elif a.attention_kind != 'softmax':
        raise ValueError(f'Unknown attention kind: {a.attention_kind}')
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
        flavor="brainfuse_dense",
        model=model_config(a),
        parallelize_fn=parallelize_llama,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def parameter_report(a, model, seq_len):
    total = sum(p.numel() for p in model.parameters())
    embedding = a.vocab_size * a.dim
    # Dense matmuls (including the shared output projection) and full causal
    # attention; forward + backward, multiply-add counts as two FLOPs.
    per_layer_linear = 2 * a.dim * (
        a.dim + 2 * a.kv_heads * (a.dim // a.heads) + a.dim
    ) + 2 * ffn_projection_parameters(a)
    linear_forward = a.layers * (per_layer_linear +
        (2*a.dim*a.dim if a.attention_kind == 'gla' else 0)) + 2 * a.vocab_size * a.dim
    window = min(a.attention_window or seq_len, seq_len)
    pairs = window * (window + 1) // 2 + (seq_len - window) * window
    attention_forward = (a.layers * 5 * a.heads * (a.dim // a.heads)**2
        if a.attention_kind == 'gla' else a.layers * 4 * a.dim * pairs / seq_len)
    report = {
        "total_parameters": total,
        "embedding_parameters": embedding,
        "non_embedding_parameters": total - embedding,
        "embedding_fraction": embedding / total,
        "linear_forward_flops_per_input_token": linear_forward,
        "full_context_training_flops_per_input_token": 3
        * (linear_forward + attention_forward),
        "flops_scope": (
            "forward+backward dense matmuls and ideal triangular attention; "
            "2 FLOPs/MAC; full-context upper bound per input token; "
            "per-step document-aware counts are reported separately; "
            "excludes softmax,norm,activation,optimizer,recompute and communication"
        ),
        "kv_cache_bytes_per_sequence_bf16": (0 if a.attention_kind == 'gla' else 2
        * a.layers
        * seq_len
        * a.kv_heads
        * (a.dim // a.heads)
        * 2),
        "recurrent_state_bytes": (a.layers*a.heads*(a.dim//a.heads)**2*4
                                  if a.attention_kind == 'gla' else 0),
        "ffn_kind": a.ffn_kind,
        "ffn_projection_parameters": a.layers * ffn_projection_parameters(a),
        "ffn_local_parameters": a.layers * local_parameters(a),
        "non_ffn_parameters": total
        - a.layers * (ffn_projection_parameters(a) + local_parameters(a)),
        "local_precision": "float32"
        if a.ffn_kind.startswith("s")
        or a.ffn_kind in (*FRONTIER_RECIPES, *RRP_RECIPES, *P_KINDS, *E_KINDS, *CF_KINDS, *LF_KINDS)
        else None,
        "step_flops_accounting": (
            "v2: dense matmuls (2/MAC, backward=2*forward), document-aware "
            "attention matmuls, direct SwiGLU basic arithmetic (11/feature), "
            "and direct local readout forward/VJP including shared coefficient "
            "reductions. SiLU exp/reciprocal counts are separate. Excludes "
            "common softmax, loss, norms, RoPE, optimizer, indexing, casts, "
            "autograd scatter temporaries, communication and monitoring. "
            "This is an explicitly scoped theoretical operation count, not "
            "measured GPU instructions or a complete hardware cost."
        ),
    }
    if a.temporal:
        k = {'t1': 1, 't4': 4, 't16': 16, 'null': 4}[a.temporal]
        report["recurrent_state_bytes"] += a.layers * a.ff_dim * k * 4
    if a.r4_scales:
        report["recurrent_state_bytes"] += a.layers * a.dim * a.r4_scales * 4
    report["activation"] = a.activation
    report["attention_window"] = a.attention_window
    report["attention_kind"] = a.attention_kind
    report["temporal"] = a.temporal
    if a.ffn_kind in LF_KINDS:
        report["local_accounting"] = "v4 nominal scalar forward operations, backward estimated at 2x forward; special functions count as one; excludes shared parameter constraints, comparisons, indexing, monitoring and compiler overhead. Not exact VJP FLOPs or measured instructions."
        report["step_flops_accounting"] = report["local_accounting"]
        report["local_sigma0"] = 1.0
        report["local_group_size"] = a.local_group_size
        report["local_parameter_sharing"] = "within each layer only"
        if a.ridge_rank:
            report['ridge_added_parameters']=a.layers*((a.ff_dim+a.ff_dim//a.local_group_size)*a.ridge_rank+1+(7 if a.ridge_kind=='rational' else 0))
            report['ridge_forward_macs_per_token']=a.layers*(a.ff_dim+a.ff_dim//a.local_group_size)*a.ridge_rank
            report['ridge_rank']=a.ridge_rank
            report['ridge_kind']=a.ridge_kind
        if a.dpr_shifts:
            report['controller_parameters'] = a.layers*a.dpr_shifts*(a.dim+1)
            report['controller_forward_macs_per_token'] = a.layers*a.dpr_shifts*a.dim
            report['interpolation_forward_flops_per_token'] = a.layers*(3*(a.ff_dim//2)+a.dpr_shifts)
            report['gather_reads_per_token'] = a.layers*a.ff_dim
            report['shift_special_functions_per_token'] = a.layers*a.dpr_shifts
            report['pairing'] = 'even/odd; positive delta reads i+delta; blockwise cyclic interpolation'
    if a.ffn_kind in CF_KINDS:
        report["local_accounting"] = "CF nominal scalar forward arithmetic; backward estimated at 2x forward. Includes per-microbatch shared z(0), excludes monitoring/optimizer/compiler overhead. Not an exact VJP/instruction count."
        report["step_flops_accounting"] = report["local_accounting"]
        report["cf_depth"] = a.cf_depth
        report["cf_schedule"] = a.cf_schedule
        report["cf_epsilon"] = 0.1
        report["cf_gamma"] = 0.1
    if a.ffn_kind in E_KINDS:
        report["local_accounting"] = (
            "E screening: nominal scalar forward estimate; backward estimated at "
            "2x forward (implicit Ref backward is not an unroll). Excludes shared "
            "spectral/Cayley/factorization work; included in wall time. Not exact FLOPs."
        )
        report["step_flops_accounting"] = report["local_accounting"]
        report["trainable_parameters"] = total
        report["frozen_parameters"] = 0
    if a.ffn_kind in P_KINDS:
        report["local_accounting"] = (
            "P v1: nominal scalar coordinate-program forward count, backward "
            "estimated at 2x forward; special functions count as one. Exact "
            "spectral-norm/SVD clipping is included in wall timings, not in "
            "nominal FLOPs. This is a scoped estimate, not a full exact VJP "
            "or GPU instruction count. See P_OPERATOR_DEFINITION.md."
        )
        report["step_flops_accounting"] = report["local_accounting"]
        report["trainable_parameters"] = total
        report["frozen_parameters"] = 0
    if a.ffn_kind in FRONTIER_RECIPES:
        from .frontier_linear import orthogonalization_estimate

        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        report["frozen_parameters"] = total - report["trainable_parameters"]
        report["orthogonalization_nominal_ops_per_microbatch"] = (
            a.layers
            * orthogonalization_estimate(
                a.local_group_size, a.local_outputs, a.local_mode
            )
        )
        report["orthogonalization_accounting"] = (
            "QR only: leading-order 6*B*K^2+4*K^3 per layer per microbatch; included in nominal FLOPs, not an exact library instruction count. Fixed A omits parameter VJP."
        )
    if a.ffn_kind in RRP_RECIPES:
        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        report["frozen_parameters"] = total - report["trainable_parameters"]
        report["local_accounting"] = (
            "Dense U4 and B contractions, including zero entries; forward and input VJP, plus learned B parameter VJP only. No QR."
        )
    if a.ffn_kind in KINDS:
        report["structured_projection"] = {
            "factor_shapes": factor_shapes(
                a.ffn_kind, a.ff_dim, a.dim, a.projection_rank, a.projection_blocks
            ),
            "coupling_parameters_per_layer": a.projection_blocks**2 * a.projection_rank
            if a.ffn_kind == "blast"
            else 0,
            "accounting": "All factor contractions are GEMMs: 2 FLOPs/MAC forward, 4 backward. BLAST coupling forward, input VJP and shared coefficient gradient are included, not additional local operations.",
            "initialization": "Independent factors; dense depth-scaled effective entry variance in expectation; right activation variance normalized; independent Rademacher BLAST couplings.",
            "padding_elements": 0,
        }
    return report


def ffn_projection_parameters(a):
    if a.sparse_kind:  # real synapses only: fan-in-limited w1 and w3, dense w2
        from .neuron_dynamics import sparse_fanin
        return (1 if a.sparse_kind == 'nsq' else 2) * a.ff_dim * sparse_fanin(a) + a.dim * a.ff_dim
    if a.dendrite_kind == 'full':
        return a.dim * a.ff_dim * (a.dendrite_branches + 2)
    if a.activation == "gelu":
        return a.dim * (a.ff_dim + a.ff_dim // a.output_compression)
    if a.ffn_kind in RRP_RECIPES:
        from .rrp_linear import output_width

        return a.dim * (
            2 * a.ff_dim
            + output_width(a.anchor_width, a.ff_dim - a.anchor_width, a.local_outputs)
        )
    if a.ffn_kind in FRONTIER_RECIPES:
        return a.dim * (
            2 * a.ff_dim
            + a.anchor_width
            + (a.ff_dim - a.anchor_width) // a.local_group_size * a.local_outputs
        )
    if a.ffn_kind in S3_RECIPES:
        return a.dim * (
            2 * a.ff_dim
            + a.anchor_width
            + (a.ff_dim - a.anchor_width) // 4 * a.local_outputs
        )
    if a.ffn_kind in KINDS:
        shapes = factor_shapes(
            a.ffn_kind, a.ff_dim, a.dim, a.projection_rank, a.projection_blocks
        )
        return 2 * a.dim * a.ff_dim + sum(prod(shape) for shape in shapes.values())
    if a.ffn_kind in S2_KINDS:
        return a.dim * (
            2 * a.ff_dim + a.anchor_width + (a.ff_dim - a.anchor_width) // 2
        )
    output_width = (
        a.ff_dim // a.output_compression
        if a.ffn_kind == "dense" or a.ffn_kind in (*P_KINDS, *E_KINDS, *CF_KINDS, *LF_KINDS)
        else a.ff_dim // a.branches
    )
    return a.dim * (2 * a.ff_dim + output_width)


def depth_parameters(a):
    if a.hc_streams:
        from .hyper_connections import hc_parameters
        return hc_parameters(a.dim, a.layers, a.hc_streams)
    if a.attnres:
        from .attn_residuals import attnres_parameters
        return attnres_parameters(a.dim, a.layers)
    if a.dca:
        from .dca import dca_parameters
        return dca_parameters(a.dim, a.layers)
    if not a.depth_groups:
        return 0
    from .depth_dendrites import depth_parameters as count
    return count(a.dim, a.layers, a.depth_groups, a.depth_mode, a.depth_hidden_mult, a.dim * a.residual_mult)


def local_parameters(a):
    if a.sparse_kind:
        from .neuron_dynamics import sparse_local_parameters
        return sparse_local_parameters(a)
    if a.biophys:
        from .neuron_dynamics import BIOPHYS
        return a.ff_dim * (4 + BIOPHYS[a.biophys])
    if a.residual_modes:
        from .neuron_dynamics import residual_local_parameters
        return residual_local_parameters(a)
    if a.dendrite_kind:
        from .neuron_dynamics import dendrite_local_parameters
        return dendrite_local_parameters(a)
    if a.dynamics:
        from .neuron_dynamics import local_parameter_count
        return local_parameter_count(a.dynamics, a.ff_dim)
    if a.temporal:
        return 2 * a.ff_dim * {'t1': 1, 't4': 4, 't16': 16, 'null': 4}[a.temporal]
    if a.r4_scales:
        return a.dim * (2 * a.r4_scales + 1)
    if a.ffn_kind in LF_KINDS:
        from .local_functions import parameter_count
        return parameter_count(a.ffn_kind[3:],a.local_group_size)+a.dpr_shifts*(a.dim+1)+(a.ridge_rank*(a.ff_dim+a.ff_dim//a.local_group_size)+1+(7 if a.ridge_kind=='rational' else 0) if a.ridge_rank else 0)
    if a.ffn_kind in CF_KINDS:
        return 10 * a.cf_depth + 4
    if a.ffn_kind in E_KINDS:
        return 16
    if a.ffn_kind in P_KINDS:
        return 0 if a.ffn_kind == "p_scale" else 4 if a.ffn_kind == "p_diag" else 16
    if a.ffn_kind in RRP_RECIPES:
        from .rrp_linear import local_numel

        return local_numel(a.local_outputs, a.local_mode)
    if a.ffn_kind in FRONTIER_RECIPES:
        return a.local_group_size * a.local_outputs
    if a.ffn_kind in S3_RECIPES:
        return (
            4
            * a.local_outputs
            * (
                (a.ff_dim - a.anchor_width) // 4
                if a.local_mode == "group_adaptive"
                else 1
            )
        )
    if a.ffn_kind in S2_KINDS:
        return 8 + q_count(a.ffn_kind)
    if a.ffn_kind == "s1_d":
        return a.branches
    if a.ffn_kind == "s1_q":
        return a.branches * (a.branches + 1) // 2
    return 0


def local_training_operations(a, physical_tokens, microbatches=1):
    if a.dynamics:
        from .neuron_dynamics import local_operations
        return local_operations(a.dynamics, a.layers, a.ff_dim, physical_tokens)
    if a.temporal:
        # EMA: 3 recurrence + 2 readout operations per state, backward 2x.
        # Null: scale reduction counted separately per microbatch.
        k = {'t1': 1, 't4': 4, 't16': 16, 'null': 4}[a.temporal]
        return 3*a.layers*(physical_tokens*a.ff_dim*(5*k if a.temporal != "null" else 2) + microbatches*a.ff_dim*(6*k if a.temporal != "null" else 3))
    if a.r4_scales:
        return 3*a.layers*(physical_tokens*a.dim*5*a.r4_scales + microbatches*a.dim*6*a.r4_scales)
    from .static_readout import readout_operations

    if a.ffn_kind in LF_KINDS:
        from .local_functions import FORWARD_OPS
        n = a.ff_dim//a.local_group_size
        ops = FORWARD_OPS[a.ffn_kind[3:]] + (8*(a.local_group_size-2) if a.ffn_kind == 'lf_safe_rational' else 0)
        return 3*a.layers*physical_tokens*(n*ops+
            (2*a.dim*a.dpr_shifts+3*(a.ff_dim//2)+5*a.dpr_shifts if a.dpr_shifts else 0)+
            (2*(a.ff_dim+n)*a.ridge_rank+2*n+(23 if a.ridge_kind=='rational' else 4 if a.ridge_kind=='silu' else 0)*a.ridge_rank if a.ridge_rank else 0))

    if a.ffn_kind in CF_KINDS:
        from .cofr_operator import forward_operations
        count = forward_operations(a.cf_depth, a.ffn_kind == "local_safe_cf")
        return 3 * a.layers * (physical_tokens * (a.ff_dim // 4) + microbatches) * count

    pooling = (a.layers * physical_tokens * (a.ff_dim // 2) * 3
               if a.output_compression == 2 else 0)

    if a.ffn_kind in E_KINDS:
        from .e_operator import forward_operations
        return pooling + 3 * a.layers * physical_tokens * (a.ff_dim // 4) * forward_operations(a.ffn_kind)
    if a.ffn_kind == "dense" or a.ffn_kind in KINDS:
        return 0
    if a.ffn_kind in P_KINDS:
        from .p_operator import local_forward_operations

        return pooling + (
            3
            * a.layers
            * physical_tokens
            * (a.ff_dim // 4)
            * local_forward_operations(a.ffn_kind)
        )
    if physical_tokens % microbatches:
        raise ValueError("Physical tokens must divide over equal microbatches")
    if a.ffn_kind in RRP_RECIPES:
        from .rrp_linear import training_operations

        return (
            a.layers
            * microbatches
            * training_operations(
                physical_tokens // microbatches,
                (a.ff_dim - a.anchor_width) // 16,
                a.local_outputs,
                a.local_mode,
            )
        )
    if a.ffn_kind in FRONTIER_RECIPES:
        from .frontier_linear import training_operations

        return (
            a.layers
            * microbatches
            * training_operations(
                physical_tokens // microbatches,
                (a.ff_dim - a.anchor_width) // a.local_group_size,
                a.local_group_size,
                a.local_outputs,
                a.local_mode,
            )
        )
    if a.ffn_kind in S3_RECIPES:
        from .s3_linear import training_operations

        return (
            a.layers
            * microbatches
            * training_operations(
                physical_tokens // microbatches,
                (a.ff_dim - a.anchor_width) // 4,
                a.local_outputs,
                a.local_mode,
            )
        )
    if a.ffn_kind in S2_KINDS:
        from .s2_readout import training_operations

        groups = physical_tokens // microbatches * ((a.ff_dim - a.anchor_width) // 4)
        return a.layers * microbatches * training_operations(groups, a.ffn_kind)
    groups = physical_tokens // microbatches * (a.ff_dim // a.branches)
    counts = readout_operations(groups, a.branches, a.ffn_kind)
    return a.layers * microbatches * sum(counts.values())
