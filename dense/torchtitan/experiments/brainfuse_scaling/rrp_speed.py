"""Frozen speed-study shapes and equivalent FFN inference implementations.

Shape suffixes: T = tokens, H = input hidden width, O = output hidden width,
D = model dimension, G = groups, K = local coordinates.
"""

import copy
import types
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from .models import Architecture, model_config
from .rrp_ffn import ResidualRecyclingFeedForward
from .rrp_linear import project

DENSE_WIDTHS = (768, 800, 832, 853, 864)
TRAIN_CANDIDATES = ("a1", "a2", "dense768", "dense800", "dense832")


def speed_architecture(name):
    base = Architecture(initialization_seed=42, initialization_reference_width=1024)
    if name == "b1":
        name = "dense853"
    if name.startswith("dense"):
        width = int(name[5:])
        if width not in DENSE_WIDTHS:
            raise ValueError("Dense width outside the frozen speed grid")
        return replace(base, ff_dim=width)
    if name == "k3":
        return replace(
            base,
            ffn_kind="s3_k3",
            ff_dim=872,
            anchor_width=640,
            local_outputs=3,
            local_mode="shared",
        )
    if name not in ("rrp", "a1", "a2"):
        raise ValueError("Unknown speed-study model")
    anchor, dendritic = {"rrp": (669, 192), "a1": (576, 256), "a2": (544, 256)}[name]
    return replace(
        base,
        ffn_kind="rrp2_learn",
        ff_dim=anchor + dendritic,
        anchor_width=anchor,
        local_group_size=16,
        local_outputs=2,
        local_mode="learnable",
    )


def speed_recipe(name):
    from .config_registry import s_milestone_recipe

    a = speed_architecture(name)
    config = s_milestone_recipe(a.ffn_kind, hidden=a.ff_dim)
    config.architecture = a
    # Same effective batch/order, four microbatches fit shared 80GB devices.
    config.training.local_batch_size = 4
    return config


def build_speed_model(
    name, *, device="cpu", dtype=torch.float32, native_attention=False
):
    from torchtitan.models.common.attention import ScaledDotProductAttention

    config = model_config(speed_architecture(name))
    if not native_attention:
        for layer in config.layers:
            layer.attention.inner_attention = ScaledDotProductAttention.Config()
    with torch.device("cpu"):
        model = config.build()
        model.init_weights()
    # Module.to(dtype=...) would also cast complex RoPE buffers to real values.
    model.to(device=device)
    if dtype != torch.float32:
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype)
    return model


def elementwise_forward(self, x_TD):
    """Expose fixed-width local arithmetic to Inductor; retain parameter keys."""
    p_TH = F.silu(self.w1(x_TD)) * self.w3(x_TD)
    with torch.autocast(device_type=p_TH.device.type, enabled=False):
        dtype = torch.float64 if p_TH.dtype == torch.float64 else torch.float32
        dend_TGK = p_TH[..., self.anchor_width :].to(dtype).unflatten(-1, (-1, 4))
        u_KK, a_KK = self.u.to(dtype), self.a.to(dtype)
        coords = tuple(dend_TGK[..., j] for j in range(4))
        levels = tuple(sum(coords[j] * u_KK[i, j] for j in range(4)) for i in range(4))
        main_TGK = torch.stack(levels[:3], -1)
        residual_TGK = levels[3].unflatten(-1, (-1, 4))
        recycle_TGK = torch.stack(
            tuple(
                sum(residual_TGK[..., j] * a_KK[i, j] for j in range(4))
                for i in range(self.r)
            ),
            -1,
        )
        local_TO = torch.cat((main_TGK.flatten(-2), recycle_TGK.flatten(-2)), -1)
    combined_TO = torch.cat(
        (p_TH[..., : self.anchor_width], local_TO.to(p_TH.dtype)), -1
    )
    return self.w2(combined_TO)


class PreMergedFeedForward(nn.Module):
    """Inference-only dense representation of a trained static RRP readout."""

    def __init__(self, source):
        super().__init__()
        if not isinstance(source, ResidualRecyclingFeedForward):
            raise TypeError("Pre-merge expects an RRP FFN")
        self.w1, self.w3 = copy.deepcopy(source.w1), copy.deepcopy(source.w3)
        device, dtype = source.w2.weight.device, source.w2.weight.dtype
        width = source.w1.out_features
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            basis_HH = torch.eye(width, device=device, dtype=torch.float64)
            local_HO, _, _, _ = project(
                basis_HH[:, source.anchor_width :],
                None if source.mode == "flat" else source.u.double(),
                source.a.double(),
                source.mode,
            )
            compression_HO = torch.cat(
                (basis_HH[:, : source.anchor_width], local_HO), -1
            )
            weight_DH = source.w2.weight.double() @ compression_HO.T
        self.w2 = nn.Linear(
            width, source.w2.out_features, bias=False, device=device, dtype=dtype
        )
        self.w2.weight = nn.Parameter(weight_DH.to(dtype), requires_grad=False)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x_TD):
        if self.training:
            raise RuntimeError("Pre-merged FFN is inference only")
        return self.w2(F.silu(self.w1(x_TD)) * self.w3(x_TD))


def configure_implementation(model, implementation):
    if implementation not in (
        "eager",
        "compile",
        "elementwise",
        "elementwise_compile",
        "merged",
        "merged_compile",
    ):
        raise ValueError("Unknown speed implementation")
    for layer in model.layers.values():
        ff = layer.feed_forward
        if isinstance(ff, ResidualRecyclingFeedForward):
            if implementation.startswith("merged"):
                layer.feed_forward = PreMergedFeedForward(ff)
            elif implementation.startswith("elementwise"):
                ff.forward = types.MethodType(elementwise_forward, ff)
        if "compile" in implementation:
            layer.feed_forward = torch.compile(
                layer.feed_forward, fullgraph=True, dynamic=False
            )
    return model


class FixedLengthKVAttention(nn.Module):
    """Real KV cache, overwritten at one fixed decode position for timing.

    B = batch, L = cached length, N = heads, D = head dimension.
    Configure outside timing. Prefill populates the prefix once; repeated decode
    writes only its single slot. Position-dependent RoPE remains upstream.
    """

    def __init__(self):
        super().__init__()
        self.prefix = None
        self.decode = False

    def allocate(self, batch, prefix, heads, head_dim, device, dtype):
        self.prefix = prefix
        self.decode = False
        self.key_BLND = torch.empty(
            batch, prefix + 1, heads, head_dim, device=device, dtype=dtype
        )
        self.value_BLND = torch.empty_like(self.key_BLND)

    def forward(
        self, q_BLND, k_BLND, v_BLND, *, scale=None, enable_gqa=False, **kwargs
    ):
        if self.prefix is None:
            raise RuntimeError("Allocate the KV cache before inference")
        if self.decode:
            if q_BLND.shape[1] != 1:
                raise ValueError("This benchmark is one-token decode")
            self.key_BLND[:, self.prefix : self.prefix + 1].copy_(k_BLND)
            self.value_BLND[:, self.prefix : self.prefix + 1].copy_(v_BLND)
            key_BLND, value_BLND = self.key_BLND, self.value_BLND
        else:
            if k_BLND.shape[1] != self.prefix:
                raise ValueError("Prefill length differs from the configured cache")
            self.key_BLND[:, : self.prefix].copy_(k_BLND)
            self.value_BLND[:, : self.prefix].copy_(v_BLND)
            key_BLND, value_BLND = k_BLND, v_BLND
        out_BNLD = F.scaled_dot_product_attention(
            q_BLND.transpose(1, 2),
            key_BLND.transpose(1, 2),
            value_BLND.transpose(1, 2),
            dropout_p=0.0,
            is_causal=not self.decode,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out_BNLD.transpose(1, 2)


def install_kv_cache(model, batch, prefix, dtype):
    for layer in model.layers.values():
        attention = layer.attention
        cache = FixedLengthKVAttention()
        cache.allocate(
            batch,
            prefix,
            attention.n_kv_heads,
            attention.head_dim,
            next(model.parameters()).device,
            dtype,
        )
        attention.inner_attention = cache


def set_decode(model, enabled=True):
    for layer in model.layers.values():
        layer.attention.inner_attention.decode = enabled
