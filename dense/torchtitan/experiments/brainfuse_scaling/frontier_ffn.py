"""Frontier shared linear projection with general B/K and constrained A."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward

from .frontier_linear import effective_a, initial_a, initial_rows, local_linear


class FrontierFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        anchor_width: int = 640
        local_outputs: int = 2
        local_mode: str = "learnable"
        local_group_size: int = 4

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd

                from torchtitan.models.common.decoder_sharding import (
                    dense_param_placement,
                )

                self.sharding_config = replace(
                    self.sharding_config,
                    state_shardings={
                        **self.sharding_config.state_shardings,
                        "a": dense_param_placement(tp=spmd.R),
                    },
                )
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        self.anchor_width = config.anchor_width
        self.k, self.mode = config.local_outputs, config.local_mode
        self.b = config.local_group_size
        dend = config.w1.out_features - self.anchor_width
        if self.anchor_width < 1 or dend < self.b or dend % self.b:
            raise ValueError("Positive anchor and complete groups required")
        self.groups = dend // self.b
        if not 1 <= self.k < self.b or self.mode not in (
            "learnable",
            "orthogonal",
            "fixed",
        ):
            raise ValueError("Invalid frontier local projection")
        if config.w2.in_features != self.anchor_width + self.groups * self.k:
            raise ValueError("Incorrect output width")
        shape = (self.k, self.b)
        self.a = nn.Parameter(torch.empty(shape), requires_grad=self.mode != "fixed")
        self.capture_diagnostics = False
        self.diagnostics = {}

    def reset_parameters(self):
        # Scalar in-place fills also work after replicated DTensor distribution.
        with torch.no_grad():
            for i, row in enumerate(initial_rows(self.b, self.k)):
                for j, value in enumerate(row):
                    self.a[i, j].fill_(value)

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            grouped = (
                p[..., self.anchor_width :]
                .unflatten(-1, (self.groups, self.b))
                .to(dtype)
            )
            local = local_linear(grouped, self.a.to(dtype), self.mode)
        combined = torch.cat(
            (p[..., : self.anchor_width], local.flatten(-2).to(p.dtype)), -1
        )
        output = self.w2(combined)
        if self.capture_diagnostics:
            with (
                torch.no_grad(),
                torch.autocast(device_type=p.device.type, enabled=False),
            ):
                # Rank/SVD/principal angles are evaluated on CPU checkpoints, not
                # every training step. These activation diagnostics are batch-local.
                raw = self.a.detach().float()
                a = effective_a(raw, self.mode)
                a0 = initial_a(self.b, self.k, device=a.device, dtype=a.dtype)
                rms = lambda v: v.float().square().mean().sqrt()
                self.diagnostics = {
                    "a_norm": a.norm(),
                    "orthogonality_error": (
                        a @ a.T - torch.eye(self.k, device=a.device)
                    ).norm(),
                    "raw_a_norm": raw.norm(),
                    "raw_min_singular_value": torch.linalg.svdvals(raw).min(),
                    "a_distance_from_init": (a - a0).norm(),
                    "anchor_rms": rms(p[..., : self.anchor_width]),
                    "dendritic_input_rms": rms(grouped),
                    "local_output_rms": rms(local),
                    "concat_rms": rms(combined),
                    "output_rms": rms(output),
                }
                weight = (
                    self.w2.weight[:, self.anchor_width :]
                    .float()
                    .reshape(-1, self.groups, self.k)
                )
                for i in range(self.k):
                    value = local[..., i]
                    self.diagnostics[f"direction_{i}_output_rms"] = rms(value)
                    self.diagnostics[f"direction_{i}_residual_rms"] = rms(
                        F.linear(value.float(), weight[..., i])
                    )
        return output
