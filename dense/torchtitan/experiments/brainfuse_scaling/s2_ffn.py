"""Partial dendritic FFN with an uncompressed anchor path."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward

from .s2_readout import initial_a, q_count, readout, symmetric_q


class S2FeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = "s2_l"
        anchor_width: int = 640

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd

                from torchtitan.models.common.decoder_sharding import (
                    dense_param_placement,
                )

                names = ["a"] + (["coefficients"] if q_count(self.kind) else [])
                self.sharding_config = replace(
                    self.sharding_config,
                    state_shardings={
                        **self.sharding_config.state_shardings,
                        **{name: dense_param_placement(tp=spmd.R) for name in names},
                    },
                )
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        self.kind, self.anchor_width = config.kind, config.anchor_width
        dendritic = config.w1.out_features - self.anchor_width
        if self.anchor_width < 1 or dendritic < 4 or dendritic % 4:
            raise ValueError(
                "S2 requires positive anchor and complete four-branch groups"
            )
        if config.w2.in_features != self.anchor_width + dendritic // 2:
            raise ValueError("Wrong S2 output projection width")
        self.a = nn.Parameter(torch.empty(2, 4))
        self.coefficients = (
            nn.Parameter(torch.empty(q_count(self.kind)))
            if q_count(self.kind)
            else None
        )
        self.capture_diagnostics = False
        self.diagnostics = {}

    def reset_parameters(self):
        with torch.no_grad():
            self.a.zero_()
            self.a[0, :2].fill_(2**-0.5)
            self.a[1, 2:].fill_(2**-0.5)
            if self.coefficients is not None:
                self.coefficients.zero_()

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            grouped = p[..., self.anchor_width :].unflatten(-1, (-1, 4)).to(dtype)
            q = None if self.coefficients is None else self.coefficients.to(dtype)
            soma, linear, quadratic = readout(
                grouped, self.a.to(dtype), q, blocked=self.kind == "s2_blocked"
            )
        combined = torch.cat(
            [p[..., : self.anchor_width], soma.flatten(-2).to(p.dtype)], dim=-1
        )
        output = self.w2(combined)
        if self.capture_diagnostics:
            with (
                torch.no_grad(),
                torch.autocast(device_type=p.device.type, enabled=False),
            ):
                lrms = linear.square().mean().sqrt()
                qrms = quadratic.square().mean().sqrt()
                # Actual cross-pair quadratic contribution, not merely ||Q||.
                cross = torch.zeros_like(quadratic)
                qnorm = qrms.new_zeros(())
                cross_norm = qrms.new_zeros(())
                internal_norm = qrms.new_zeros(())
                if q is not None:
                    matrix = symmetric_q(q, blocked=self.kind == "s2_blocked")
                    qnorm = matrix.norm()
                    cross_norm = (2 * matrix[:, :2, 2:].square().sum()).sqrt()
                    internal_norm = (
                        matrix[:, :2, :2].square().sum()
                        + matrix[:, 2:, 2:].square().sum()
                    ).sqrt()
                    cross = 2 * torch.einsum(
                        "...gi,kij,...gj->...gk",
                        grouped[..., :2],
                        matrix[:, :2, 2:],
                        grouped[..., 2:],
                    )
                contribution = F.linear(
                    quadratic.flatten(-2),
                    self.w2.weight[:, self.anchor_width :].to(dtype),
                )
                self.diagnostics = {
                    "a_norm": self.a.norm(),
                    "anchor_rms": p[..., : self.anchor_width]
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "dendritic_input_rms": grouped.square().mean().sqrt(),
                    "dendritic_output_rms": soma.square().mean().sqrt(),
                    "concat_rms": combined.float().square().mean().sqrt(),
                    "cross_pair_q_norm": cross_norm,
                    "internal_q_norm": internal_norm,
                    "cross_to_internal_q_norm": cross_norm
                    / internal_norm.clamp_min(1e-12),
                    "linear_rms": lrms,
                    "quadratic_rms": qrms,
                    "quadratic_to_linear_rms": qrms / lrms.clamp_min(1e-12),
                    "cross_pair_quadratic_rms": cross.square().mean().sqrt(),
                    "quadratic_output_rms": contribution.square().mean().sqrt(),
                    "symmetric_q_norm": qnorm,
                    "a_distance_from_init": (
                        self.a - initial_a(device=self.a.device, dtype=self.a.dtype)
                    ).norm(),
                    "output_rms": output.float().square().mean().sqrt(),
                    "max_abs_branch": grouped.abs().max(),
                }
        return output
