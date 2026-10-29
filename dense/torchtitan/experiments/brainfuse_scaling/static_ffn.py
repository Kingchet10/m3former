# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stateless grouped SwiGLU: wide input features, small output projection."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward

from .static_readout import coefficient_pairs, grouped_readout


class StaticFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = "s0"
        branches: int = 2

        def build(self, **kwargs):
            # Llama's update_from_config installs the ordinary FFN placements
            # immediately before build. Extend them with our own scalar state.
            if self.kind != "s0" and self.sharding_config is not None:
                import spmd_types as spmd

                from torchtitan.models.common.decoder_sharding import (
                    dense_param_placement,
                )

                self.sharding_config = replace(
                    self.sharding_config,
                    state_shardings={
                        **self.sharding_config.state_shardings,
                        "coefficients": dense_param_placement(tp=spmd.R),
                    },
                )
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        self.kind = config.kind
        self.branches = config.branches
        if self.kind not in ("s0", "s1_d", "s1_q"):
            raise ValueError("Unknown static readout")
        if self.branches < 1 or config.w1.out_features % self.branches:
            raise ValueError("Hidden width must divide into complete branch groups")
        if config.w2.in_features != config.w1.out_features // self.branches:
            raise ValueError("Output projection must consume soma features")
        if self.kind == "s0":
            self.register_parameter("coefficients", None)
        else:
            count = len(coefficient_pairs(self.branches, self.kind == "s1_d"))
            self.coefficients = nn.Parameter(torch.empty(count))
        self.capture_diagnostics = False
        self.diagnostics = {}

    def reset_parameters(self):
        if self.coefficients is not None:
            nn.init.zeros_(self.coefficients)

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        grouped = p.unflatten(-1, (-1, self.branches)).float()
        coefficients = None if self.coefficients is None else self.coefficients.float()
        soma, linear, quadratic = grouped_readout(
            grouped, coefficients, diagonal=self.kind == "s1_d"
        )
        output = self.w2(soma.to(p.dtype))
        if self.capture_diagnostics:
            with torch.no_grad():
                linear_rms = linear.square().mean().sqrt()
                quadratic_rms = quadratic.square().mean().sqrt()
                self.diagnostics = {
                    "linear_rms": linear_rms.detach(),
                    "quadratic_rms": quadratic_rms.detach(),
                    "quadratic_to_linear_rms": (
                        quadratic_rms / linear_rms.clamp_min(1e-12)
                    ).detach(),
                    "output_rms": output.float().square().mean().sqrt().detach(),
                    "max_abs_branch": grouped.abs().max().detach(),
                }
        return output
