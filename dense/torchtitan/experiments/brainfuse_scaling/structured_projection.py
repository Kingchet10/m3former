# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the root LICENSE file.

"""Configurable structured linear layers for TP=CP=PP=1 research recipes."""

import math
from dataclasses import dataclass
from functools import partial

import spmd_types as spmd
import torch
from torch import nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .structured_ops import factor_shapes, project


def rademacher_(tensor):
    with torch.no_grad():
        return tensor.bernoulli_(0.5).mul_(2).sub_(1)


def factor_initializers(kind, inputs, rank, blocks, effective_std):
    """Match dense W2 entry variance in expectation and keep z variance ~1.

    Independent Rademacher BLAST couplings avoid the all-ones initial matrix
    (which would collapse BLAST to a global low-rank factorization).
    """
    right_fan = inputs if kind == "low_rank" else inputs // blocks
    paths = blocks * rank if kind == "blast" else rank
    result = {
        "right": partial(nn.init.normal_, std=right_fan**-0.5),
        "left": partial(nn.init.normal_, std=effective_std * math.sqrt(inputs / paths)),
    }
    if kind == "blast":
        result["coupling"] = rademacher_
    return result


class StructuredProjection(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        kind: str = "low_rank"
        rank: int = 140
        blocks: int = 4

        def build(self, **kwargs):
            shapes = factor_shapes(
                self.kind, self.in_features, self.out_features, self.rank, self.blocks
            )
            if self.sharding_config is not None:
                # Llama installs ordinary w2.weight placements during config
                # update. Replace them with replicated factor placements;
                # outer FSDP owns data sharding. Other mesh axes are rejected.
                self.sharding_config = ShardingConfig(
                    state_shardings={
                        n: dense_param_placement(tp=spmd.R) for n in shapes
                    }
                )
            return Module.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__()
        if config.bias:
            raise ValueError("Structured research projections are bias-free")
        self.kind, self.rank, self.blocks = config.kind, config.rank, config.blocks
        self.in_features, self.out_features = config.in_features, config.out_features
        for name, shape in factor_shapes(
            self.kind, self.in_features, self.out_features, self.rank, self.blocks
        ).items():
            self.register_parameter(name, nn.Parameter(torch.empty(shape)))
        self.capture_diagnostics = False
        self.diagnostics = {}

    def parallelize(self, parallel_dims):
        if parallel_dims.tp != 1 or parallel_dims.cp != 1 or parallel_dims.pp != 1:
            raise ValueError("Structured reference projections support DP only")
        return super().parallelize(parallel_dims)

    def forward(self, input):
        output = project(
            input,
            self.right,
            self.left,
            kind=self.kind,
            coupling=getattr(self, "coupling", None),
        )
        if self.capture_diagnostics:
            with torch.no_grad():
                self.diagnostics = {
                    "input_rms": input.float().square().mean().sqrt().detach(),
                    "output_rms": output.float().square().mean().sqrt().detach(),
                    **{
                        n + "_rms": p.detach().float().square().mean().sqrt()
                        for n, p in self.named_parameters(recurse=False)
                    },
                }
        return output
