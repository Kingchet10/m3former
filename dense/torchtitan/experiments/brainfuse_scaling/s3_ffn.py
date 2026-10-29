"""S3 anchor-preserving SwiGLU with purely linear local compression."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward

from .s3_linear import initial_a, local_linear


class S3FeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        anchor_width: int = 640
        local_outputs: int = 2
        local_mode: str = "shared"

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
        dend = config.w1.out_features - self.anchor_width
        if self.anchor_width < 1 or dend < 4 or dend % 4:
            raise ValueError("Positive anchor and complete groups required")
        self.groups = dend // 4
        if self.k not in (2, 3) or self.mode not in ("shared", "group_adaptive"):
            raise ValueError("Invalid S3 local projection")
        if config.w2.in_features != self.anchor_width + self.groups * self.k:
            raise ValueError("Incorrect output width")
        shape = (self.k, 4) if self.mode == "shared" else (self.groups, self.k, 4)
        self.a = nn.Parameter(torch.empty(shape))
        self.capture_diagnostics = False
        self.diagnostics = {}

    def reset_parameters(self):
        # Scalar in-place fills also work after replicated DTensor distribution.
        with torch.no_grad():
            self.a.zero_()
            self.a[..., 0, :2].fill_(2**-0.5)
            self.a[..., 1, 2:].fill_(2**-0.5)
            if self.k == 3:
                self.a[..., 2, 0::2].fill_(0.5)
                self.a[..., 2, 1::2].fill_(-0.5)

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            grouped = (
                p[..., self.anchor_width :].unflatten(-1, (self.groups, 4)).to(dtype)
            )
            local = local_linear(grouped, self.a.to(dtype))
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
                a = self.a.detach().float()
                a0 = initial_a(self.k, device=a.device, dtype=a.dtype)
                rms = lambda v: v.float().square().mean().sqrt()
                self.diagnostics = {
                    "a_norm": a.norm(),
                    "a_distance_from_init": (a - a0).norm(),
                    "anchor_rms": rms(p[..., : self.anchor_width]),
                    "dendritic_input_rms": rms(grouped),
                    "local_output_rms": rms(local),
                    "concat_rms": rms(combined),
                    "output_rms": rms(output),
                }
                if self.k == 3:
                    third = local[..., 2]
                    weight = (
                        self.w2.weight[:, self.anchor_width :]
                        .float()
                        .reshape(-1, self.groups, 3)[..., 2]
                    )
                    self.diagnostics.update(
                        third_output_rms=rms(third),
                        third_residual_rms=rms(F.linear(third.float(), weight)),
                        third_a_drift=(a[..., 2, :] - a0[2]).norm(),
                    )
        return output
