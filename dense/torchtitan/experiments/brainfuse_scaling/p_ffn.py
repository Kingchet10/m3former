"""G0 full-width local P map between SwiGLU and the dense output projection.

Shape suffixes: T = any token dimensions, H = FFN width, G = groups,
B = local width, J = residual-MLP bottleneck, D = model dimension.
"""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward
from .g1_readout import compress_pairs

from .p_operator import (
    apply_compiled,
    apply_reference,
    constrain,
    gate,
    initial_d,
    initial_d_values,
    P_KINDS,
)


class PFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = "p_k1"
        group_size: int = 4
        alpha: float = 1.0
        local_compile: bool = True
        output_compression: int = 1

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import (
                    dense_param_placement,
                )

                names = (
                    ("local_w1", "local_w2")
                    if self.kind == "p_mlp"
                    else ()
                    if self.kind == "p_scale"
                    else ("d_raw",)
                )
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
        if config.kind not in P_KINDS or config.group_size != 4:
            raise ValueError("Frozen G0 FFNs require a registered P kind and b=4")
        if (
            config.w1.out_features % config.group_size
            or config.output_compression not in (1, 2)
            or config.w2.in_features * config.output_compression != config.w1.out_features
        ):
            raise ValueError(
                "P requires full width or paired G1 compression"
            )
        if not 0 <= config.alpha <= 1:
            raise ValueError("The verified P contract requires alpha in [0, 1]")
        self.kind, self.group_size = config.kind, config.group_size
        self.output_compression = config.output_compression
        self.alpha, self.local_compile = config.alpha, config.local_compile
        if self.kind == "p_mlp":
            self.local_w1 = nn.Parameter(torch.empty(2, 4))
            self.local_w2 = nn.Parameter(torch.empty(4, 2))
        elif self.kind != "p_scale":
            self.d_raw = nn.Parameter(
                torch.empty(4) if self.kind == "p_diag" else torch.empty(4, 4)
            )
        self.capture_diagnostics = False
        self.diagnostics = {}

    def reset_parameters(self):
        with torch.no_grad():
            if self.kind == "p_mlp":
                # Equal fixed local starts across layers; dense weights remain name-seeded.
                rows = ((0.5, 0.5, -0.5, -0.5), (0.5, -0.5, 0.5, -0.5))
                for j, row in enumerate(rows):
                    for i, value in enumerate(row):
                        self.local_w1[j, i].fill_(value)
                        self.local_w2[i, j].fill_(0.1 * value)
            elif self.kind != "p_scale":
                # Scalar fill works before and after DTensor parameter distribution.
                for i, row in enumerate(initial_d_values(4)):
                    if self.kind == "p_diag":
                        self.d_raw[i].fill_(row[i])
                    else:
                        for j, value in enumerate(row):
                            self.d_raw[i, j].fill_(value)

    def forward(self, x_TD):
        p_TH = F.silu(self.w1(x_TD)) * self.w3(x_TD)
        with torch.autocast(device_type=p_TH.device.type, enabled=False):
            dtype = torch.float64 if p_TH.dtype == torch.float64 else torch.float32
            u_TGB = p_TH.to(dtype).unflatten(-1, (-1, self.group_size))
            d_BB = constrain(self.d_raw.to(dtype)) if hasattr(self, "d_raw") else None
            w1_JB = self.local_w1.to(dtype) if self.kind == "p_mlp" else None
            w2_BJ = self.local_w2.to(dtype) if self.kind == "p_mlp" else None
            if self.local_compile:
                # Downstream likelihood batches have many sequence lengths.
                y_TGB = apply_compiled(
                    u_TGB,
                    d_BB,
                    self.kind,
                    self.alpha,
                    w1_JB,
                    w2_BJ,
                    dynamic=not self.training,
                )
            else:
                y_TGB = apply_reference(
                    u_TGB, d_BB, self.kind, self.alpha, w1_JB, w2_BJ
                )
        readout = compress_pairs(y_TGB) if self.output_compression == 2 else y_TGB
        output_TD = self.w2(readout.flatten(-2).to(p_TH.dtype))
        if self.capture_diagnostics:
            with torch.no_grad():
                tiny = torch.finfo(dtype).tiny
                self.diagnostics = {
                    "input_rms": u_TGB.square().mean().sqrt(),
                    "output_rms": y_TGB.square().mean().sqrt(),
                    "local_relative_change": (y_TGB - u_TGB).square().mean().sqrt()
                    / u_TGB.square().mean().sqrt().clamp_min(tiny),
                    "gate_mean": gate(u_TGB).mean(),
                    "gate_saturated_fraction": (gate(u_TGB) > 0.9).float().mean(),
                }
                if d_BB is not None:
                    self.diagnostics["d_spectral_norm"] = (
                        d_BB.abs().amax()
                        if d_BB.ndim == 1
                        else torch.linalg.matrix_norm(d_BB, ord=2)
                    )
                    self.diagnostics["d_distance_from_init"] = (
                        self.d_raw
                        - (
                            initial_d(4, device=d_BB.device).diag()
                            if self.kind == "p_diag"
                            else initial_d(4, device=d_BB.device)
                        )
                    ).norm()
        return output_TD
