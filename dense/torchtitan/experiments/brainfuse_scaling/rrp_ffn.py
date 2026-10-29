"""Two-level fixed local geometry and an optional learned recycle matrix."""

from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.feed_forward import FeedForward
from .rrp_linear import a_rows, u_rows, project, output_width


class ResidualRecyclingFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        anchor_width: int = 669
        local_outputs: int = 2
        local_mode: str = "fixed"

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import (
                    dense_param_placement,
                )

                names = ("a",) if self.local_mode == "flat" else ("u", "a")
                self.sharding_config = replace(
                    self.sharding_config,
                    state_shardings={
                        **self.sharding_config.state_shardings,
                        **{n: dense_param_placement(tp=spmd.R) for n in names},
                    },
                )
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        self.anchor_width = config.anchor_width
        self.r, self.mode = config.local_outputs, config.local_mode
        self.dend = config.w1.out_features - self.anchor_width
        rows = a_rows(self.r, self.mode)
        if self.anchor_width < 1 or config.w2.in_features != output_width(
            self.anchor_width, self.dend, self.r
        ):
            raise ValueError("Invalid FFN widths")
        self.a = nn.Parameter(
            torch.empty(len(rows), len(rows[0])), requires_grad=self.mode == "learnable"
        )
        if self.mode != "flat":
            self.u = nn.Parameter(torch.empty(4, 4), requires_grad=False)
        self.capture_diagnostics = False
        self.diagnostics = {}

    def reset_parameters(self):
        with torch.no_grad():
            entries = [(self.a, a_rows(self.r, self.mode))]
            if self.mode != "flat":
                entries.append((self.u, u_rows()))
            for parameter, rows in entries:
                for i, row in enumerate(rows):
                    for j, value in enumerate(row):
                        parameter[i, j].fill_(value)

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            dend = p[..., self.anchor_width :].to(dtype)
            local, main, residual, recycle = project(
                dend,
                None if self.mode == "flat" else self.u.to(dtype),
                self.a.to(dtype),
                self.mode,
            )
        combined = torch.cat((p[..., : self.anchor_width], local.to(p.dtype)), -1)
        output = self.w2(combined)
        if self.capture_diagnostics:
            with (
                torch.no_grad(),
                torch.autocast(device_type=p.device.type, enabled=False),
            ):

                def rms(t):
                    return t.float().square().mean().sqrt()

                a0 = torch.tensor(a_rows(self.r, self.mode), device=p.device)
                self.diagnostics = {
                    "anchor_rms": rms(p[..., : self.anchor_width]),
                    "dendritic_input_rms": rms(dend),
                    "local_output_rms": rms(local),
                    "concat_rms": rms(combined),
                    "output_rms": rms(output),
                    "a_distance_from_init": (self.a.float() - a0).norm(),
                }
                if self.mode != "flat":
                    u = self.u.float()
                    u0 = torch.tensor(u_rows(), device=p.device)
                    energy = dend.square().sum(-1)
                    em = main.square().sum((-1, -2))
                    er = residual.square().sum(-1)
                    tiny = torch.finfo(torch.float32).tiny
                    ratio = er / energy.clamp_min(tiny)
                    # Distribution across physical token positions; padding convention unchanged.
                    self.diagnostics.update(
                        level1_main_rms=rms(main),
                        level1_residual_rms=rms(residual),
                        level2_recycle_rms=rms(recycle),
                        u_distance_from_init=(u - u0).norm(),
                        energy_reconstruction_relative_error=(
                            (energy - em - er).abs() / energy.clamp_min(tiny)
                        ).mean(),
                        residual_energy_fraction=er.sum()
                        / energy.sum().clamp_min(tiny),
                        residual_energy_fraction_token_mean=ratio.mean(),
                        residual_energy_fraction_token_std=ratio.std(unbiased=False),
                        residual_energy_fraction_token_min=ratio.min(),
                        residual_energy_fraction_token_max=ratio.max(),
                        recycled_energy_fraction=recycle.square().sum()
                        / er.sum().clamp_min(tiny),
                    )
                    split = self.anchor_width + self.dend // 4 * 3
                    w = self.w2.weight.float()
                    main_contrib = F.linear(
                        main.flatten(-2).float(), w[:, self.anchor_width : split]
                    )
                    rec_contrib = F.linear(recycle.flatten(-2).float(), w[:, split:])
                    # Use one FP32 linear calculation for the ratio numerator and denominator.
                    total = F.linear(combined.float(), w)
                    self.diagnostics.update(
                        main_contribution_rms=rms(main_contrib),
                        recycled_contribution_rms=rms(rec_contrib),
                        recycle_to_total_contribution_rms=rms(rec_contrib)
                        / rms(total).clamp_min(tiny),
                    )
                    wr = w[:, split:].reshape(w.shape[0], self.dend // 16, self.r)
                    for i in range(self.r):
                        v = recycle[..., i]
                        self.diagnostics[f"recycle_{i}_output_rms"] = rms(v)
                        self.diagnostics[f"recycle_{i}_contribution_rms"] = rms(
                            F.linear(v.float(), wr[..., i])
                        )
        return output
