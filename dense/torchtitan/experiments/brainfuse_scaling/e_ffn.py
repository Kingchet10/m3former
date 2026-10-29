"""Full-width static E map after SwiGLU, before output projection."""
from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.feed_forward import FeedForward
from .e_operator import E_KINDS, apply, compiled, multiply, orthogonal
from .p_operator import constrain, initial_d_values
from .g1_readout import compress_pairs


class EFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = "e_k1"
        group_size: int = 4
        alpha: float = 1.0
        local_compile: bool = True
        output_compression: int = 1

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings, "d_raw": dense_param_placement(tp=spmd.R)})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        if config.kind not in E_KINDS or config.group_size != 4 or not 0 <= config.alpha <= 1:
            raise ValueError("E screening requires b=4, lambda in [0,1], registered kind")
        if (config.output_compression not in (1, 2) or config.w1.out_features % 4
                or config.w2.in_features * config.output_compression != config.w1.out_features):
            raise ValueError("E requires full width or paired G1 compression")
        self.output_compression = config.output_compression
        self.kind, self.alpha = config.kind, config.alpha
        self.local_compile = config.local_compile
        self.d_raw = nn.Parameter(torch.empty(4, 4))
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        with torch.no_grad():
            for i, row in enumerate(initial_d_values(4)):
                for j, value in enumerate(row):
                    self.d_raw[i, j].fill_(value)

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            u = p.to(dtype).unflatten(-1, (-1, 4))
            raw = self.d_raw.to(dtype)
            d = orthogonal(raw) if self.kind == "e_closed" else constrain(raw)
            y = (compiled(u, d, self.kind, self.alpha, dynamic=not self.training)
                 if self.local_compile else apply(u, d, self.kind, self.alpha))
        readout = compress_pairs(y) if self.output_compression == 2 else y
        output = self.w2(readout.flatten(-2).to(p.dtype))
        if self.capture_diagnostics:
            with torch.no_grad():
                z = multiply(y, d)
                penalty = z if self.kind == "e_quadratic" else z.clamp(-1, 1)
                residual = y - u + self.alpha * multiply(penalty, d, True)
                self.diagnostics = {
                    "input_rms": u.square().mean().sqrt(),
                    "output_rms": y.square().mean().sqrt(),
                    "local_relative_change": (y-u).norm()/u.norm().clamp_min(1e-20),
                    "huber_quadratic_fraction": (z.abs()<=1).float().mean(),
                    "relative_stationarity_residual": residual.norm()/u.norm().clamp_min(1e-20),
                    "max_vector_relative_residual": (residual.norm(dim=-1)/u.norm(dim=-1).clamp_min(1e-12)).max(),
                    "d_spectral_norm": torch.linalg.matrix_norm(d, ord=2),
                }
        return output
