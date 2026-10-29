"""Two shared continued-fraction ladders per four SwiGLU features, G1 4->2."""
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward
from .cofr_operator import active_depth, apply, compiler, continuant


class ContinuedFractionFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        depth: int = 3
        safe: bool = False
        scheduled: bool = False
        local_compile: bool = True

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                placement = {name: dense_param_placement(tp=spmd.R)
                             for name in ("readout", *(f"cf_level_{i}" for i in range(self.depth)))}
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings, **placement})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        if config.depth not in (1, 3) or config.w1.out_features % 4 or config.w2.in_features * 2 != config.w1.out_features:
            raise ValueError("CF requires depth1/3 and four-to-two local groups")
        self.depth, self.safe, self.scheduled = config.depth, config.safe, config.scheduled
        self.local_compile = config.local_compile
        for i in range(self.depth):
            self.register_parameter(f"cf_level_{i}", nn.Parameter(torch.empty(2, 5)))
        self.readout = nn.Parameter(torch.empty(2, 2))
        self.active_depth = self.depth
        self.capture_diagnostics, self.diagnostics = False, {}

    @property
    def levels(self):
        # Plain ParameterList is rejected by upstream Module protocol validation.
        return tuple(getattr(self, f"cf_level_{i}") for i in range(self.depth))

    def reset_parameters(self):
        with torch.no_grad():
            for level in self.levels:
                for i, row in enumerate(((.1, -.1, .05, .05, 1.), (-.05, .05, .1, -.1, 1.))):
                    for j, value in enumerate(row):
                        level[i, j].fill_(value)
            for i in range(2):
                for j in range(2):
                    self.readout[i, j].fill_(float(i == j))

    def set_schedule(self, completed_steps, total_steps):
        self.active_depth = active_depth(completed_steps, total_steps, self.depth, self.scheduled)

    def forward(self, x):
        p = F.silu(self.w1(x)) * self.w3(x)
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            u = p.to(dtype).unflatten(-1, (-1, 4))
            # Frozen parameters remain in optimizer/checkpoint schema, but grad=None
            # prevents AdamW decay AND moment updates. Forward always uses full depth.
            levels = tuple((v if i < self.active_depth else v.detach()).to(dtype)
                           for i, v in enumerate(self.levels))
            fn = compiler(dynamic=not self.training) if self.local_compile else apply
            y = fn(u, levels, self.readout.to(dtype), self.safe)
            if self.capture_diagnostics:
                with torch.no_grad():
                    values = [sum(u[..., j:j+1]*v[:, j] for j in range(4))+v[:, 4] for v in levels]
                    _, denominator = continuant([1+a.square() for a in values] if self.safe else values)
                    self.diagnostics = dict(
                        active_depth=float(self.active_depth),
                        denominator_abs_min=denominator.abs().min(),
                        denominator_clamp_fraction=(denominator.abs()<.1).float().mean(),
                        local_output_rms=y.square().mean().sqrt(),
                        local_output_abs_max=y.abs().max(),
                    )
        return self.w2(y.flatten(-2).to(p.dtype))
