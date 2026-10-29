"""Step 1 v1.1: additive temporal state and gate/window ablations.

The affine scan is the scalar scan from dynamic_memory.py in T-series, with
an ADDITIVE readout. Every chunk carries differentiable state; no truncated BPTT.
Legacy C1 spatial FFNs retain their original classes and exact execution path.
"""
from dataclasses import dataclass, fields, replace
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.common.attention import (
    get_causal_mask_mod, get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
)
from .local_functions import apply, compiler, initial_values, parameter_count


def affine_prefix(a, b, initial):
    stride = 1
    while stride < b.shape[1]:
        aa = a[:, :-stride] * a[:, stride:]
        bb = b[:, :-stride] * a[:, stride:] + b[:, stride:]
        a = torch.cat((a[:, :stride], aa), 1)
        b = torch.cat((b[:, :stride], bb), 1)
        stride *= 2
    return initial[:, None] * a + b


def ema_scan(h, rho, reset, chunk_size=128):
    """Differentiable FP32 reference; rho has [hidden, scales] shape."""
    state = h.new_zeros(h.shape[0], h.shape[-1], rho.shape[-1])
    result = []
    for start in range(0, h.shape[1], chunk_size):
        x = h[:, start:start + chunk_size, :, None]
        a = torch.where(reset[:, start:start + chunk_size, None, None], 0., rho)
        s = affine_prefix(a, (1-rho)*x, state)
        result.append(s)
        state = s[:, -1]
    return torch.cat(result, 1)


def initial_theta(parameter):
    # Scalar fill also supports TorchTitan's replicated DTensor parameters.
    k = parameter.shape[-1]
    taus = {1: (64.,), 4: (4., 16., 64., 256.)}.get(k)
    if taus is None:
        if k not in (9, 16):
            raise ValueError(f'Unsupported temporal scale count: {k}')
        taus = tuple(math.exp(math.log(2.) + i * math.log(1024.) / (k - 1)) for i in range(k))
    with torch.no_grad():
        for i, tau in enumerate(taus):
            # The bounded Step 1 sigmoid parameterization cannot represent its
            # exact endpoints with a finite theta. Approximate only the two
            # endpoints of the preregistered log-uniform T16/R4 grids.
            z = min(1. - 1e-6, max(1e-6,
                (math.log(tau) - math.log(2)) / math.log(1024)))
            parameter[:, i].fill_(math.log(z / (1-z)))


def temporal_readout(h, g, theta, positions, null=False, backend='auto'):
    with torch.autocast(device_type=h.device.type, enabled=False):
        x = h.float() if h.dtype != torch.float64 else h
        if null:
            # Keep theta as an identically inactive parameter, with zero VJP.
            return (x + x * g.to(x).sum(-1) + theta.to(x).sum()*0).to(h.dtype)
        rho = torch.exp(-torch.exp(-(math.log(2) + math.log(1024)*theta.to(x).sigmoid())))
        reset = positions == 0
        if backend == 'auto' and x.is_cuda and x.dtype == torch.float32:
            from .neuron_ema_triton import ema
            s = ema(x, rho, reset)
        else:
            s = ema_scan(x, rho, reset)
        return (x + (s * g.to(x)).sum(-1)).to(h.dtype)


def temporal_contribution(h, g, theta, positions, backend='auto'):
    """Only the R4 residual, so zero gain has an exact identity VJP."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        x = h.float() if h.dtype != torch.float64 else h
        rho = torch.exp(-torch.exp(-(math.log(2) + math.log(1024)*theta.to(x).sigmoid())))
        reset = positions == 0
        if backend == 'auto' and x.is_cuda and x.dtype == torch.float32:
            from .neuron_ema_triton import ema
            s = ema(x, rho, reset)
        else:
            s = ema_scan(x, rho, reset)
        return (s * g.to(x)).sum(-1).to(h.dtype)


class ComplexityFFN(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        activation: str = 'swiglu'
        integration: str = 'dense'
        group_size: int = 1
        temporal: str = ''
        local_compile: bool = True

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                names = ('g', 'theta') if self.temporal else (('local_theta',) if self.integration == 'safe_rational' else ())
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings,
                    **{n: dense_param_placement(tp=spmd.R) for n in names}})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        # Deliberately never allocate an unused w3 for GeLU.
        nn.Module.__init__(self)
        self.w1, self.w2 = config.w1.build(), config.w2.build()
        self.w3 = config.w3.build() if config.activation == 'swiglu' else None
        self.activation, self.integration = config.activation, config.integration
        self.group_size, self.temporal = config.group_size, config.temporal
        self.local_compile = config.local_compile
        h = config.w1.out_features
        self.local_theta = nn.Parameter(torch.empty(parameter_count('safe_rational', self.group_size))) if self.integration == 'safe_rational' else None
        k = {'t1': 1, 't4': 4, 't16': 16, 'null': 4}.get(self.temporal)
        self.g = nn.Parameter(torch.empty(h, k)) if k else None
        self.theta = nn.Parameter(torch.empty(h, k)) if k else None
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        with torch.no_grad():
            if self.local_theta is not None:
                for i, value in enumerate(initial_values('safe_rational', self.group_size)):
                    self.local_theta[i].fill_(value)
            if self.g is not None:
                self.g.zero_()
                initial_theta(self.theta)

    def forward(self, x, *, positions=None):
        h = F.gelu(self.w1(x), approximate='none') if self.activation == 'gelu' else F.silu(self.w1(x))*self.w3(x)
        if self.temporal:
            if positions is None:
                raise ValueError('Temporal FFN requires explicit document positions')
            y = temporal_readout(h, self.g, self.theta, positions, self.temporal == 'null')
            if self.capture_diagnostics:
                self.diagnostics = {'g_norm': self.g.detach().norm(), 'theta_norm': self.theta.detach().norm()}
                if self.temporal == 'null':
                    column_norms = self.w2.weight.detach().float().norm(dim=0)
                    self.diagnostics.update(
                        w2_column_norm_mean=column_norms.mean(),
                        w2_column_norm_p90=torch.quantile(column_norms, .9),
                        w2_column_norm_max=column_norms.max(),
                    )
        elif self.integration != 'dense':
            with torch.autocast(device_type=h.device.type, enabled=False):
                dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
                u = h.to(dtype).unflatten(-1, (-1, self.group_size))
                theta = self.local_theta.to(dtype) if self.local_theta is not None else u.new_empty(0)
                fn = compiler(dynamic=not self.training) if self.local_compile else apply
                y = fn(u, theta, self.integration).to(h.dtype)
        else:
            y = h
        return self.w2(y)


class TemporalBlock(Llama3TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        pass

    def forward(self, x, attention_masks, positions=None):
        h = x + self.attention(self.attention_norm(x), attention_masks, positions)
        return h + self.feed_forward(self.ffn_norm(h), positions=positions)


class RecurrentResidualBlock(TemporalBlock):
    """R4 state on the residual stream, immediately before the FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TemporalBlock.Config):
        r4_scales: int = 9
        r4_norm: object = None

    def __init__(self, config):
        super().__init__(config)
        self.r4_norm = config.r4_norm.build()
        d = config.r4_norm.normalized_shape
        self.r4_g = nn.Parameter(torch.empty(d, config.r4_scales))
        self.r4_theta = nn.Parameter(torch.empty(d, config.r4_scales))
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        with torch.no_grad():
            self.r4_g.zero_()
            initial_theta(self.r4_theta)

    def forward(self, x, attention_masks, positions=None):
        if positions is None:
            raise ValueError('R4 requires explicit document positions')
        h = x + self.attention(self.attention_norm(x), attention_masks, positions)
        u = self.r4_norm(h)
        h = h + temporal_contribution(u, self.r4_g, self.r4_theta, positions)
        if self.capture_diagnostics:
            self.diagnostics = {'g_norm': self.r4_g.detach().norm(),
                                'theta_norm': self.r4_theta.detach().norm()}
        return h + self.feed_forward(self.ffn_norm(h))


class WindowModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        window: int = 32

    def get_attention_masks(self, positions):
        return self._create_flex_attention_mask(positions, self.config.first_attention, [
            get_causal_mask_mod(), get_efficient_causal_mask_mod_for_packed_document(positions),
            get_sliding_window_mask_mod(self.config.window)])


def extend_config(config, a):
    if a.activation not in ('swiglu', 'gelu') or a.temporal not in ('', 't1', 't4', 't16', 'null') or a.attention_window not in (0, 16, 32, 128, 512):
        raise ValueError('Outside frozen Step 1 implementation')
    if a.temporal and a.ffn_kind != 'dense':
        raise ValueError('Temporal and branch interventions must not be combined')
    if a.activation == 'gelu' or a.temporal or a.r4_scales:
        for i, layer in enumerate(config.layers):
            f = layer.feed_forward
            layer.feed_forward = ComplexityFFN.Config(w1=f.w1, w2=f.w2, w3=f.w3,
                activation=a.activation, integration='dense' if a.ffn_kind == 'dense' else a.ffn_kind[3:],
                group_size=a.local_group_size, temporal=a.temporal, local_compile=a.operator_compile)
            if a.temporal or a.r4_scales:
                config.layers[i] = TemporalBlock.Config(**{v.name: getattr(layer, v.name) for v in fields(layer)})
            if a.r4_scales:
                from torchtitan.models.common import RMSNorm
                from torchtitan.models.llama3 import _NORM_INIT
                layer = config.layers[i]
                config.layers[i] = RecurrentResidualBlock.Config(
                    **{v.name: getattr(layer, v.name) for v in fields(layer)},
                    r4_scales=a.r4_scales,
                    r4_norm=RMSNorm.Config(normalized_shape=a.dim, param_init=_NORM_INIT),
                )
    if a.attention_window:
        config = WindowModel.Config(**{v.name: getattr(config, v.name) for v in fields(config)}, window=a.attention_window)
    return config
