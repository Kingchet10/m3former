"""Dynamic HC, ICLR 2025 paper Algorithms 1--3, adapted to this Llama backbone.

Input expansion and output sum follow Algorithm 1. Routing LayerNorm uses
PyTorch defaults (affine, bias, eps=1e-5); these details are not specified by
the paper's LayerNorm(dim) pseudocode. Backbone paired initialization is kept;
the OLMo-specific output-projection initialization adjustment is not applied.
Static alpha/beta must be excluded from weight decay by the recipe.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from .depth_dendrites import _local


def hc_parameters(dim, layers, streams):
    return 2 * layers * (dim * (streams + 4) + streams * (streams + 2) + 2)


class HyperConnection(nn.Module):
    def __init__(self, dim, streams, layer_id):
        super().__init__()
        self.dim, self.streams, self.layer_id = dim, streams, layer_id
        self.static_alpha = nn.Parameter(torch.empty(streams, streams + 1))
        self.static_beta = nn.Parameter(torch.empty(streams))
        self.dynamic_alpha_fn = nn.Parameter(torch.empty(dim, streams + 1))
        self.dynamic_beta_fn = nn.Parameter(torch.empty(dim))
        self.dynamic_alpha_scale = nn.Parameter(torch.empty(1))
        self.dynamic_beta_scale = nn.Parameter(torch.empty(1))
        self.norm_weight = nn.Parameter(torch.empty(dim))
        self.norm_bias = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            a = _local(self.static_alpha)
            a.zero_()
            a[self.layer_id % self.streams, 0] = 1
            a[:, 1:].copy_(torch.eye(self.streams, device=a.device, dtype=a.dtype))
            _local(self.static_beta).fill_(1)
            _local(self.dynamic_alpha_fn).zero_()
            _local(self.dynamic_beta_fn).zero_()
            _local(self.dynamic_alpha_scale).fill_(0.01)
            _local(self.dynamic_beta_scale).fill_(0.01)
            _local(self.norm_weight).fill_(1)
            _local(self.norm_bias).zero_()

    def width_connection(self, h):
        # Float32 LayerNorm statistics, matching torch LayerNorm under mixed precision.
        z = F.layer_norm(h.float(), (self.dim,), _local(self.norm_weight).float(),
                         _local(self.norm_bias).float(), 1e-5).to(h.dtype)
        alpha = ((z @ _local(self.dynamic_alpha_fn).to(h.dtype)).tanh()
                 * _local(self.dynamic_alpha_scale).to(h.dtype)
                 + _local(self.static_alpha).to(h.dtype))
        beta = ((z @ _local(self.dynamic_beta_fn).to(h.dtype)).tanh()
                * _local(self.dynamic_beta_scale).to(h.dtype)
                + _local(self.static_beta).to(h.dtype))
        return alpha.transpose(-1, -2) @ h, beta

    @staticmethod
    def depth_connection(mixed, output, beta):
        return mixed[..., 1:, :] + beta.unsqueeze(-1) * output.unsqueeze(-2)


class HCBlock(Llama3TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        layer_id: int = 0
        dim: int = 0
        streams: int = 4

    def __init__(self, config):
        super().__init__(config)
        self.hc_attn = HyperConnection(config.dim, config.streams, 2 * config.layer_id)
        self.hc_ffn = HyperConnection(config.dim, config.streams, 2 * config.layer_id + 1)

    def reset_parameters(self):
        # Titan skips initialization of plain nn.Module children after to_empty.
        self.hc_attn.reset_parameters()
        self.hc_ffn.reset_parameters()

    def forward(self, h, attention_masks, positions=None):
        mixed, beta = self.hc_attn.width_connection(h)
        a = self.attention(self.attention_norm(mixed[..., 0, :]), attention_masks, positions)
        h = self.hc_attn.depth_connection(mixed, a, beta)
        mixed, beta = self.hc_ffn.width_connection(h)
        f = self.feed_forward(self.ffn_norm(mixed[..., 0, :]))
        return self.hc_ffn.depth_connection(mixed, f, beta)


class HCModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        streams: int = 4

    def __init__(self, config):
        super().__init__(config)
        self.streams = config.streams

    def forward(self, tokens, positions=None, attention_masks=None):
        h = self.tok_embeddings(tokens).unsqueeze(-2).expand(-1, -1, self.streams, -1)
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)
        h = self.norm(h.sum(-2))
        return h if self._skip_lm_head else self.lm_head(h)
