"""DeepCrossAttention (Heddes et al., ICML 2025, arXiv 2502.06785): our re-implementation as a coarse baseline.

The stack G_t holds the model input x (f_0 = x) and the outputs f_1 .. f_t of the earlier decoder blocks. Each block
composes its attention query, key and value inputs with three independent GRN-v3 instances

    g(G) = sum_j (b_j + sigma(w . f_j)) * f_j,        b_j in R^d (per-dimension, static), w in R^d, sigma = ReLU

(the input-dependent gate sigma(w . f_j) is one scalar per source and token, shared by all d dimensions), and the
model output is one more GRN-v3 over the whole stack. Block wiring (the paper's Fig. 4 is not given in equations;
inferred from "the remaining skip connections ensure that the inputs are not added to the outputs of the decoder
block, but are included in the inputs of both the attention and the feed forward module"):

    a = Attention(norm_q(g_q), norm_k(g_k), norm_v(g_v));   h = g_q + a;   f_t = a + FFN(norm(h))

Initialisation b = 1, w = 0 makes every GRN the plain sum of the stack, i.e. the residual stream, so the model starts
as the standard pre-norm Transformer (as the paper states). The ReLU gate uses clamp(min=0), whose subgradient at 0 is
1: with w = 0 every gate input is exactly 0, and the usual relu'(0) = 0 would leave w untrained.
"""
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

from .depth_dendrites import DepthBlock, _local  # also raises the torch.compile cache limits (one graph per block)

ROLES = ('q', 'k', 'v')


def dca_parameters(dim, layers):
    """Per block l (sources l + 1): three GRNs (static b and gate w) plus key / value input norms; output GRN."""
    return sum(3 * ((l + 1) * dim + dim) + 2 * dim for l in range(layers)) + (layers + 1) * dim + dim


def grn(stack, b, w):
    out = None
    for j, f in enumerate(stack):
        gate = torch.clamp((f @ w.to(f.dtype)).float(), min=0.).to(f.dtype)[..., None]
        term = (b[j].to(f.dtype) + gate) * f
        out = term if out is None else out + term
    return out


class DCABlock(Llama3TransformerBlock):
    """Called with the stack [x, f_1, ..., f_l]; returns f_{l+1} (the last block returns the GRN model output)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        layer_id: int = 0
        n_layers: int = 1
        dim: int = 0
        input_norm: object = None

    attend = DepthBlock.attend  # attention with separate query / key / value inputs (fused QKV weight's slices)

    def __init__(self, config):
        super().__init__(config)
        self.layer_id, self.n_layers, self.dim = config.layer_id, config.n_layers, config.dim
        sources = config.layer_id + 1
        for r in ROLES:
            setattr(self, f'b_{r}', nn.Parameter(torch.empty(sources, config.dim)))
            setattr(self, f'w_{r}', nn.Parameter(torch.empty(config.dim)))
        self.norm_k, self.norm_v = config.input_norm.build(), config.input_norm.build()
        if config.layer_id == config.n_layers - 1:
            self.b_out = nn.Parameter(torch.empty(config.n_layers + 1, config.dim))
            self.w_out = nn.Parameter(torch.empty(config.dim))

    def _grns(self):
        return [r for r in (*ROLES, 'out') if f'b_{r}' in self._parameters]

    def reset_parameters(self):
        with torch.no_grad():
            for r in self._grns():
                _local(getattr(self, f'b_{r}')).fill_(1.)
                _local(getattr(self, f'w_{r}')).zero_()

    def forward(self, stack, attention_masks, positions=None):
        xq, xk, xv = (grn(stack, _local(getattr(self, f'b_{r}')), _local(getattr(self, f'w_{r}'))) for r in ROLES)
        a = self.attend(self.attention_norm(xq), self.norm_k(xk), self.norm_v(xv), attention_masks, positions)
        h = xq + a
        f = a + self.feed_forward(self.ffn_norm(h))
        if self.layer_id == self.n_layers - 1:
            return grn(list(stack) + [f], _local(self.b_out), _local(self.w_out))
        return f


class DCAModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def forward(self, tokens, positions=None, attention_masks=None):
        stack = [self.tok_embeddings(tokens)]
        out = None
        layers = list(self.layers.values())
        for i, layer in enumerate(layers):
            r = layer(stack, attention_masks, positions)
            if i == len(layers) - 1:
                out = r
            else:
                stack.append(r)
        h = self.norm(out)
        if self._skip_lm_head:
            return h
        return self.lm_head(h)
