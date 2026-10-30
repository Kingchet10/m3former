"""Full Attention Residuals (Kimi Team 2026, arXiv 2603.15031, eqs. 1-4).

Every sublayer l (each attention and each FFN is one "layer") reads a softmax mixture over depth instead of the
residual sum:

    h_l = sum_{i < l} alpha_{i->l} v_i,    alpha_{i->l} = softmax_i( w_l . RMSNorm_l(v_i) )

with values v_0 = token embedding and v_i = f_i(h_i) the outputs of the earlier sublayers; w_l is a learned
pseudo-query per sublayer and RMSNorm_l a per-sublayer key norm with its own gain. The model output aggregates all
values with one more query. The pseudo-queries start at zero (as the paper requires), so every mixture starts as the
equal-weight average; under pre-norm the average and the residual sum differ only by a scale, so the model starts as
the standard pre-norm Transformer up to the RMSNorm epsilon.

Multi-head Attention Residuals (MHAR, arXiv 2607.27230, eq. 3): with H heads the channels split into H contiguous
slices of width d/H; head h scores every value with its slice of the pseudo-query against the RMS-normalised slice
of the value, takes its own softmax over depth and mixes only its slice. The H queries are a reshape of the single
query, so MHAR adds no parameters; H = 1 is Attention Residuals. MHAR sets H to the number of KV heads.

Implementation: the logits use the factorisation w . RMSNorm(v) = (v . (w * g)) / rms(v), with 1/rms(v) computed once
per value; the weighted sum uses the fused depth-mixing kernel (mix_kernel.py, one stream, one group).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

from .m3 import FUSED, _local  # also raises the torch.compile cache limits (one graph per block)

EPS = 1e-6


def attnres_parameters(dim, layers):
    """Pseudo-query and key-norm gain for each of the 2L sublayers, plus the output aggregation."""
    return 4 * dim * layers + 2 * dim


def inv_rms(v, heads=1):
    """1 / rms over the channels (heads = 1) or over each head's channel slice (shape [..., heads])."""
    v = v.float()
    if heads == 1:
        return torch.rsqrt(v.square().mean(-1) + EPS)
    return torch.rsqrt(v.unflatten(-1, (heads, -1)).square().mean(-1) + EPS)


class AttnResBlock(Llama3TransformerBlock):
    """Called with the values v_0 .. v_{2i} and their inverse RMS; returns the two new values (attention output,
    FFN output) and their inverse RMS, and for the last block also the aggregated model output."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        layer_id: int = 0
        n_layers: int = 1
        dim: int = 0
        heads: int = 1  # MHAR routing heads (1 = Attention Residuals)

    def __init__(self, config):
        super().__init__(config)
        self.layer_id, self.n_layers, self.dim = config.layer_id, config.n_layers, config.dim
        if config.dim % config.heads:
            raise ValueError((config.dim, config.heads))
        self.heads = config.heads
        names = ['attn', 'ffn'] + (['out'] if config.layer_id == config.n_layers - 1 else [])
        for n in names:
            setattr(self, f'q_{n}', nn.Parameter(torch.empty(config.dim)))
            setattr(self, f'g_{n}', nn.Parameter(torch.empty(config.dim)))
        self._names = names

    def reset_parameters(self):
        with torch.no_grad():
            for n in self._names:
                _local(getattr(self, f'q_{n}')).zero_()
                _local(getattr(self, f'g_{n}')).fill_(1.)

    def weights(self, values, inv, name):
        """alpha[b, t, j] over the values (heads = 1), or alpha[b, t, j, h] per routing head."""
        u = (_local(getattr(self, f'q_{name}')) * _local(getattr(self, f'g_{name}'))).to(values[0].dtype)
        if self.heads == 1:
            logits = torch.stack([(v @ u).float() * r for v, r in zip(values, inv)], -1)
            return logits.softmax(-1).to(values[0].dtype)
        uh = u.view(self.heads, -1)
        logits = torch.stack([(v.unflatten(-1, (self.heads, -1)) * uh).sum(-1).float() * r
                              for v, r in zip(values, inv)], -2)
        return logits.softmax(-2).to(values[0].dtype)

    def aggregate(self, values, inv, name):
        alpha = self.weights(values, inv, name)
        heads = self.heads
        if FUSED and values[0].is_cuda:
            from .mix_kernel import mix as fused_mix
            b, t, s = alpha.shape[:3]
            return fused_mix(list(values), alpha.view(b, t, 1, s, heads).contiguous(), 1, heads)[0]
        out = None
        for j, v in enumerate(values):
            if heads == 1:
                term = alpha[..., j, None] * v
            else:
                term = (alpha[..., j, :, None] * v.unflatten(-1, (heads, -1))).flatten(-2)
            out = term if out is None else out + term
        return out

    def forward(self, values, inv, attention_masks, positions=None):
        h = self.aggregate(values, inv, 'attn')
        a = self.attention(self.attention_norm(h), attention_masks, positions)
        ra = inv_rms(a, self.heads)
        h = self.aggregate(list(values) + [a], list(inv) + [ra], 'ffn')
        f = self.feed_forward(self.ffn_norm(h))
        rf = inv_rms(f, self.heads)
        if self.layer_id == self.n_layers - 1:
            return a, f, ra, rf, self.aggregate(list(values) + [a, f], list(inv) + [ra, rf], 'out')
        return a, f, ra, rf


class AttnResModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def forward(self, tokens, positions=None, attention_masks=None):
        e = self.tok_embeddings(tokens)
        heads = next(iter(self.layers.values())).heads
        values, inv = [e], [inv_rms(e, heads)]
        out = None
        for layer in self.layers.values():
            res = layer(values, inv, attention_masks, positions)
            values += [res[0], res[1]]
            inv += [res[2], res[3]]
            if len(res) == 5:
                out = res[4]
        h = self.norm(out)
        if self._skip_lm_head:
            return h
        return self.lm_head(h)
