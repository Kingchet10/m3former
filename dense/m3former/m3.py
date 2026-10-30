"""M3: Multi-Role Multi-Group Mixing Residual (the residual connection of M3Former).

Block l (l >= 1) reads its query, key, value and residual inputs (the four roles) as token-dependent mixtures of
all earlier hidden states h_0 (token embedding), ..., h_{l-1} (output of the previous block). The channels of every
hidden state are split into G contiguous groups, and every role and group has its own mixture:

    z[r, g] = sum_j A[r, j, g] * h_j[g],    A = W2 GELU(W1 RMSNorm_noscale(h_{l-1})) + a

A single generator per block produces all 4 * G * l coefficients from the latest state of the same token. The hidden
width of the generator is the smallest multiple of 64 that exceeds 4 l, as in the MUDDFormer reference
implementation, and the key and value inputs get their own RMSNorm. G = 1 is MUDDFormer (multi-role mixing). The
last block also forms the model output as a mixture of h_0 .. h_L (one role, four times the hidden width).

Initialization (W2 = 0, a = 1 on the latest state and 0 elsewhere, W1 ~ N(0, 1/d)) makes the model compute exactly
the standard pre-norm Transformer at the start of training.

Modes:
- 'dense': four roles (M3 for G > 1, MUDDFormer for G = 1).
- 'dense1': multi-group mixing, the ablation of the paper. One mixture per group is both the residual and the
  attention input, so the four roles share their coefficients.
- 'static': ablation with the static coefficients a only (no token dependence). Not used in the paper.
"""
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

STREAMS = 4  # query, key, value, residual
FUSED = os.environ.get('M3_FUSED_MIX', '1') != '0'  # Triton mixing kernels (mix_kernel.py); 0 = PyTorch loop
MODES = ('dense', 'dense1', 'static')

# Every block has its own input arity (one more source per layer), so torch.compile keeps one graph per block.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)


def mixer_hidden(sources, streams, expand=1):
    return (sources * streams * expand // 64 + 1) * 64  # MUDDFormer reference: round up to a multiple of 64


def mixer_parameters(dim, sources, streams, groups, expand=1):
    hidden = mixer_hidden(sources, streams, expand)
    out = sources * streams * groups
    return dim * hidden + hidden * out + out


def m3_parameters(dim, layers, groups, mode='dense'):
    """Parameters added to the Transformer: the input mixers of blocks 2 .. L (four roles, or one for 'dense1'),
    their key and value input norms, and the output mixer of the last block (one role, 4x hidden width)."""
    if mode == 'static':
        return (layers + 1) * groups + sum((i + 1) * STREAMS * groups + 2 * dim for i in range(1, layers))
    total = mixer_parameters(dim, layers + 1, 1, groups, expand=4)
    if mode == 'dense1':
        return total + sum(mixer_parameters(dim, i + 1, 1, groups) for i in range(1, layers))
    for i in range(1, layers):
        total += mixer_parameters(dim, i + 1, STREAMS, groups) + 2 * dim
    return total


def _local(t):
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t


class M3Block(Llama3TransformerBlock):
    """Pre-norm block whose inputs are mixtures of all earlier hidden states. Called with the list of hidden states
    h_0 .. h_{l-1}; returns its output h_l (the last block returns the mixed model output)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        layer_id: int = 0
        n_layers: int = 1
        groups: int = 1
        dim: int = 0
        input_norm: object = None
        mode: str = 'dense'  # see MODES

    def __init__(self, config):
        super().__init__(config)
        i, n, dim, groups = config.layer_id, config.n_layers, config.dim, config.groups
        if dim % groups or not 0 <= i < n:
            raise ValueError((dim, groups, i, n))
        if config.mode not in MODES:
            raise ValueError(config.mode)
        self.layer_id, self.n_layers, self.dim, self.groups, self.mode = i, n, dim, groups, config.mode
        dynamic = config.mode != 'static'
        if i > 0:
            if config.mode != 'dense1':
                self.norm_k, self.norm_v = config.input_norm.build(), config.input_norm.build()
            streams, sources = (1, i + 1) if config.mode == 'dense1' else (STREAMS, i + 1)
            hidden = mixer_hidden(sources, streams)
            if dynamic:
                self.in_w1 = nn.Parameter(torch.empty(hidden, dim))
                self.in_w2 = nn.Parameter(torch.empty(streams * sources * groups, hidden))
            self.in_a = nn.Parameter(torch.empty(streams, sources, groups))
        if i == n - 1:
            hidden = mixer_hidden(n + 1, 1, expand=4)
            if dynamic:
                self.out_w1 = nn.Parameter(torch.empty(hidden, dim))
                self.out_w2 = nn.Parameter(torch.empty((n + 1) * groups, hidden))
            self.out_a = nn.Parameter(torch.empty(1, n + 1, groups))

    def _mixers(self):
        return [p for p in ('in', 'out') if f'{p}_a' in self._parameters]

    def reset_parameters(self):
        with torch.no_grad():
            for p in self._mixers():  # on local tensors: slice assignment has no DTensor sharding rule
                if f'{p}_w1' in self._parameters:
                    nn.init.normal_(_local(getattr(self, f'{p}_w1')), std=self.dim ** -0.5)
                    _local(getattr(self, f'{p}_w2')).zero_()
                a = _local(getattr(self, f'{p}_a'))
                a.zero_()
                a[:, -1, :] = 1.

    def mixing_weights(self, x, prefix, streams):
        """A[b, t, c, j, g] for the mixer `prefix`, computed from the latest hidden state x."""
        a = _local(getattr(self, f'{prefix}_a'))
        b, t, _ = x.shape
        if f'{prefix}_w1' not in self._parameters:  # static ablation
            return a.to(x.dtype).expand(b, t, *a.shape)
        w1, w2 = (_local(getattr(self, f'{prefix}_{n}')) for n in ('w1', 'w2'))
        xf = x.float()
        xn = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        w = F.linear(F.gelu(F.linear(xn, w1.to(x.dtype))), w2.to(x.dtype))
        return w.view(b, t, streams, a.shape[1], self.groups) + a.to(x.dtype)

    def mix(self, hiddens, prefix, streams):
        b, t, d = hiddens[-1].shape
        w = self.mixing_weights(hiddens[-1], prefix, streams)
        if FUSED and hiddens[-1].is_cuda:
            from .mix_kernel import mix as fused_mix
            return list(fused_mix(list(hiddens), w.contiguous(), streams, self.groups).unbind(0))
        out = []
        for c in range(streams):
            acc = None
            for j, h in enumerate(hiddens):
                term = w[:, :, c, j, :, None] * h.view(b, t, self.groups, d // self.groups)
                acc = term if acc is None else acc + term
            out.append(acc.reshape(b, t, d))
        return out

    def attend(self, xq, xk, xv, attention_masks, positions):
        """Self-attention with separate query, key and value inputs, using the fused QKV weight's slices."""
        att = self.attention
        proj = att.qkv_linear
        if getattr(proj.wqkv, 'bias', None) is not None:
            raise NotImplementedError('fused QKV bias')
        w = _local(proj.wqkv.weight)
        hd, hpk = proj.head_dim, proj.heads_per_kv
        w = w.view(-1, hpk + 2, hd, w.shape[-1])
        b, t, _ = xq.shape
        q = F.linear(xq, w[:, :hpk].reshape(-1, w.shape[-1])).view(b, t, -1, hd)
        k = F.linear(xk, w[:, hpk].reshape(-1, w.shape[-1])).view(b, t, -1, hd)
        v = F.linear(xv, w[:, hpk + 1].reshape(-1, w.shape[-1])).view(b, t, -1, hd)
        if att.q_norm is not None:
            q, k = att.q_norm(q), att.k_norm(k)
        q, k = att.rope(q, k, positions)
        o = att.inner_attention(q, k, v, attention_masks=attention_masks, scale=att.scaling,
                                enable_gqa=att.enable_gqa).contiguous()
        return att.wo(o.flatten(2))

    def forward(self, hiddens, attention_masks, positions=None):
        if self.layer_id == 0:
            x = hiddens[-1]
            a = self.attention(self.attention_norm(x), attention_masks, positions)
        elif self.mode == 'dense1':  # one mixture is the residual and the attention input
            x = self.mix(hiddens, 'in', 1)[0]
            a = self.attention(self.attention_norm(x), attention_masks, positions)
        else:
            xq, xk, xv, x = self.mix(hiddens, 'in', STREAMS)
            a = self.attend(self.attention_norm(xq), self.norm_k(xk), self.norm_v(xv), attention_masks, positions)
        h = x + a
        out = h + self.feed_forward(self.ffn_norm(h))
        if self.layer_id == self.n_layers - 1:
            out = self.mix(list(hiddens) + [out], 'out', 1)[0]
        return out


class M3Model(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        depth_groups: int = 1

    def forward(self, tokens, positions=None, attention_masks=None):
        hiddens = [self.tok_embeddings(tokens)]
        for layer in self.layers.values():
            hiddens.append(layer(hiddens, attention_masks, positions))
        h = self.norm(hiddens[-1])
        if self._skip_lm_head:
            return h
        return self.lm_head(h)
