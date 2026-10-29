"""Depth dendrites: multiway dynamic dense connections across depth, with per-compartment source selection.

Block i (i >= 1) reads its four input streams (query, key, value, residual) as input-dependent mixtures of all
earlier hidden states h_0 (embedding), ..., h_i (output of block i - 1):

    x_c = sum_j A[c, j] * h_j,    A = W2 GELU(W1 RMSNorm_noscale(h_i)) + a

which is MUDDFormer's multiway dynamic dense connection (Xiao et al. 2025, eq. 6; hidden width rounded as in the
reference implementation, separate input norms for query, key and value). With groups G > 1 the residual
channels form G compartments, each with its own source weights A[c, j, g]: a neuron type reads each depth with its
own weight, as dendritic compartments of different neuron types sample different input layers. G = 1 is MUDD.
The last block also forms the model output as such a mixture of h_0 .. h_L (one stream, 4x hidden width).
Initialisation (W2 = 0, a = 1 on the latest state, W1 ~ N(0, 1/D)) reduces the model exactly to the standard
pre-norm Transformer.

Ablations (mode): 'static' keeps only the static weights a (no input dependence); 'vemb' keeps a dynamic mixture only
for the value stream and only over {h_0 (token embedding), h_i}, every other input being h_i as in the baseline;
'plain' has no depth mixing at all (the baseline block; used with a wide residual stream).
'dense1' is single-way dynamic dense reading (MUDDFormer's "dynamic dense" ablation, Table 5): one mixture
x = sum_j A[j, g] h_j is both the block's residual and its attention input, so the query, key, value and residual
roles share one set of source weights; G > 1 splits the channels into G subspaces with their own weights (the
single-way analogue of Multi-Head Attention Residuals). Not defined for a wide residual stream.

Wide residual stream (residual R > D): the hidden states h_j have width R while every block computes at width D.
Block i reads its inputs through a learned projection P_i (R -> D) and writes its update through Q_i (D -> R); the
token embedding (width D) enters zero-padded and the last block reads the output through P_out (R -> D). Each is
the fixed identity [I 0] (or its transpose) plus a learned delta, so weight decay pulls towards the width-D model
rather than towards cutting the residual path; the delta's identity block starts at 0 and its extra entries at
N(0, 0.02^2). This gives the network a learnable way around the fixed-width residual sum (each block can write to
and read from its own subspaces), the depth analogue of a learnable presynaptic basis.

LDSD studies (plans/LDSD_跨层连接_QA_QB_QC_预注册_2026-10-10.md):
- lns: LayerNorm Scaling (Sun et al. 2025, "The Curse of Depth"): every pre-norm output of block i (attention, key,
  value and FFN input norms) is multiplied by 1/sqrt(i + 1). With mode 'plain' this is the LNS baseline.
- mix_norm 'l1': mixing weights normalised over sources, A[c, j, g] / sum_j |A[c, j, g]|, so every mixture is a signed
  convex-type combination with gain <= 1 (a bounded counterpart of MUDD; the one-hot initialisation is unchanged).
"""
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

STREAMS = 4  # query, key, value, residual
FUSED = os.environ.get('SA_DEPTH_FUSED', '1') != '0'  # Triton mixing kernels (depth_mix_kernel); 0 = PyTorch loop
WIDE_INIT_STD = 0.02  # extra entries of the read / write projections of a wide residual stream

# Every block has its own input arity (one more source per layer), so torch.compile keeps one graph per block.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)


def mixer_hidden(sources, streams, expand=1, mult=1):
    return (sources * streams * expand // 64 + 1) * 64 * mult  # MUDDFormer reference: round64; mult widens it


def mixer_parameters(dim, sources, streams, groups, expand=1, mult=1):
    """dim: width of the hidden states the mixer reads (the residual width)."""
    hidden = mixer_hidden(sources, streams, expand, mult)
    out = sources * streams * groups
    return dim * hidden + hidden * out + out


def depth_parameters(dim, layers, groups, mode='dense', mult=1, residual=0):
    """Input mixers of blocks 1 .. L-1 (4 streams), their key/value input norms, and the output mixer of the last
    block (1 stream, 4x hidden width). 'static': the static weights only; 'vemb': a value-stream mixer over 2 sources;
    'plain': none. A wide residual stream (residual > dim) adds the read / write projections of every block and the
    output read."""
    r = residual or dim
    total = 2 * r * dim * layers + r * dim if r != dim else 0
    if mode == 'plain':
        return total
    if mode == 'static':
        return total + (layers + 1) * groups + sum((i + 1) * STREAMS * groups + 2 * dim for i in range(1, layers))
    if mode == 'vemb':
        return total + (layers - 1) * (mixer_parameters(r, 2, 1, groups) + 2 * dim)
    if mode == 'dense1':
        total += mixer_parameters(r, layers + 1, 1, groups, expand=4, mult=mult)
        return total + sum(mixer_parameters(r, i + 1, 1, groups, mult=mult) for i in range(1, layers))
    total += mixer_parameters(r, layers + 1, 1, groups, expand=4, mult=mult)
    for i in range(1, layers):
        total += mixer_parameters(r, i + 1, STREAMS, groups, mult=mult) + 2 * dim
    return total


def _local(t):
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t


class DepthBlock(Llama3TransformerBlock):
    """Pre-norm block whose inputs are dynamic mixtures of all earlier hidden states. Called with the list of
    hidden states h_0 .. h_i; returns its output h_{i+1} (the last block returns the mixed model output)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        layer_id: int = 0
        n_layers: int = 1
        groups: int = 1
        dim: int = 0
        input_norm: object = None
        mode: str = 'dense'  # 'dense' | 'dense1' (single-way) | 'static' | 'vemb' | 'plain' (ablations / no mixing)
        hidden_mult: int = 1  # mixer hidden width multiplier (parameter-matched control for more groups)
        residual: int = 0  # residual stream width; 0 = dim
        lns: bool = False  # LayerNorm Scaling: pre-norm outputs x 1/sqrt(layer_id + 1)
        mix_norm: str = ''  # '' | 'l1' (mixing weights normalised over sources)

    def __init__(self, config):
        super().__init__(config)
        i, n, dim, groups = config.layer_id, config.n_layers, config.dim, config.groups
        r = config.residual or dim
        if r % groups or r < dim or not 0 <= i < n:
            raise ValueError((dim, r, groups, i, n))
        if config.mode not in ('dense', 'dense1', 'static', 'vemb', 'plain') or (config.mode in ('vemb', 'dense1') and r != dim):
            raise ValueError(config.mode)
        if config.mix_norm not in ('', 'l1'):
            raise ValueError(config.mix_norm)
        self.lns_scale = (i + 1) ** -0.5 if config.lns else 1.
        self.mix_norm = config.mix_norm
        self.layer_id, self.n_layers, self.dim, self.groups, self.mode = i, n, dim, groups, config.mode
        self.residual = r
        if r != dim:
            self.read = nn.Parameter(torch.empty(dim, r))
            self.write = nn.Parameter(torch.empty(r, dim))
            if i == n - 1:
                self.out_read = nn.Parameter(torch.empty(dim, r))
        dynamic = config.mode not in ('static', 'plain')
        if i > 0 and config.mode != 'plain':
            if config.mode != 'dense1':
                self.norm_k, self.norm_v = config.input_norm.build(), config.input_norm.build()
            streams, sources = {'vemb': (1, 2), 'dense1': (1, i + 1)}.get(config.mode, (STREAMS, i + 1))
            hidden = mixer_hidden(sources, streams, mult=config.hidden_mult)
            if dynamic:
                self.in_w1 = nn.Parameter(torch.empty(hidden, r))
                self.in_w2 = nn.Parameter(torch.empty(streams * sources * groups, hidden))
            self.in_a = nn.Parameter(torch.empty(streams, sources, groups))
        if i == n - 1 and config.mode not in ('vemb', 'plain'):
            hidden = mixer_hidden(n + 1, 1, expand=4, mult=config.hidden_mult)
            if dynamic:
                self.out_w1 = nn.Parameter(torch.empty(hidden, r))
                self.out_w2 = nn.Parameter(torch.empty((n + 1) * groups, hidden))
            self.out_a = nn.Parameter(torch.empty(1, n + 1, groups))

    def _mixers(self):
        return [p for p in ('in', 'out') if f'{p}_a' in self._parameters]

    def reset_parameters(self):
        with torch.no_grad():
            for p in self._mixers():  # on local tensors: slice assignment has no DTensor sharding rule
                if f'{p}_w1' in self._parameters:
                    nn.init.normal_(_local(getattr(self, f'{p}_w1')), std=self.residual ** -0.5)
                    _local(getattr(self, f'{p}_w2')).zero_()
                a = _local(getattr(self, f'{p}_a'))
                a.zero_()
                a[:, -1, :] = 1.
            for name in ('read', 'write', 'out_read'):  # deltas from [I 0]: identity block 0, extra entries small
                if name in self._parameters:
                    w = _local(getattr(self, name))
                    nn.init.normal_(w, std=WIDE_INIT_STD)
                    (w[:, :self.dim] if name != 'write' else w[:self.dim, :]).zero_()

    def _proj(self, x, name):
        """[I 0] + delta: reads (R -> D) keep the first D channels, the write (D -> R) zero-pads."""
        if name not in self._parameters:
            return x
        y = F.linear(x, _local(getattr(self, name)).to(x.dtype))
        return y + (x[..., :self.dim] if name != 'write' else F.pad(x, (0, self.residual - self.dim)))

    def mixing_weights(self, x, prefix, streams):
        """A[b, t, c, j, g] for the mixer `prefix`, computed from the latest hidden state x."""
        a = _local(getattr(self, f'{prefix}_a'))
        b, t, _ = x.shape
        if f'{prefix}_w1' not in self._parameters:  # static ablation
            if self.mix_norm == 'l1':
                a = a / (a.abs().sum(1, keepdim=True) + 1e-6)
            return a.to(x.dtype).expand(b, t, *a.shape)
        w1, w2 = (_local(getattr(self, f'{prefix}_{n}')) for n in ('w1', 'w2'))
        xf = x.float()
        xn = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        w = F.linear(F.gelu(F.linear(xn, w1.to(x.dtype))), w2.to(x.dtype))
        w = w.view(b, t, streams, a.shape[1], self.groups) + a.to(x.dtype)
        if self.mix_norm == 'l1':  # gain <= 1 per mixture: sum over sources of |A| is 1
            wf = w.float()
            w = (wf / (wf.abs().sum(3, keepdim=True) + 1e-6)).to(x.dtype)
        return w

    def mix(self, hiddens, prefix, streams):
        b, t, d = hiddens[-1].shape
        w = self.mixing_weights(hiddens[-1], prefix, streams)
        if FUSED and hiddens[-1].is_cuda:
            from .depth_mix_kernel import mix as fused_mix
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

    def _norm(self, norm, x):
        y = norm(x)
        return y * self.lns_scale if self.lns_scale != 1. else y

    def forward(self, hiddens, attention_masks, positions=None):
        if self.layer_id == 0 or self.mode == 'plain':
            xr = hiddens[-1]
            x = self._proj(xr, 'read')
            a = self.attention(self._norm(self.attention_norm, x), attention_masks, positions)
        elif self.mode == 'vemb':  # only the value stream reads a mixture of the embedding and the latest state
            xr = x = hiddens[-1]
            xv = self.mix([hiddens[0], x], 'in', 1)[0]
            a = self.attend(self._norm(self.attention_norm, x), self._norm(self.norm_k, x),
                            self._norm(self.norm_v, xv), attention_masks, positions)
        elif self.mode == 'dense1':  # one mixture is the residual and the attention input
            xr = x = self.mix(hiddens, 'in', 1)[0]
            a = self.attention(self._norm(self.attention_norm, x), attention_masks, positions)
        else:
            xq, xk, xv, xr = self.mix(hiddens, 'in', STREAMS)
            x = self._proj(xr, 'read')
            a = self.attend(self._norm(self.attention_norm, self._proj(xq, 'read')),
                            self._norm(self.norm_k, self._proj(xk, 'read')),
                            self._norm(self.norm_v, self._proj(xv, 'read')), attention_masks, positions)
        h = x + a
        if 'write' in self._parameters:  # wide residual: the block's update is written back through Q
            out = xr + self._proj(a + self.feed_forward(self._norm(self.ffn_norm, h)), 'write')
        else:
            out = h + self.feed_forward(self._norm(self.ffn_norm, h))
        if self.layer_id == self.n_layers - 1:
            if self.mode not in ('vemb', 'plain'):
                out = self.mix(list(hiddens) + [out], 'out', 1)[0]
            out = self._proj(out, 'out_read')
        return out


class DepthModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        depth_groups: int = 1
        residual: int = 0  # residual stream width; 0 = dim

    def forward(self, tokens, positions=None, attention_masks=None):
        h0 = self.tok_embeddings(tokens)
        r = self.config.residual or h0.shape[-1]
        hiddens = [F.pad(h0, (0, r - h0.shape[-1])) if r != h0.shape[-1] else h0]
        for layer in self.layers.values():
            hiddens.append(layer(hiddens, attention_masks, positions))
        h = self.norm(hiddens[-1])
        if self._skip_lm_head:
            return h
        return self.lm_head(h)
