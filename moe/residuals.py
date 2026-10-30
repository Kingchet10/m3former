"""Residual connections of the MoE models: Transformer, MUDDFormer, M3 and block attention residuals.

History mixing in [sequence, microbatch, hidden] layout; no cross-step cache.
"""
import os
import re

import torch
from torch import nn
from torch.nn import functional as F


# Original run names (left) are accepted so that the released checkpoints load unchanged.
ALIASES = {'baseline': 'transformer', 'mudd': 'muddformer', 'attnres_block4': 'attnres-block'}


def parse_variant(variant):
    """Return (kind, groups): kind is 'transformer', 'm3' or 'attnres-block'; groups is the number of groups per
    role of M3 (1 = MUDDFormer) and None otherwise."""
    v = ALIASES.get(variant, variant)
    if v == 'muddformer':
        return 'm3', 1
    m = re.fullmatch(r'(?:m3-g|dd)(\d+)', v)
    if m:
        return 'm3', int(m.group(1))
    if v in ('transformer', 'attnres-block'):
        return v, None
    raise ValueError(f'Unknown MoE variant: {variant}')


def _rms(x,weight,eps):
    y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)
    return (y*weight).to(x.dtype)

_compiled_rms=torch.compile(_rms,fullgraph=True)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        if os.environ.get('M3_MOE_COMPILE_NORM','0') == '1' and x.is_cuda:
            return _compiled_rms(x,self.weight,self.eps)
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


class DepthMixer(nn.Module):
    def __init__(self, dim, sources, streams, groups, expand=1):
        super().__init__()
        assert dim % groups == 0
        self.groups, self.streams, self.sources = groups, streams, sources
        hidden = (sources * streams * expand // 64 + 1) * 64
        self.w1 = nn.Parameter(torch.empty(hidden, dim))
        self.w2 = nn.Parameter(torch.zeros(streams * sources * groups, hidden))
        self.a = nn.Parameter(torch.zeros(streams, sources, groups))
        nn.init.normal_(self.w1, std=dim ** -.5)
        with torch.no_grad():
            self.a[:, -1, :] = 1.

    def forward(self, history):
        assert len(history) == self.sources
        x = history[-1]
        xn = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        w = F.linear(F.gelu(F.linear(xn, self.w1.to(x.dtype))), self.w2.to(x.dtype))
        w = w.reshape(*x.shape[:-1], self.streams, self.sources, self.groups) + self.a.to(x.dtype)
        # Fused existing project operator, if available, avoids a history stack.
        if x.is_cuda:
            from depth_mix_kernel import mix
            return list(mix(list(history), w.contiguous(), self.streams, self.groups).unbind(0))
        values = torch.stack(history, dim=-2).unflatten(-1, (self.groups, -1))
        return list(torch.einsum('...cjg,...jgd->c...gd', w, values).flatten(-2).unbind(0))


class AttentionReader(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.key_scale = nn.Parameter(torch.ones(dim))

    def forward(self, values):
        if values[0].is_cuda:
            from fla_attnres_op import mix
            return mix(self.query, values, self.key_scale)
        v = torch.stack(values, dim=0)
        keys = v.float() * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + 1e-6)
        logits = (keys * self.key_scale * self.query).sum(-1)
        weights = logits.softmax(0).to(v.dtype)
        return (weights[..., None] * v).sum(0)


class ResidualDecoder(nn.Module):
    """Megatron decoder modules with explicit residual wiring; TP/PP/EP = 1.

    History is local to a single forward. All learned mixers are owned by this
    module and synchronized by the outer DDP wrapper. No FSDP cross-owner use.
    """
    def __init__(self, native_decoder, config, variant):
        super().__init__()
        self.config, self.variant = config, variant
        self.layers = native_decoder.layers
        self.final_layernorm = native_decoder.final_layernorm
        self.num_layers_per_pipeline_rank = len(self.layers)
        dim, layers = config.hidden_size, len(self.layers)
        self.mixers = nn.ModuleDict()
        self.readers = nn.ModuleDict()
        self.kind, groups = parse_variant(variant)
        if self.kind == 'm3':
            for i in range(1, layers):
                self.mixers[str(i)] = DepthMixer(dim, i+1, 4, groups)
                self.layers[i].norm_k = RMSNorm(dim, config.layernorm_epsilon)
                self.layers[i].norm_v = RMSNorm(dim, config.layernorm_epsilon)
            self.mixers['out'] = DepthMixer(dim, layers+1, 1, groups, expand=4)
        if self.kind == 'attnres-block':
            for i in range(layers):
                for part in ('attn', 'moe'):
                    self.readers[f'{i}_{part}'] = AttentionReader(dim)
            self.readers['out'] = AttentionReader(dim)

    def set_input_tensor(self, tensor):
        raise NotImplementedError('Pipeline parallelism is not enabled in this recipe')

    @staticmethod
    def output(pair):
        value, bias = pair
        return value if bias is None else value + bias

    def forward(self, hidden_states, attention_mask, rotary_pos_emb=None,
                packed_seq_params=None, inference_context=None, **unused):
        if inference_context is not None:
            raise NotImplementedError('This implementation is the pretraining/evaluation path')
        h = hidden_states.to(torch.bfloat16) if self.config.bf16 else hidden_states
        history, blocks, partial = [h], [h], None
        for i, layer in enumerate(self.layers):
            kwargs = dict(attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb,
                          packed_seq_params=packed_seq_params)
            if self.kind == 'attnres-block':
                if i and i % 4 == 0:
                    blocks.append(partial)
                    partial = None
                values = blocks + ([] if partial is None else [partial])
                x = layer.input_layernorm(self.readers[f'{i}_attn'](values))
                attn = self.output(layer.self_attention(x, **kwargs))
                partial = attn if partial is None else partial + attn
                x = layer.pre_mlp_layernorm(self.readers[f'{i}_moe'](blocks + [partial]))
                partial = partial + self.output(layer.mlp(x))
                continue
            if self.kind == 'm3' and i:
                q, k, v, residual = self.mixers[str(i)](history)
                attn = self.output(layer.self_attention(
                    layer.input_layernorm(q), key_value_states=(layer.norm_k(k), layer.norm_v(v)),
                    **kwargs))
            else:
                residual = h
                attn = self.output(layer.self_attention(layer.input_layernorm(h), **kwargs))
            h = residual + attn
            h = h + self.output(layer.mlp(layer.pre_mlp_layernorm(h)))
            if self.kind == 'm3':
                history.append(h)
        if self.kind == 'attnres-block':
            h = self.readers['out'](blocks + [partial])
        elif self.kind == 'm3':
            h = self.mixers['out'](history)[0]
        return self.final_layernorm(h)


def attention_class():
    from megatron.core.transformer.attention import SelfAttention

    class MultiwaySelfAttention(SelfAttention):
        def get_query_key_value_tensors(self, hidden_states, key_value_states=None,
                                       output_gate=False, split_qkv=True):
            if key_value_states is None:
                return super().get_query_key_value_tensors(hidden_states, None, output_gate, split_qkv)
            assert split_qkv and not output_gate and self.config.tensor_model_parallel_size == 1
            # MCore stores [head, Q/K/V, head_dim, input_dim] for MHA.
            d, heads = self.config.hidden_size, self.config.num_attention_heads
            weight = self.linear_qkv.weight.reshape(heads, 3, d//heads, d)
            result = []
            for j, x in enumerate((hidden_states, *key_value_states)):
                y = F.linear(x, weight[:, j].reshape(d, d).to(x.dtype))
                result.append(y.reshape(*x.shape[:-1], heads, d//heads))
            return tuple(result)
    return MultiwaySelfAttention
