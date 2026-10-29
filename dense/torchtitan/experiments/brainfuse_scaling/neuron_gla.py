"""Document-reset gated linear attention for Step 2.

The projection and RoPE path is shared with the frozen softmax model.  Only
the attention memory is replaced by a per-key-channel-decayed KV state.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.llama3 import _LINEAR_INIT
from torchtitan.models.llama3.model import Llama3Model


def document_cu_seqlens(positions):
    """One independent state per packed document, including batch boundaries."""
    if positions is None or positions.ndim != 2 or not bool(torch.all(positions[:, 0] == 0)):
        raise ValueError('GLA requires [batch, time] document positions beginning at zero')
    starts = (positions == 0).flatten().nonzero().flatten()
    return torch.cat((starts, starts.new_tensor([positions.numel()])))


def gla_reference(q, k, v, g, positions):
    """Slow independent serial oracle for causal and document-reset tests."""
    b, t, heads, d = q.shape
    state = q.new_zeros(b, heads, d, v.shape[-1])
    outputs = []
    for j in range(t):
        state = torch.where((positions[:, j] == 0)[:, None, None, None], 0., state)
        state = g[:, j].exp()[..., None] * state + k[:, j][..., None] * v[:, j][..., None, :]
        outputs.append(torch.einsum('bhd,bhde->bhe', q[:, j], state) * d**-0.5)
    return torch.stack(outputs, 1)


def gla(q, k, v, g, positions):
    if not q.is_cuda:
        return gla_reference(q, k, v, g, positions)
    from fla.ops.gla import chunk_gla

    b, t, heads, d = q.shape
    boundaries = document_cu_seqlens(positions)
    out, _ = chunk_gla(
        q.reshape(1, b*t, heads, d), k.reshape(1, b*t, heads, d),
        v.reshape(1, b*t, heads, v.shape[-1]),
        g.reshape(1, b*t, heads, d),
        cu_seqlens=boundaries,
    )
    return out.reshape(b, t, heads, v.shape[-1])


class DocumentGLAAttention(GQAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        gate: Linear.Config

    def __init__(self, config):
        super().__init__(config)
        if self.n_heads != self.n_kv_heads:
            raise ValueError('Step 2 GLA requires the same number of Q and KV heads')
        self.gate = config.gate.build()

    def forward(self, x, attention_masks, positions=None):
        if positions is None:
            raise ValueError('GLA requires explicit document positions')
        q, k, v = self.qkv_linear(x)
        q, k = self.rope(q, k, positions)
        g = F.logsigmoid(self.gate(x).unflatten(-1, (self.n_heads, self.head_dim))) / 16.
        out = gla(q, k, v, g, positions)
        # Per-head output normalization is the GLA layer's standard stabilizer.
        out = F.rms_norm(out, (out.shape[-1],))
        return self.wo(out.flatten(2))


class DocumentGLAModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def get_attention_masks(self, positions):
        # The GLA recurrence receives document boundaries from positions.
        return None


def extend_gla_config(config):
    for layer in config.layers:
        original = layer.attention
        layer.attention = DocumentGLAAttention.Config(
            **{field.name: getattr(original, field.name) for field in fields(original)},
            gate=Linear.Config(in_features=original.dim, out_features=original.dim,
                               bias=True, param_init=_LINEAR_INIT),
        )
    return DocumentGLAModel.Config(
        **{field.name: getattr(config, field.name) for field in fields(config)})
