"""Eight Block AttnRes blocks for 32 Transformer layers, with pinned FLA kernels."""
from dataclasses import dataclass

from .fla_attnres_op import mix
from torchtitan.models.llama3.model import Llama3Model
from .attn_residuals import AttnResBlock
from .m3 import _local


class BlockAttnResBlock(AttnResBlock):
    def read(self, values, name, norm=None):
        h = mix(
            query=_local(getattr(self, f'q_{name}')),
            residuals=values,
            rms_weight=_local(getattr(self, f'g_{name}')),
        )
        # Backbone epsilon is 1e-5, while the routing key epsilon is 1e-6.
        # FLA's optional fused output norm shares epsilon; preserve both values.
        return norm(h) if norm is not None else h

    def forward(self, blocks, partial, attention_masks, positions=None):
        values = list(blocks) + ([] if partial is None else [partial])
        h = self.read(values, 'attn', self.attention_norm)
        a = self.attention(h, attention_masks, positions)
        partial = a if partial is None else partial + a
        h = self.read(list(blocks) + [partial], 'ffn', self.ffn_norm)
        partial = partial + self.feed_forward(h)
        if self.layer_id == self.n_layers - 1:
            return partial, self.read(list(blocks) + [partial], 'out')
        return partial, None

    @dataclass(kw_only=True, slots=True)
    class Config(AttnResBlock.Config):
        pass


class BlockAttnResModel(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        attnres_block_size: int = 8  # Counts sublayers: 4 Attention + 4 FFN.

    def __init__(self, config):
        super().__init__(config)
        assert config.attnres_block_size > 0 and config.attnres_block_size % 2 == 0
        self.layers_per_block = config.attnres_block_size // 2

    def forward(self, tokens, positions=None, attention_masks=None):
        blocks = [self.tok_embeddings(tokens)]
        partial, out = None, None
        for i, layer in enumerate(self.layers.values()):
            if i and i % self.layers_per_block == 0:
                blocks.append(partial)
                partial = None
            partial, out = layer(blocks, partial, attention_masks, positions)
        h = self.norm(out)
        return h if self._skip_lm_head else self.lm_head(h)
