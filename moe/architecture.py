"""Megatron Core MoE backbone; the residual connection is replaced in residuals.ResidualDecoder.

No dataset, optimizer or job submission side effects.
"""
from dataclasses import dataclass, asdict
import json
import os


@dataclass(frozen=True)
class Architecture:
    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_query_groups: int = 16
    kv_channels: int = 64
    num_experts: int = 64
    topk: int = 8
    expert_hidden_size: int = 352
    vocab_size: int = 32000
    sequence_length: int = 2048
    norm_epsilon: float = 1e-5
    rotary_base: int = 10000

    def parameter_counts(self):
        # Includes both untied vocabulary matrices in active count, as in common
        # architecture-level MoE reporting (not literal embedding row accesses).
        d, n = self.hidden_size, self.num_layers
        embeddings = 2 * self.vocab_size * d
        attention = n * 4 * d * d
        norms = (2 * n + 1) * d
        routers = n * d * self.num_experts
        one_expert_per_layer = n * 3 * d * self.expert_hidden_size
        common = embeddings + attention + norms + routers
        return {
            'total': common + self.num_experts * one_expert_per_layer,
            'active': common + self.topk * one_expert_per_layer,
            'vocabulary_matrices': embeddings,
            'stored_excluding_both_vocabulary_matrices': common - embeddings + self.num_experts * one_expert_per_layer,
            'active_excluding_both_vocabulary_matrices': common - embeddings + self.topk * one_expert_per_layer,
            'basis': 'analytic unsharded baseline; includes router, two RMSNorms/layer and final RMSNorm; no padding',
        }


def build_model(*, load_balancing_type, aux_loss_coeff, z_loss_coeff, variant='transformer',
                tensor_parallel=1, pipeline_parallel=1, expert_parallel=1,
                pre_process=True, post_process=True, architecture=None, use_te=True):
    """Build after Megatron distributed/model-parallel initialization.

    Router-loss settings must come from the experiment recipe, not this backbone.
    Uses TE grouped GEMM/all-to-all; no reference-project training entry point.
    """
    import torch
    import torch.nn.functional as F
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    # The Megatron Core snapshot contains optional Qwen3 routing extensions. Keep them
    # off explicitly; never invoke its loop/FFN-recurrence entry points.
    os.environ['QWEN3_DEFAULT_MOE'] = '0'
    os.environ['QWEN3_EXPERT_DROPOUT_P'] = '0'
    from residuals import parse_variant
    parse_variant(variant)
    assert tensor_parallel == pipeline_parallel == expert_parallel == 1
    a = architecture or Architecture()
    kwargs = dict(
        num_layers=a.num_layers, hidden_size=a.hidden_size,
        num_attention_heads=a.num_attention_heads, num_query_groups=a.num_query_groups,
        kv_channels=a.kv_channels, ffn_hidden_size=a.expert_hidden_size,
        moe_ffn_hidden_size=a.expert_hidden_size, num_moe_experts=a.num_experts,
        moe_router_topk=a.topk, moe_layer_freq=1,
        moe_shared_expert_intermediate_size=None,
        normalization='RMSNorm', layernorm_epsilon=a.norm_epsilon,
        qk_layernorm=False, add_bias_linear=False, add_qkv_bias=False,
        gated_linear_unit=True, activation_func=F.silu,
        hidden_dropout=0.0, attention_dropout=0.0,
        moe_router_score_function='softmax', moe_router_pre_softmax=False,
        moe_expert_capacity_factor=None, moe_token_dropping=False,
        moe_pad_expert_input_to_capacity=False, moe_router_enable_expert_bias=False,
        moe_router_num_groups=None, moe_router_group_topk=None,
        moe_input_jitter_eps=None, moe_router_force_load_balancing=False,
        moe_router_load_balancing_type=load_balancing_type,
        moe_aux_loss_coeff=aux_loss_coeff, moe_z_loss_coeff=z_loss_coeff,
        moe_grouped_gemm=use_te, moe_token_dispatcher_type='alltoall' if use_te else 'allgather',
        moe_router_dtype='fp32',
        gradient_accumulation_fusion=False, bias_activation_fusion=False,
        bias_dropout_fusion=False, apply_rope_fusion=use_te,
        init_method_std=0.02,
        tensor_model_parallel_size=tensor_parallel,
        pipeline_model_parallel_size=pipeline_parallel,
        expert_model_parallel_size=expert_parallel,
        sequence_parallel=tensor_parallel > 1,
        bf16=use_te, params_dtype=torch.float32, use_cpu_initialization=not use_te,
    )
    for field in ('moe_dynamic_topk', 'moe_router_logit_noise',
                  'stateful_prototype_router', 'geopath_router'):
        if field in TransformerConfig.__dataclass_fields__:
            kwargs[field] = False
    config = TransformerConfig(**kwargs)
    if os.environ.get('M3_MOE_FUSION', '0') == '1':
        config.moe_permute_fusion = True
        config.moe_router_fusion = True
        config.bias_activation_fusion = True
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te,
                                     normalization='RMSNorm', qk_l2_norm=False)
    from residuals import RMSNorm, ResidualDecoder, attention_class
    from megatron.core.transformer.moe.router import TopKRouter

    class FP32Router(TopKRouter):
        def gating(self, inputs):
            # Native custom backward expects FP32 logits for FP32 routing.
            # Disable the outer BF16 autocast for this explicitly FP32 operation.
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                return super().gating(inputs)

    for layer_spec in spec.layer_specs:
        # Keep norms explicit for the separate query, key and value inputs of MUDDFormer / M3.
        layer_spec.submodules.input_layernorm = RMSNormAdapter
        layer_spec.submodules.pre_mlp_layernorm = RMSNormAdapter
        layer_spec.submodules.self_attention.module = attention_class()
        layer_spec.submodules.mlp.submodules.router = FP32Router
        if use_te:
            from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
    spec.layer_norm = RMSNormAdapter
    # Backbone initialization is identical before variant-specific modules exist.
    model = GPTModel(
        config=config, transformer_layer_spec=spec,
        vocab_size=a.vocab_size, max_sequence_length=a.sequence_length,
        pre_process=pre_process, post_process=post_process,
        share_embeddings_and_output_weights=False,
        position_embedding_type='rope', rotary_percent=1.0,
        rotary_base=a.rotary_base, rope_scaling=False,
    )
    model.decoder = ResidualDecoder(model.decoder, config, variant)
    if os.environ.get('M3_MOE_PACKED','0') == '1':
        from packed_experts import replace_experts
        replace_experts(model)
    return model


from residuals import RMSNorm


class RMSNormAdapter(RMSNorm):
    def __init__(self, config, hidden_size, eps=None, **kwargs):
        super().__init__(hidden_size, config.layernorm_epsilon if eps is None else eps)


if __name__ == '__main__':
    architecture = Architecture()
    print(json.dumps({'architecture': asdict(architecture),
                      'parameters': architecture.parameter_counts()}, indent=2))
