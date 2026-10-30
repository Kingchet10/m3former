"""Model sizes, methods and the training recipe of the M3Former paper.

One registered config (`config_registry.wsd`) serves every run. The model size, the method and the few per-run
choices come from M3_* environment variables, which `scripts/launch.py` sets from the command line:

    M3_SIZE       100M | 300M | 500M | 700M | 1.4B
    M3_METHOD     transformer | muddformer | m3-g<G> | multigroup-g<G> | attnres-block | m3-static-g<G>
    M3_LR         peak learning rate (default: the preset of the size, shared by all methods)
    M3_SEED       seed of the parameter initialization and of the run (default 47)
    M3_DATA_ROOT  frozen corpus directory (see docs/DATA.md)
    M3_LOCAL_BATCH, M3_TOKENS_PER_PARAM, M3_CKPT_INTERVAL, M3_COMPILE (1 = per-block torch.compile)

G is the number of groups per role; the paper uses G = H, 2H and 4H, where H = d / 64 is the number of attention
heads (groups of width 64, 32 and 16). The 5N and 10N runs branch from checkpoints of the 20N run (wsd_runtime).
"""
import math
import os
import re
from pathlib import Path

# layers, width, attention heads (width 64), SwiGLU width, peak learning rate
SIZES = {
    '100M': (16, 640, 10, 1792, 7.6e-4),
    '300M': (24, 1024, 16, 2816, 4.55e-4),
    '500M': (28, 1152, 18, 3200, 3.2e-4),
    '700M': (32, 1280, 20, 3520, 2.6e-4),
    '1.4B': (40, 1664, 26, 4608, 1.9e-4),
}
TOKENS_PER_PARAM = 20
SEQ_LEN = 2048
GLOBAL_BATCH = 64  # sequences
VALIDATION_SEQUENCES = 48_768  # 99,876,864 held-out tokens
ATTNRES_BLOCK_SUBLAYERS = 8  # block attention residuals: blocks of four layers (eight sublayers)


def architecture_for(size, method, seed=47):
    from .models import Architecture

    layers, dim, heads, width, _ = SIZES[size]
    a = Architecture(layers=layers, dim=dim, heads=heads, kv_heads=heads, ff_dim=width,
                     initialization_seed=seed, initialization_reference_width=width)
    if method == 'transformer':
        return a
    if method == 'muddformer':
        a.depth_groups, a.depth_mode = 1, 'dense'
        return a
    if method == 'attnres-block':
        a.attnres, a.attnres_block_size = True, ATTNRES_BLOCK_SUBLAYERS
        return a
    m = re.fullmatch(r'(m3|multigroup|m3-static)-g(\d+)', method)
    if m is None:
        raise ValueError(f'Unknown method: {method}')
    a.depth_groups = int(m.group(2))
    a.depth_mode = {'m3': 'dense', 'multigroup': 'dense1', 'm3-static': 'static'}[m.group(1)]
    if dim % a.depth_groups:
        raise ValueError(f'G = {a.depth_groups} does not divide the width {dim}')
    return a


def transformer_nonembedding(size):
    """N: attention and SwiGLU weights and the RMSNorm gains of the Transformer (embedding excluded)."""
    layers, dim, _, width, _ = SIZES[size]
    return layers * (4 * dim * dim + 2 * dim + 3 * dim * width) + dim


def budget(size, tokens_per_param=TOKENS_PER_PARAM):
    """Training tokens and optimizer steps; N is the non-embedding parameter count of the Transformer."""
    tokens = int(round(tokens_per_param * transformer_nonembedding(size)))
    return tokens, math.ceil(tokens / (GLOBAL_BATCH * SEQ_LEN))


def schedule(size, budget_n):
    """WSD schedule of one budget: warmup over 1% of the 20N run; the last 20% of each budget decays."""
    _, steps20 = budget(size, TOKENS_PER_PARAM)
    tokens, steps = budget(size, budget_n)
    return dict(tokens=tokens, steps=steps, warmup=math.ceil(steps20 / 100), tail=math.ceil(0.2 * steps))


def recipe():
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.components.optimizer import default_adamw
    from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig

    from .artifacts import read_manifest
    from .loader import FrozenTokenLoader
    from .tinyllama_tokenizer import TinyLlamaTokenizer
    from .trainer import ResearchMetrics
    from .validation import ResearchValidator
    from .wsd_runtime import WSDSchedulers, WSDTrainer

    e = os.environ
    size, method = e.get('M3_SIZE', '100M'), e.get('M3_METHOD', 'transformer')
    seed = int(e.get('M3_SEED', '47'))
    lr = float(e.get('M3_LR', SIZES[size][4]))
    root = Path(e.get('M3_DATA_ROOT', './data'))
    s = schedule(size, float(e.get('M3_TOKENS_PER_PARAM', TOKENS_PER_PARAM)))
    a = architecture_for(size, method, seed)
    if (root / 'manifest.json').exists():
        manifest = read_manifest(str(root))
        if manifest.get('format') != 'megatron_document_pairs_v1' or manifest['tokenizer']['vocab_size'] != 32000:
            raise ValueError('Expected a TinyLlama 32K corpus in megatron_document_pairs_v1 format')
        if manifest.get('source_binary_hashes_verified') is not True:
            raise ValueError('Verify the source binary hashes of the corpus before training')
    local_batch = int(e.get('M3_LOCAL_BATCH', '4'))
    optimizer = default_adamw(lr=lr)
    return WSDTrainer.Config(
        architecture=a,
        hf_assets_path=str(root / 'tokenizer'),
        tokenizer=TinyLlamaTokenizer.Config(),
        dataloader=FrozenTokenLoader.Config(dataset_path=str(root), pin_memory=True),
        loss=CrossEntropyLoss.Config(global_vocab_size=a.vocab_size),
        optimizer=optimizer,
        lr_scheduler=WSDSchedulers.Config(warmup_steps=s['warmup'], total_steps=s['steps'], tail_steps=s['tail'],
                                          min_lr_factor=0.1, decay_type='sqrt', decay_ratio=0.2),
        training=TrainingConfig(local_batch_size=local_batch, global_batch_size=GLOBAL_BATCH, seq_len=SEQ_LEN,
                                steps=s['steps'], dtype='float32', mixed_precision_param='bfloat16',
                                disable_cuda_graphs=True),
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        activation_checkpoint=None,
        compile=CompileConfig(enable=e.get('M3_COMPILE', '0') == '1'),
        checkpoint=CheckpointManager.Config(enable=True, interval=int(e.get('M3_CKPT_INTERVAL', 2**31 - 1)),
                                            keep_latest_k=2, last_save_model_only=False),
        metrics=ResearchMetrics.Config(log_freq=1, disable_color_printing=True, enable_tensorboard=True),
        validator=ResearchValidator.Config(
            enable=True, freq=max(1, s['steps'] // 2), steps=-1,
            dataloader=FrozenTokenLoader.Config(dataset_path=str(root), split='validation', pin_memory=True)),
        debug=DebugConfig(seed=seed, save_config_file='config.json', deterministic=False),
        token_budget=s['tokens'],
        paired_initialization=True,
    )
