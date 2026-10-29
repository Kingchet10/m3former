"""Neuron complexity x temporal communication scaling: model matrix and recipe.

Plan: plans/神经元复杂度×时间通信_Scaling研究_整体方案_v1_2026-09-29.md (v1.1).
All run choices come from SA_* environment variables so one registered config
serves every stage; the scheduler records the environment of each run.
"""
import math
import os
import re

from .models import Architecture

# layers, dim, heads, dense SwiGLU width, local batch (global batch is 16 x 2048)
SCALES = {
    'R0': (4, 192, 3, 528, 16),
    'S': (6, 256, 4, 704, 16),
    'M': (10, 384, 6, 1024, 16),
    'M+': (12, 512, 8, 1408, 16),
    'L': (16, 640, 10, 1792, 8),
    'XL': (24, 1024, 16, 2816, 4),
    # Routed Residuals extension (collaborator runs): same depth/width ratio (~0.024) and FFN ~2.75 x dim
    'XXL': (28, 1152, 18, 3200, 2),
    'XXXL': (32, 1280, 20, 3520, 2),
    'XXXXL': (40, 1664, 26, 4608, 4),
}
# W8/W16/W64: baseline-only windows that refine the equivalent-window (memory exchange rate) interpolation
WINDOWS = {'W4': 4, 'W8': 8, 'W16': 16, 'W32': 32, 'W64': 64, 'W128': 128, 'Full': 0}
DYNAMICS = ('m1', 'c1', 'c2')
DENDRITES = {'dfull': 'full', 'duniform': 'uniform', 'dlearned': 'learned', 'dlinear': 'linear'}
SPARSE = {'sp_point': 'point', 'sp_dend': 'dend', 'sp_dlin': 'dlin', 'sp_wide': 'point', 'sp_dplat': 'dplat',
          'sp_dsq': 'dsq', 'sp_ndend': 'ndend', 'sp_nsq': 'nsq'}  # spatial axis; sp_n*: non-gated neurons
# depth dendrites: compartments (channel groups) with own source weights; mudd_static / vemb are ablations
DEPTH = {'mudd': (1, 'dense'), 'dd4': (4, 'dense'), 'dd16': (16, 'dense'), 'mudd_static': (1, 'static'),
         'vemb': (1, 'vemb'), 'dd64': (64, 'dense'), 'mudd_wide': (1, 'dense', 4),  # mudd_wide: params ~ dd64
         'mudd_nrm': (1, 'dense'),
         'sw1': (1, 'dense1'), 'sw4': (4, 'dense1'), 'sw16': (16, 'dense1')}  # single-way, G subspaces (LDSD Q-D)
MIX_NORM = {'mudd_nrm': 'l1'}  # bounded MUDD: mixing weights normalised over sources (LDSD Q-B)
TOKENS_PER_PARAM = 20
BATCH_TOKENS = 16 * 2048


def parameter_count(a):
    from .models import depth_parameters, ffn_projection_parameters, local_parameters
    return (a.vocab_size * a.dim + a.layers * (4 * a.dim * a.dim + 2 * a.dim) + a.dim
            + a.layers * (ffn_projection_parameters(a) + local_parameters(a)) + depth_parameters(a))


def nonembedding(a):
    return parameter_count(a) - a.vocab_size * a.dim


def _base(scale, window, seed):
    layers, dim, heads, width, _ = SCALES[scale]
    return Architecture(layers=layers, dim=dim, heads=heads, kv_heads=heads, ff_dim=width,
                        initialization_seed=seed, initialization_reference_width=width,
                        attention_window=WINDOWS[window])


def architecture_for(scale, window, variant, seed=47, placement='gate', fanin=1.0):
    a = _base(scale, window, seed)
    if variant == 'hc4':
        a.hc_streams = 4
        return a
    if variant == 'base':
        return a
    if variant not in DEPTH:  # generic granularity: ddG (four-way) / swG (single-way), G must divide dim
        m = re.fullmatch(r'(dd|sw)(\d+)', variant)
        if m:
            DEPTH[variant] = (int(m.group(2)), 'dense' if m.group(1) == 'dd' else 'dense1')
    if variant in DEPTH:
        # Depth dendrites: same widths as the baseline; mixer parameters counted as built
        a.depth_groups, a.depth_mode, *rest = DEPTH[variant]
        a.depth_hidden_mult = rest[0] if rest else 1
        a.depth_mix_norm = MIX_NORM.get(variant, '')
        return a
    if variant == 'blockattnres8':
        a.attnres, a.attnres_block_size = True, 8
        return a
    if variant == 'attnres':  # Full Attention Residuals; pseudo-queries and key-norm gains counted as built
        a.attnres = True
        return a
    m = re.fullmatch(r'mhar(\d+)', variant)
    if m:  # Multi-Head Attention Residuals with H routing heads (H = number of KV heads in the paper)
        a.attnres, a.attnres_heads = True, int(m.group(1))
        return a
    if variant == 'dca':  # DeepCrossAttention (GRN-v3 for queries / keys / values and the output)
        a.dca = True
        return a
    if variant in SPARSE:
        # Spatial axis: same widths as the baseline; parameters are counted as real synapses.
        # sp_wide gets its 4x width and fan-in / 4 from SA_FF and SA_FANIN in the job.
        a.sparse_kind, a.fanin, a.mask_seed = SPARSE[variant], float(fanin), seed
        return a
    target = parameter_count(a)
    if variant in DYNAMICS:
        a.dynamics, a.placement = variant, placement
    elif variant in ('alif', 'adex', 'hh'):
        a.biophys = variant
    elif variant == 'rmodes':  # residual-stream control with the c2 state budget (4 real states per neuron)
        a.residual_modes = max(1, round(2 * a.ff_dim / a.dim))
    elif variant in DENDRITES:
        a.dendrite_kind, a.dendrite_branches = DENDRITES[variant], 4
    elif variant == 'T4':
        if window not in ('W32', 'W128', 'Full'):
            raise ValueError('Legacy T4 control is defined for W32/W128/Full only')
        a.temporal = 't4'
    else:
        raise ValueError(variant)
    width = SCALES[scale][3]
    a.ff_dim = min(range(max(1, width // 3), width + 257),
                   key=lambda h: (abs(_count_at(a, h) - target), h))
    if abs(parameter_count(a) / target - 1) > .001:
        raise AssertionError(f'Parameter mismatch {(scale, window, variant, a.ff_dim)}')
    return a


def apply_overrides(a, e):
    """Job-level shape overrides (also used by the analysis to rebuild a run's architecture from its environment).

    SA_FF: explicit FFN width (shape-matched comparisons); parameters are counted as built.
    SA_SHAPE='layers,dim': depth x width at the scale's token budget (E1); heads of width 64, initialization
    reference width = the FFN width (no down-projection gain).
    SA_RESIDUAL_MULT=k: residual stream of width k * dim around width-dim blocks (E2); the baseline becomes
    depth-dendrite blocks without mixing ('plain').
    SA_LNS=1: LayerNorm Scaling in every block (LDSD Q-C); the baseline becomes 'plain' depth-dendrite blocks."""
    if 'SA_SHAPE' in e:
        a.layers, a.dim = (int(x) for x in str(e['SA_SHAPE']).split(','))
        a.heads = a.kv_heads = a.dim // 64
    if 'SA_FF' in e:
        a.ff_dim = int(e['SA_FF'])
    if 'SA_SHAPE' in e:
        a.initialization_reference_width = a.ff_dim
    k = int(e.get('SA_RESIDUAL_MULT', 1))
    if k != 1:
        if k < 1 or a.depth_mode == 'vemb':
            raise ValueError(f'residual multiplier {k} with {a.depth_mode}')
        a.residual_mult = k
        if not a.depth_groups:
            a.depth_groups, a.depth_mode = 1, 'plain'
    if str(e.get('SA_LNS', '0')) == '1':
        if a.attnres:
            raise ValueError('SA_LNS is defined for depth-dendrite blocks only')
        a.depth_lns = True
        if not a.depth_groups:
            a.depth_groups, a.depth_mode = 1, 'plain'
    return a


def _count_at(a, h):
    a.ff_dim = h
    return parameter_count(a)


def budget(scale, ratio=TOKENS_PER_PARAM):
    tokens = int(round(ratio * nonembedding(_base(scale, 'Full', 47))))
    return tokens, math.ceil(tokens / BATCH_TOKENS)


def gpt2_anchor_recipe(e):
    """llm.c GPT-2 (124M) budget on GPT-2-tokenized FineWeb-10BT, with our Llama-style block."""
    import math as _m
    from .config_registry import lf_recipe
    c = lf_recipe('narrow')
    c.architecture = Architecture(layers=12, dim=768, heads=12, kv_heads=12, ff_dim=2048, vocab_size=50304,
                                  initialization_seed=int(e.get('SA_SEED', '47')),
                                  initialization_reference_width=2048, attention_window=0)
    c.debug.seed = c.architecture.initialization_seed
    c.training.seq_len = 1024
    c.training.global_batch_size = 512
    c.training.local_batch_size = int(e.get('SA_LOCAL_BATCH', '32'))
    tokens = int(e.get('SA_TOKENS', 9_900_000_000))
    c.token_budget = tokens
    c.training.steps = _m.ceil(tokens / (512 * 1024))
    c.lr_scheduler.warmup_steps = 700
    c.optimizer.param_groups[0].optimizer_kwargs['lr'] = float(e.get('SA_LR', '6e-4'))
    c.validator.freq = max(1, c.training.steps // 4)
    c.checkpoint.interval = max(1, _m.ceil(c.training.steps / 8))
    c.checkpoint.keep_latest_k = 2
    c.debug.deterministic = False
    c.audit_batches = False
    if 'SA_STOP_AFTER' in e:
        c.stop_after_step = int(e['SA_STOP_AFTER'])
        c.checkpoint.enable = False
    return c


def recipe():
    from .config_registry import lf_recipe
    if os.environ.get('SA_SCALE') == 'GPT2':
        return gpt2_anchor_recipe(os.environ)
    from torchtitan.components.optimizer import ParamGroupConfig
    e = os.environ
    scale, window, variant = e['SA_SCALE'], e['SA_WINDOW'], e['SA_VARIANT']
    seed, lr = int(e.get('SA_SEED', '47')), float(e['SA_LR'])
    c = lf_recipe('narrow')
    c.architecture = architecture_for(scale, window, variant, seed, e.get('SA_PLACEMENT', 'gate'),
                                      float(e.get('SA_FANIN', '1')))
    if 'SA_BRANCHES' in e:
        c.architecture.dendrite_branches = int(e['SA_BRANCHES'])
    if 'SA_WIRING' in e:  # spatial axis: 'learned' = the synapse placement is learned (top-k scores, straight-through)
        c.architecture.wiring = e['SA_WIRING']
    apply_overrides(c.architecture, e)
    if e.get('SA_CORPUS_FORMAT') == 'megatron_document_pairs_v1':
        from .artifacts import read_manifest
        from .tinyllama_tokenizer import TinyLlamaTokenizer
        manifest = read_manifest(c.dataloader.dataset_path)
        if manifest.get('format') != 'megatron_document_pairs_v1' or manifest['tokenizer']['vocab_size'] != 32000:
            raise ValueError('Expected a bound TinyLlama 32K Megatron corpus')
        if manifest.get('source_binary_hashes_verified') is not True:
            raise ValueError('Verify source binary hashes before training')
        c.architecture.vocab_size = 32000
        c.tokenizer = TinyLlamaTokenizer.Config()
    elif e.get('SA_CORPUS_FORMAT'):
        raise ValueError('Unknown SA_CORPUS_FORMAT')
    c.debug.seed = seed
    tokens, steps = budget(scale, float(e.get('SA_TOKENS_PER_PARAM', TOKENS_PER_PARAM)))
    c.token_budget, c.training.steps = tokens, steps
    c.training.local_batch_size = int(e.get('SA_LOCAL_BATCH', SCALES[scale][4]))
    c.lr_scheduler.warmup_steps = max(math.ceil(steps / 100), 100)
    other = c.optimizer.param_groups[0]
    other.optimizer_kwargs['lr'] = lr
    if variant == 'hc4':
        static_kwargs = dict(other.optimizer_kwargs, weight_decay=0.0)
        c.optimizer.param_groups = [
            ParamGroupConfig(pattern=r'\.hc_(attn|ffn)\.static_(alpha|beta)$',
                             optimizer_name='AdamW', optimizer_kwargs=static_kwargs), other]
    if variant in DYNAMICS or variant in ('rmodes', 'alif', 'adex', 'hh', 'sp_dend', 'sp_dplat', 'sp_dsq', 'sp_ndend'):
        kwargs = dict(other.optimizer_kwargs)
        kwargs['lr'] = lr * float(e.get('SA_STATE_LR_MULT', '1'))
        if 'SA_STATE_WD' in e:
            kwargs['weight_decay'] = float(e['SA_STATE_WD'])
        c.optimizer.param_groups = [
            ParamGroupConfig(pattern=r'(\.feed_forward\.|\.rm_)(nu|theta|c_re|c_im|bp|readout|gamma)$',
                             optimizer_name='AdamW', optimizer_kwargs=kwargs),
            other,
        ]
    c.validator.freq = max(1, steps // 2)
    c.checkpoint.interval = int(e.get('SA_CKPT_INTERVAL', max(1, math.ceil(steps / 4))))  # frequent for resumable runs
    c.checkpoint.keep_latest_k = 2
    if e.get('SA_EXACT', '0') != '1':
        c.debug.deterministic = False
        c.audit_batches = False
    if e.get('SA_COMPILE', '0') == '1':  # per-block torch.compile for speed; numerics differ slightly from eager
        c.compile.enable = True
    if 'SA_STOP_AFTER' in e:
        c.stop_after_step = int(e['SA_STOP_AFTER'])
        c.checkpoint.enable = False
    return c


def matrix():
    rows = []
    for scale in SCALES:
        tokens, steps = budget(scale)
        for window in WINDOWS:
            for variant in ('base', *DYNAMICS, *DENDRITES, 'rmodes', 'alif', 'adex', 'hh'):
                a = architecture_for(scale, window, variant)
                rows.append(dict(scale=scale, window=window, variant=variant, ff_dim=a.ff_dim,
                                 total=parameter_count(a), nonembedding=nonembedding(a),
                                 tokens=tokens, steps=steps))
    return rows
