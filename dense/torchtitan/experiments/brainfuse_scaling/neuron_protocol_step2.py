"""Frozen Step 2 v1 model matrix and training recipe.

The scientific definitions come from the 2026-09-25 preregistration; this
module deliberately keeps the Step 1 protocol untouched.
"""

from dataclasses import asdict
import math
import os

from .models import Architecture

SPECS = {
    'S': dict(layers=6, dim=256, heads=4, dense=704,
              ladder=(440, 587, 822), steps=8310, tokens=272302080,
              warmup=83),
    'M': dict(layers=10, dim=384, heads=6, dense=1024,
              ladder=(640, 853, 1195), steps=30518, tokens=1000000000,
              warmup=306),
    'L': dict(layers=16, dim=640, heads=10, dense=1792,
              ladder=(1120, 1493, 2090), steps=140129,
              tokens=4591747072, warmup=1402),
}
CELLS = {
    'W16': 16, 'W32': 32, 'W128': 128, 'W512': 512,
    'Full': 0, 'LA': 0,
}
VARIANTS = ('L0.75', 'L1', 'L1.4', 'T1', 'T4', 'T16', 'T4-null', 'R4')
ALLOWED_CELLS = {
    'S': tuple(CELLS),
    'M': ('W32', 'W128', 'Full', 'LA'),
    'L': ('W32', 'Full'),
}


def parameter_count(a):
    from .models import ffn_projection_parameters, local_parameters
    nonffn = (a.vocab_size*a.dim + a.layers*(4*a.dim*a.dim+2*a.dim)
              + a.dim)
    if a.attention_kind == 'gla':
        nonffn += a.layers*(a.dim*a.dim+a.dim)
    return nonffn + a.layers*(ffn_projection_parameters(a)+local_parameters(a))


def architecture_for(scale, cell, variant, seed=47):
    if scale not in SPECS or cell not in ALLOWED_CELLS[scale] or variant not in VARIANTS:
        raise ValueError((scale, cell, variant))
    if scale == 'L' and variant in ('T1', 'T16'):
        raise ValueError('L has no T1/T16 arm')
    p = SPECS[scale]
    a = Architecture(layers=p['layers'], dim=p['dim'], heads=p['heads'],
                     kv_heads=p['heads'], ff_dim=p['ladder'][1],
                     initialization_seed=seed,
                     initialization_reference_width=p['dense'],
                     attention_window=CELLS[cell],
                     attention_kind='gla' if cell == 'LA' else 'softmax')
    target = parameter_count(a)
    if variant.startswith('L'):
        a.ff_dim = p['ladder'][('L0.75', 'L1', 'L1.4').index(variant)]
        return a
    if variant == 'R4':
        a.r4_scales = round(p['ladder'][1]*4/p['dim'])
    else:
        a.temporal = {'T1': 't1', 'T4': 't4', 'T16': 't16',
                      'T4-null': 'null'}[variant]
    # SwiGLU width changes the projection count by exactly 3*dim per unit.
    # Local state/parameter terms are included in parameter_count.
    approximate = p['ladder'][1]
    width = min(range(max(1, approximate-384), approximate+385),
                key=lambda h: (abs(_count_at(a, h)-target), h))
    a.ff_dim = width
    if abs(parameter_count(a)/target-1) > .001:
        raise AssertionError(f'Parameter mismatch {(scale,cell,variant,width)}')
    return a


def _count_at(a, h):
    a.ff_dim = h
    return parameter_count(a)


def recipe():
    from .config_registry import lf_recipe
    from torchtitan.components.optimizer import ParamGroupConfig
    scale = os.environ.get('NC2_SCALE', 'S')
    cell = os.environ.get('NC2_CELL', 'Full')
    variant = os.environ.get('NC2_VARIANT', 'L1')
    p = SPECS[scale]
    c = lf_recipe('narrow')
    fix_null = variant == 'T4-null-fix'
    c.architecture = architecture_for(scale, cell, 'T4-null' if fix_null else variant,
                                      int(os.environ.get('NC2_SEED', '47')))
    c.debug.seed = c.architecture.initialization_seed
    c.training.steps = p['steps']
    c.token_budget = p['tokens']
    c.lr_scheduler.warmup_steps = p['warmup']
    lr = float(os.environ.get('NC2_LR', '.002'))
    c.optimizer.param_groups[0].optimizer_kwargs['lr'] = lr
    if fix_null:
        if scale != 'M' or cell not in ('W32', 'Full'):
            raise ValueError('T4-null fix is restricted to M/W32 and M/Full')
        other = c.optimizer.param_groups[0]
        c.optimizer.param_groups = [
            ParamGroupConfig(pattern=r'\.feed_forward\.g$', optimizer_name='AdamW',
                             optimizer_kwargs={**other.optimizer_kwargs, 'weight_decay': 0.0}),
            other,
        ]
    c.validator.freq = 831 if scale == 'S' else 3052 if scale == 'M' else 14013
    if scale == 'L':
        c.checkpoint.interval = 512
        c.checkpoint.keep_latest_k = 2
    return c


def audit():
    from .models import model_config, parameter_report, local_training_operations
    import torch
    rows = []
    for scale in SPECS:
        for cell in ALLOWED_CELLS[scale]:
            for variant in VARIANTS:
                if scale == 'L' and variant in ('T1', 'T16'):
                    continue
                a = architecture_for(scale, cell, variant)
                with torch.device('meta'):
                    model = model_config(a).build()
                report = parameter_report(a, model, 2048)
                if report['total_parameters'] != parameter_count(a):
                    raise AssertionError((scale, cell, variant, report['total_parameters'], parameter_count(a)))
                report['nominal_training_flops_per_input_token'] = (
                    report['full_context_training_flops_per_input_token']
                    + 11*a.layers*a.ff_dim
                    + local_training_operations(a, 32768, 4)/32768)
                rows.append(dict(scale=scale, cell=cell, variant=variant,
                                 architecture=asdict(a), **report))
    return rows
