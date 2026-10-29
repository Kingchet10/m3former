"""Frozen two-seed, fixed-budget B x rank matrix (28 new runs)."""
WIDTHS = {2: {0:1024,4:1018,8:1012,16:1000},
          4: {0:1136,4:1132,8:1124,16:1112},
          8: {0:1208,4:1200,8:1192,16:1176}}
MODELS = {f'grid_b{b}_r{r}':(b,r,'rational',h) for b,items in WIDTHS.items() for r,h in items.items()}
MODELS.update({f'grid_b{b}_silu8':(b,8,'silu',items[8]) for b,items in WIDTHS.items()})
NEW_MODELS = tuple(n for n in MODELS if n != 'grid_b2_r0')
CONFIGS = tuple((s,n) for s in (47,48) for n in NEW_MODELS)


def recipe(name):
    from .config_registry import lf_recipe
    b,r,kind,h = MODELS[name]
    config = lf_recipe('safe_rational')
    a=config.architecture
    a.ff_dim=h;a.local_group_size=b;a.output_compression=b
    a.ridge_rank=r;a.ridge_kind=kind
    return config
