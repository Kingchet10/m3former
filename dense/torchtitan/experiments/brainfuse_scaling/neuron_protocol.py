"""Frozen Step 1 matrix, exact integer width audit, and training recipe."""
from dataclasses import asdict
from .models import Architecture
SPECS = {'S': (6,256,4,704,[440,587,704,822],8310,272302080,83),
         'M': (10,384,6,1024,[640,853,1024,1195],30518,1000000000,306)}
CELLS = ('C1','C2','C3','C4')
VARIANTS = ('L0.75','L1','L1.2','L1.4','B2','B4','B2-lin','T4','T4-null')

def count(a):
    from .models import ffn_projection_parameters, local_parameters
    nonffn = a.vocab_size*a.dim + a.layers*(4*a.dim*a.dim+2*a.dim) + a.dim
    return nonffn + a.layers*(ffn_projection_parameters(a)+local_parameters(a))

def architecture_for(scale,cell,variant,seed=47):
    assert cell in CELLS and variant in VARIANTS
    layers,dim,heads,dense,ladder,steps,tokens,warmup=SPECS[scale]
    gelu=cell in ('C2','C4')
    a=Architecture(layers=layers,dim=dim,heads=heads,kv_heads=heads,
                   ff_dim=round(ladder[1]*1.5) if gelu else ladder[1],
                   initialization_seed=seed,initialization_reference_width=dense,
                   activation='gelu' if gelu else 'swiglu',attention_window=32 if cell in ('C3','C4') else 0)
    target=count(a)
    if variant.startswith('L'):
        a.ff_dim=round(ladder[VARIANTS.index(variant)]*1.5) if gelu else ladder[VARIANTS.index(variant)]
    else:
        if variant.startswith('B'):
            b=4 if variant=='B4' else 2
            a.ffn_kind='lf_sum' if variant=='B2-lin' else 'lf_safe_rational'
            a.output_compression=a.local_group_size=b
            a.local_outputs=1
        else:
            b=1
            a.temporal='null' if variant=='T4-null' else 't4'
        choices=[]
        for h in range(b,4*dense+1,b):
            a.ff_dim=h
            choices.append((abs(count(a)-target),h))
        a.ff_dim=min(choices)[1]
        # Explicit frozen M-C1 branches override nearest rounding (§4/§12).
        if scale=='M' and not gelu and variant in ('B2','B2-lin','B4'):
            a.ff_dim=1136 if variant=='B4' else 1024
        assert abs(count(a)/target-1)<=.001,(scale,cell,variant,a.ff_dim)
    return a

def recipe():
    import os
    from .config_registry import lf_recipe
    scale=os.environ.get('NC_SCALE','S');cell=os.environ.get('NC_CELL','C1');variant=os.environ.get('NC_VARIANT','L1')
    c=lf_recipe('narrow')
    c.architecture=architecture_for(scale,cell,variant,int(os.environ.get('NC_SEED','47')))
    c.debug.seed=c.architecture.initialization_seed
    c.training.steps=SPECS[scale][5]
    c.token_budget=SPECS[scale][6]
    c.lr_scheduler.warmup_steps=SPECS[scale][7]
    c.optimizer.param_groups[0].optimizer_kwargs['lr']=float(os.environ.get('NC_LR','.003'))
    c.validator.freq=831 if scale=='S' else 3052
    return c

def audit():
    from .models import model_config,parameter_report,local_training_operations
    import torch
    rows=[]
    for scale in SPECS:
        for cell in CELLS:
            for variant in VARIANTS:
                a=architecture_for(scale,cell,variant)
                with torch.device('meta'):
                    m=model_config(a).build()
                p=parameter_report(a,m,2048)
                assert p['total_parameters']==count(a),(scale,cell,variant,p,count(a))
                ops=11 if a.activation=='swiglu' else 13
                p['nominal_training_flops_per_input_token']=p['full_context_training_flops_per_input_token']+ops*a.layers*a.ff_dim+local_training_operations(a,32768,4)/32768
                p['activation_counting']='SwiGLU 11 basic ops; GeLU 13 basic ops; special functions separate; backward 2x forward for EMA/local map, same scoped convention'
                rows.append(dict(scale=scale,cell=cell,variant=variant,architecture=asdict(a),**p))
    return rows
