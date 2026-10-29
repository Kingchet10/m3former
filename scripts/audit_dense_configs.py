"""Resolve representative configs with the Dense environment, without training."""
import argparse
import json
import os
from pathlib import Path
import sys

from launch import ROOT, command

os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '1'
sys.path[:0] = [str(ROOT/'dense'), str(ROOT/'third_party/fla-kernels')]
from torchtitan.config import ConfigManager

results=[]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',required=True,help='Existing frozen-data directory; read-only metadata access')
options=parser.parse_args()
for pattern in ['rr100-base*20N*','rr100-mudd*20N*','rr100-dd10*20N*',
                'rr100-blockattnres8*20N*','rr700-base*5N*']:
    path=next((ROOT/'dense/configs').glob(pattern+'.json'))
    cmd,env,_,expected=command(argparse.Namespace(family='dense',config=str(path),
        data_root=options.data_root,runs_root='/example/runs'))
    for key in list(os.environ):
        if key.startswith('SA_'):del os.environ[key]
    os.environ.update({k:v for k,v in env.items() if k.startswith('SA_') or k=='BRAINFUSE_DATA_ROOT'})
    cli=cmd[cmd.index('torchtitan.train')+1:]
    config=json.loads(json.dumps(ConfigManager().parse_args(cli).to_dict()))
    for key in ['training','lr_scheduler','optimizer','architecture']:
        saved=expected['resolved_config'][key]
        actual=dict(config[key])
        if key=='architecture':
            for field in ['hc_streams','attnres_block_size']:
                if field not in saved:
                    assert actual.pop(field)==0, (path.name,field)
        assert actual==saved, (path.name,key,{'actual':actual,'saved':saved})
    results.append({'config':path.name,'resolved_scientific_config':'matches_saved_recipe',
        'additional_disabled_fields_allowed':['hc_streams=0','attnres_block_size=0']})
print(json.dumps({'status':'passed','config_resolution':results,'training_started':False},indent=2))
