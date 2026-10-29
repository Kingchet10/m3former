"""CPU-only packaging/configuration checks; does not claim GPU correctness."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    pyfiles = list(ROOT.rglob('*.py'))
    for p in pyfiles:
        ast.parse(p.read_text(), filename=str(p))
    source = json.loads((ROOT / 'provenance/source-files.json').read_text())
    for item in source:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    dc = list((ROOT / 'dense/configs').glob('*.json'))
    for p in dc:
        d = json.loads(p.read_text()); c = d['resolved_config']
        assert c['training']['global_batch_size'] == 64
        assert c['training']['seq_len'] == 2048
        assert c['lr_scheduler']['min_lr_factor'] == .1
        assert c['lr_scheduler']['warmup_steps'] == d['warmup']
        assert c['lr_scheduler']['tail_steps'] == d['tail_steps']
        assert d['budget_N'] in [5, 10, 20]
    mc = list((ROOT / 'moe/configs').glob('*/*.json'))
    for p in mc:
        d = json.loads(p.read_text())
        assert d['global_batch_sequences'] == 128 and d['sequence_length'] == 2048
        assert d['peak_lr'] == .0003 and d['schedule'] == 'cosine'
        assert d['train_tokens'] == d['train_steps'] * 128 * 2048
        assert d['architecture']['hidden_size'] % (d.get('residual_groups') or 1) == 0
    # Evaluate the actual scheduler function without importing the GPU runtime.
    source_path = ROOT / 'dense/torchtitan/experiments/brainfuse_scaling/wsd_runtime.py'
    tree = ast.parse(source_path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'factor')
    ns = {'math': __import__('math')}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source_path), 'exec'), ns)
    factor = ns['factor']
    for p in dc:
        d = json.loads(p.read_text()); t, w, tail = d['steps'], d['warmup'], d['tail_steps']
        assert abs(factor(0, w, t, tail) - 1/w) < 1e-12
        assert factor(w-1, w, t, tail) == 1
        assert factor(t-tail, w, t, tail) == 1
        assert abs(factor(t-1, w, t, tail)-.1) < 1e-12
    # Check the new command wrapper for every recipe, without subprocess launch.
    spec=importlib.util.spec_from_file_location('export_launch',ROOT/'scripts/launch.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    from argparse import Namespace
    for family, configs in [('dense',dc),('moe',mc)]:
        for p in configs:
            cmd,env,out,resolved=mod.command(Namespace(family=family, config=str(p),
                data_root='/example/data', runs_root='/example/runs'))
            assert '${' not in ' '.join(cmd)
            if family=='dense' and resolved['budget_N']!=20:
                assert cmd[cmd.index('--module')+1]=='rr_wsd_rng_fix.config'
    print(json.dumps({'status':'passed','python_files_parsed':len(pyfiles),
        'source_hashes_verified':len(source),'dense_configs_checked':len(dc),
        'moe_configs_checked':len(mc),'scheduler_boundaries':'passed',
        'portable_command_expansion':'passed','gpu_training_tested':False},indent=2))


if __name__=='__main__':main()
