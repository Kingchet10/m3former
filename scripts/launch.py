"""Portable entry point. No scheduler submission or GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    config = Path(args.config).resolve()
    data = Path(args.data_root).resolve()
    runs = Path(args.runs_root).resolve()
    r = json.loads(config.read_text())
    env = os.environ.copy()
    # Do not inherit a previous experiment's architecture/recipe overrides.
    for key in list(env):
        if key.startswith('SA_'):
            del env[key]
    env.update(FLA_DISABLE_BACKEND_DISPATCH='1', OMP_NUM_THREADS='4',
               TORCHINDUCTOR_COMPILE_THREADS='1')
    if args.family == 'dense':
        root = ROOT / 'dense'
        replacements = {'${DATA_ROOT}': str(data), '${RUNS_ROOT}': str(runs), '${DENSE_ROOT}': str(root)}
        def expand(s):
            for k, v in replacements.items():
                s = s.replace(k, v)
            if '${' in s:
                raise ValueError(f'Unresolved path placeholder: {s}')
            return s
        cli = [expand(s) for s in r['args']]
        env.update({k: expand(v) for k, v in r['recipe_env'].items()})
        env['PYTHONPATH'] = os.pathsep.join(map(str, [root / 'runtime_hooks', root, ROOT / 'third_party/fla-kernels']))
        nproc = int(cli[cli.index('--parallelism.data_parallel_replicate_degree') + 1])
        # All branches use the deployed RNG-preserving audit wrapper. This does
        # not alter the optimizer/schedule or relax restoration assertions.
        if int(r['budget_N']) != 20:
            cli[cli.index('--module') + 1] = 'rr_wsd_rng_fix.config'
        output = Path(cli[cli.index('--dump-folder') + 1])
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={nproc}', '-m', 'torchtitan.train', *cli]
        resolved = r
    else:
        root = ROOT / 'moe'
        hidden=r['architecture']['hidden_size']
        if hidden not in (1024,1536):
            raise ValueError('This export provides the fixed small/large MoE recipes only')
        import_roots=([root/'variants/small'] if hidden==1024 else []) + [root]
        r['data_root'] = str(data)
        nproc = 8
        output = runs / f'{config.parent.name}-{config.stem}'
        target = output / 'launch-config.json'
        env.update(QWEN3_DEFAULT_MOE='0', QWEN3_EXPERT_DROPOUT_P='0')
        env['PYTHONPATH'] = os.pathsep.join(map(str, import_roots + [ROOT / 'third_party/Megatron-LM', ROOT / 'third_party/fla-kernels']))
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={nproc}', '-m', 'train',
               '--config', str(target), '--output', str(output)]
        resolved = r
    return cmd, env, output, resolved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family', choices=['dense', 'moe'])
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--runs-root', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cmd, env, output, resolved = command(a)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        print(json.dumps({'output': str(output), 'environment': {k: v for k, v in env.items()
            if k in ['PYTHONPATH', 'BRAINFUSE_DATA_ROOT', 'FLA_DISABLE_BACKEND_DISPATCH'] or k.startswith('SA_')}}, indent=2))
        return
    if not (Path(a.data_root) / 'DATASET_READY').is_file():
        raise FileNotFoundError('Expected the frozen document-pair data manifest and DATASET_READY; see docs/DATA.md')
    if (output / 'config.json').exists():
        raise FileExistsError(f'Existing run at {output}; use a fresh runs root. This launcher does not overwrite/resume live runs.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'launch-config.json').write_text(json.dumps(resolved, indent=2) + '\n')
    manifest = ROOT / 'provenance/SHA256SUMS.json'
    (output / 'export-launch.json').write_text(json.dumps({
        'command': cmd, 'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_manifest': json.loads(manifest.read_text()), 'config': str(Path(a.config).resolve()),
        'family': a.family}, indent=2) + '\n')
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
