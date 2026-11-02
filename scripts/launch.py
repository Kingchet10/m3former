"""Launch one training run on a single node (torch.distributed.run). No scheduler, no GPU allocation.

Dense (TorchTitan):

    python scripts/launch.py dense --size 700M --method m3-g80 --budget 20N --gpus 4 \
        --data-root /path/to/corpus --runs-root /path/to/runs

    methods: transformer | muddformer | m3-g<G> | multigroup-g<G> | attnres-block | m3-static-g<G>

    A 20N run saves the checkpoints from which the 5N and 10N runs of the same model branch; launch it first and
    pass the same --runs-root to the branches. `--paper` lists the dense runs of the paper.

MoE (Megatron Core):

    python scripts/launch.py moe --config moe/configs/small/m3-g64.json \
        --data-root /path/to/corpus --runs-root /path/to/runs

`--dry-run` prints the command and the environment without starting anything.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dense'))

BUDGETS = (5, 10, 20)
LOCAL_BATCH = 4
NO_CHECKPOINT = 2**31 - 1  # periodic checkpoints off; WSD anchors and the final step are still saved


def run_name(size, method, budget):
    return f'{size.lower()}-{method}-{budget}n'


def dense_command(size, method, budget, gpus, data, runs, lr=None, seed=47):
    """Command line and environment of one dense run, as used for the paper."""
    from m3former.recipe import SIZES, TOKENS_PER_PARAM, VALIDATION_SEQUENCES, schedule

    if budget not in BUDGETS:
        raise ValueError(f'budget must be one of {BUDGETS} tokens per parameter')
    if VALIDATION_SEQUENCES % (gpus * LOCAL_BATCH):
        raise ValueError('the validation split must divide over the data-parallel ranks')
    s = schedule(size, budget)
    name = run_name(size, method, budget)
    out = Path(runs) / name
    cli = ['--module', 'm3former', '--config', 'wsd',
           '--dump-folder', str(out), '--tracking-dir', str(out / 'research-tensorboard'),
           '--dataloader.no-verify-files', '--validator.dataloader.no-verify-files',
           '--parallelism.data_parallel_replicate_degree', str(gpus),
           '--parallelism.spmd_backend', 'partial_dtensor',
           '--training.global_batch_size', '64', '--training.local_batch_size', str(LOCAL_BATCH),
           '--token-budget', str(s['tokens']), '--training.steps', str(s['steps']),
           '--lr-scheduler.total_steps', str(s['steps']), '--lr-scheduler.warmup_steps', str(s['warmup']),
           '--lr-scheduler.tail_steps', str(s['tail']),
           '--lr-scheduler.decay_type', 'sqrt', '--lr-scheduler.min_lr_factor', '0.1',
           '--validator.freq', str(s['steps'] // 2),
           '--validator.steps', str(VALIDATION_SEQUENCES // (gpus * LOCAL_BATCH)),
           '--checkpoint.interval', str(NO_CHECKPOINT), '--checkpoint.keep_latest_k', '4']
    if budget == TOKENS_PER_PARAM:
        anchors = []
        for b in BUDGETS:
            if b != budget:
                t = schedule(size, b)
                anchors.append(t['steps'] - t['tail'])
        cli += ['--wsd-anchor-steps', *map(str, anchors)]
    else:
        parent = Path(runs) / run_name(size, method, TOKENS_PER_PARAM)
        restore = s['steps'] - s['tail']
        cli += ['--wsd-parent-run', str(parent), '--wsd-restore-step', str(restore),
                '--checkpoint.initial_load_path', str(parent / 'checkpoint' / f'step-{restore}'),
                '--checkpoint.no-initial-load-model-only', '--wsd-preserve-rng-audit']
    env = {'M3_DATA_ROOT': str(data), 'M3_SIZE': size, 'M3_METHOD': method, 'M3_SEED': str(seed),
           'M3_LR': repr(SIZES[size][4] if lr is None else lr), 'M3_TOKENS_PER_PARAM': str(TOKENS_PER_PARAM),
           'M3_LOCAL_BATCH': str(LOCAL_BATCH), 'M3_CKPT_INTERVAL': str(NO_CHECKPOINT), 'M3_COMPILE': '1',
           'FLA_DISABLE_BACKEND_DISPATCH': '1', 'OMP_NUM_THREADS': '4', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
           'PYTHONPATH': os.pathsep.join(map(str, [ROOT / 'dense', ROOT / 'third_party/fla-kernels']))}
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}',
           '-m', 'torchtitan.train', *cli]
    return cmd, env, out


def moe_command(config, data, runs, gpus=8):
    r = json.loads(Path(config).read_text())
    r['data_root'] = str(data)
    out = Path(runs) / f'moe-{Path(config).parent.name}-{Path(config).stem}'
    env = {'QWEN3_DEFAULT_MOE': '0', 'QWEN3_EXPERT_DROPOUT_P': '0',  # optional router extensions of the snapshot
           'FLA_DISABLE_BACKEND_DISPATCH': '1', 'OMP_NUM_THREADS': '4', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
           'PYTHONPATH': os.pathsep.join(map(str, [ROOT / 'moe', ROOT / 'third_party/Megatron-LM',
                                                   ROOT / 'third_party/fla-kernels']))}
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}',
           '-m', 'train', '--config', str(out / 'launch-config.json'), '--output', str(out)]
    return cmd, env, out, r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('family', choices=['dense', 'moe'])
    p.add_argument('--size', choices=['100M', '300M', '500M', '700M', '1.4B'])
    p.add_argument('--method')
    p.add_argument('--budget', default='20N', help='5N, 10N or 20N training tokens per parameter')
    p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--lr', type=float, help='peak learning rate (default: the preset of the size)')
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--config', help='MoE recipe, e.g. moe/configs/small/m3-g64.json')
    p.add_argument('--data-root')
    p.add_argument('--runs-root')
    p.add_argument('--paper', action='store_true', help='list the dense runs of the paper and exit')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.paper:
        for r in json.loads((ROOT / 'dense/configs/paper_runs.json').read_text())['runs']:
            print(f"--size {r['size']} --method {r['method']} --gpus {r['gpus']}  (budgets 20N, 10N, 5N)")
        return
    if not a.data_root or not a.runs_root:
        p.error('--data-root and --runs-root are required')
    data, runs = Path(a.data_root).resolve(), Path(a.runs_root).resolve()
    resolved = None
    if a.family == 'dense':
        if not a.size or not a.method:
            p.error('dense runs need --size and --method')
        cmd, env, out = dense_command(a.size, a.method, int(a.budget.rstrip('Nn')), a.gpus, data, runs, a.lr, a.seed)
    else:
        if not a.config:
            p.error('MoE runs need --config')
        cmd, env, out, resolved = moe_command(a.config, data, runs, a.gpus)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        print(json.dumps({'output': str(out), 'environment': env}, indent=2))
        return
    if not (data / 'DATASET_READY').is_file():
        raise FileNotFoundError(f'{data} is not a prepared corpus (no DATASET_READY); see docs/DATA.md')
    if (out / 'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; use a new --runs-root')
    out.mkdir(parents=True, exist_ok=True)
    if resolved is not None:
        (out / 'launch-config.json').write_text(json.dumps(resolved, indent=2) + '\n')
    for key in [k for k in os.environ if k.startswith('M3_') and k not in ('M3_MIX_KERNELS', 'M3_FUSED_MIX')]:
        del os.environ[key]  # do not inherit the choices of another run (the kernel choice is kept: see m3.py)
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=True)


if __name__ == '__main__':
    main()
