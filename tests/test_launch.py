"""The launcher reproduces the command lines of all 66 dense runs of the paper (no GPU or torch needed).

tests/data/historical_dense_runs.json holds the arguments and environment with which each run was launched, in the
naming of the original research code (module torchtitan.experiments.brainfuse_scaling, SA_* variables).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'dense'))

from launch import dense_command, run_name  # noqa: E402

HISTORICAL = json.loads((ROOT / 'tests/data/historical_dense_runs.json').read_text())
SIZE = {'L': '100M', 'XL': '300M', 'XXL': '500M', 'XXXL': '700M', 'XXXXL': '1.4B'}


def method_of(variant):
    if variant in ('base', 'mudd', 'blockattnres8'):
        return {'base': 'transformer', 'mudd': 'muddformer', 'blockattnres8': 'attnres-block'}[variant]
    assert variant.startswith('dd')
    return 'm3-g' + variant[2:]


def historical_names():
    names = {}
    for r in HISTORICAL:
        old = Path(r['args'][r['args'].index('--dump-folder') + 1]).name
        names[old] = run_name(r['size'], r['method'], int(r['budget'][:-1]))
    return names


def test_dense_runs_match_history():
    names = historical_names()
    assert len(HISTORICAL) == 66
    for r in HISTORICAL:
        budget = int(r['budget'][:-1])
        cmd, env, _ = dense_command(r['size'], r['method'], budget, r['gpus'], '${DATA_ROOT}', '${RUNS_ROOT}')
        new = cmd[cmd.index('torchtitan.train') + 1:]
        if budget != 20:  # the original launcher switched the module to enable this for branches
            new.remove('--wsd-preserve-rng-audit')
        old = list(r['args'])
        old[old.index('--module') + 1] = 'm3former'
        for i, a in enumerate(old):
            for o, n in names.items():
                a = a.replace('/' + o, '/' + n)
            old[i] = a
        assert new == old, (r['size'], r['method'], r['budget'], [x for x in zip(new, old) if x[0] != x[1]][:3])
        e = r['env']
        assert SIZE[e['SA_SCALE']] == env['M3_SIZE'] == r['size']
        assert method_of(e['SA_VARIANT']) == env['M3_METHOD'] == r['method']
        assert float(e['SA_LR']) == float(env['M3_LR']) == r['lr']
        for old_key, new_key in [('SA_SEED', 'M3_SEED'), ('SA_TOKENS_PER_PARAM', 'M3_TOKENS_PER_PARAM'),
                                 ('SA_LOCAL_BATCH', 'M3_LOCAL_BATCH'), ('SA_CKPT_INTERVAL', 'M3_CKPT_INTERVAL'),
                                 ('SA_COMPILE', 'M3_COMPILE')]:
            assert e[old_key] == env[new_key], (old_key, e[old_key], env[new_key])
        assert e['SA_WINDOW'] == 'Full' and e['SA_CORPUS_FORMAT'] == 'megatron_document_pairs_v1'


def test_paper_run_table():
    table = json.loads((ROOT / 'dense/configs/paper_runs.json').read_text())['runs']
    assert {(r['size'], r['method'], r['gpus']) for r in table} == {
        (r['size'], r['method'], r['gpus']) for r in HISTORICAL}
    assert len(table) * 3 == len(HISTORICAL)


if __name__ == '__main__':
    test_dense_runs_match_history()
    test_paper_run_table()
    print('ok')
