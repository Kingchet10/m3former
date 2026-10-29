import hashlib
import json
from pathlib import Path

from torchtitan.experiments.brainfuse_scaling import wsd_runtime as runtime


def restore_rng(state):
    runtime.random.setstate(state['python'])
    runtime.np.random.set_state(state['numpy'])
    runtime.torch.set_rng_state(state['cpu'])
    runtime.torch.cuda.set_rng_state(state['cuda'])


_original_fingerprints = runtime.WSDTrainer.fingerprints


def fingerprints_preserving_rng(self):
    # StatefulDataLoader.state_dict() creates its iterator after load_state_dict,
    # which draws a CPU base seed. Auditing must not perturb the restored stream.
    saved = runtime.rng_state()
    try:
        result = _original_fingerprints(self)
    finally:
        restore_rng(saved)
    restored = runtime.digest(runtime.rng_state())
    assert restored == runtime.digest(saved)
    observed = result['rng']
    result['rng'] = restored
    path = self.output / f'rng-audit-repair-rank{runtime.dist.get_rank()}.json'
    if not path.exists():
        path.write_text(json.dumps({'source': __file__,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'saved_rng': restored, 'rng_after_original_audit': observed,
            'rng_restored_after_audit': restored,
            'all_original_branch_fingerprint_assertions_retained': True}, indent=2))
    return result


def wsd():
    runtime.WSDTrainer.fingerprints = fingerprints_preserving_rng
    return runtime.wsd_recipe()
