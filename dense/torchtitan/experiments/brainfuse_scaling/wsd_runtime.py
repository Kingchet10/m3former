"""Exact WSD branches; isolated addition to the sealed DP4 source.

Optimizer updates are 1-indexed. Tail update b+1 uses p=0 and update T
uses p=1. Branch restore accepts only the declared budget/schedule change.
"""
import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchtitan.components.optimizer import LRSchedulersContainer
from .trainer import ResearchTrainer


def factor(index, warmup, total, tail):
    update = index + 1
    if update <= warmup:
        return update / warmup
    branch = total - tail
    if update <= branch:
        return 1.0
    p = min(1.0, max(0.0, (update - branch - 1) / (tail - 1)))
    return 0.1 + 0.9 * (1 - math.sqrt(p))


class WSDSchedulers(LRSchedulersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        tail_steps: int = 2

        def build(self, *, optimizers, training_steps):
            total = self.total_steps or training_steps
            assert self.tail_steps >= 2 and self.warmup_steps < total - self.tail_steps
            return WSDSchedulers(optimizers, lambda i: factor(i, self.warmup_steps, total, self.tail_steps))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for sched in self.schedulers:
            for group, value in zip(sched.optimizer.param_groups, sched._last_lr):
                group['lr'] = value


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


class RankRNG:
    def state_dict(self):
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, rng_state())
        return {'all_ranks': pickle.dumps(ranks, protocol=5)}

    def load_state_dict(self, state):
        ranks = pickle.loads(state['all_ranks'])
        assert len(ranks) == dist.get_world_size()
        self.pending = ranks[dist.get_rank()]
        self.restore()

    def restore(self):
        local = self.pending
        random.setstate(local['python'])
        np.random.set_state(local['numpy'])
        torch.set_rng_state(local['cpu'])
        torch.cuda.set_rng_state(local['cuda'])


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if hasattr(x, 'to_local'):
            x = x.to_local()
        if isinstance(x, torch.Tensor):
            x = x.detach().contiguous().cpu()
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x: visit(v)
        else:
            h.update(pickle.dumps(x, protocol=5))
    visit(value)
    return h.hexdigest()


def scientific_signature(c):
    # Only the known branch changes are removed. Everything else is compared.
    keys = ['architecture', 'training', 'optimizer', 'loss', 'compile',
            'activation_checkpoint', 'parallelism', 'dataloader', 'tokenizer', 'debug']
    s = json.loads(json.dumps({k: c[k] for k in keys}))
    s['training'].pop('steps')
    return s


class WSDTrainer(ResearchTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ResearchTrainer.Config):
        wsd_anchor_steps: list[int] = field(default_factory=list)
        wsd_parent_run: str | None = None
        wsd_restore_step: int = -1
        wsd_qualification: bool = False

    def __init__(self, config):
        self.branch_proof = None
        self.branch_verified = False
        super().__init__(config)
        assert config.checkpoint.enable and not config.checkpoint.last_save_model_only
        assert str(config.checkpoint.async_mode) == 'disabled'
        self.rank_rng = RankRNG()
        self.checkpointer.states['wsd_rng'] = self.rank_rng
        if config.wsd_parent_run:
            parent = Path(config.wsd_parent_run)
            pc = json.loads((parent / 'config.json').read_text())
            step = config.wsd_restore_step
            ck = parent / 'checkpoint' / f'step-{step}'
            assert Path(config.checkpoint.initial_load_path).resolve() == ck.resolve()
            assert not config.checkpoint.initial_load_model_only
            assert scientific_signature(pc) == scientific_signature(config.to_dict())
            assert pc['lr_scheduler']['warmup_steps'] == config.lr_scheduler.warmup_steps
            assert pc['lr_scheduler']['min_lr_factor'] == config.lr_scheduler.min_lr_factor == 0.1
            assert step in pc['wsd_anchor_steps']
            assert step == config.training.steps - config.lr_scheduler.tail_steps
            assert pc['training']['steps'] >= config.training.steps
            proof = json.loads((parent / f'checkpoint-proof-{step}.json').read_text())
            assert proof['artifact_id'] == self.manifest['artifact_id']
            self.branch_proof = proof
            self.resume_lineage = {'parent_protocol_id': proof['protocol_id'],
                                   'checkpoint_state': proof['train_state']}
        original_save = self.checkpointer.save
        def save(step, last_step=False):
            anchor = step in (config.wsd_anchor_steps or [])
            if anchor or last_step:
                reserve = Path(__import__('os').environ['RR_WSD_RESERVE'])
                if self.is_primary and reserve.exists():
                    amount = int(__import__('os').environ['RR_WSD_CHECKPOINT_BYTES'])
                    with reserve.open('r+b') as f:
                        f.truncate(max(0, reserve.stat().st_size - amount))
                dist.barrier()
                local = self.fingerprints()
                all_ranks = [None] * dist.get_world_size()
                dist.all_gather_object(all_ranks, local)
                saved = original_save(step, last_step=True)
                if saved and self.is_primary:
                    proof = {'step': step, 'protocol_id': self.protocol_id,
                             'artifact_id': self.manifest['artifact_id'], 'ranks': all_ranks,
                             'train_state': {'train_state.' + k: v for k, v in self.state_dict().items()}}
                    (self.output / f'checkpoint-proof-{step}.json').write_text(json.dumps(proof, indent=2))
                dist.barrier()
                return saved
            return False
        self.checkpointer.save = save

    def fingerprints(self):
        loader = {k: pickle.loads(v) if isinstance(v, bytes) else v
                  for k, v in self.dataloader.state_dict().items()}
        return {'model': digest(self.checkpointer.states['model'].state_dict()),
                'optimizer': digest(self.optimizers.state_dict()),
                'loader': digest(loader),
                'rng': digest(rng_state()),
                'scheduler_step': self.lr_schedulers.state_dict()['last_epoch'],
                'valid_labels': self.valid_tokens_seen}

    def batch_generator(self, data_iterable):
        first = True
        for batch in super().batch_generator(data_iterable):
            # FrozenTokenLoader is lazy: iterator construction can draw a base
            # seed on the first next(), after train_step's restore check.
            if first and self.branch_proof is not None:
                self.rank_rng.restore()
            first = False
            yield batch

    def train_step(self, iterator):
        if self.branch_proof is not None and not self.branch_verified:
            # DataLoader iterator construction can consume a CPU RNG draw after
            # DCP restore. Reinstate the saved per-rank stream before any update.
            self.rank_rng.restore()
            expected = self.branch_proof['ranks'][dist.get_rank()]
            actual = self.fingerprints()
            assert expected == actual, {'expected': expected, 'actual': actual}
            (self.output / f'branch-restore-rank{dist.get_rank()}.json').write_text(
                json.dumps({'status': 'passed', 'parent': self.config.wsd_parent_run,
                            'step': self.step - 1, 'fingerprints': actual}, indent=2))
            dist.barrier()
            self.branch_verified = True
        applied = [g['lr'] for opt in self.optimizers for g in opt.param_groups]
        super().train_step(iterator)
        if self.is_primary:
            with (self.output / 'wsd-lr.jsonl').open('a') as f:
                f.write(json.dumps({'step': self.step, 'applied_lr': applied,
                                    'valid_labels_seen': self.valid_tokens_seen}) + '\n')


def wsd_recipe():
    from .stagea_protocol import recipe
    base = recipe()
    args = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    args['lr_scheduler'] = WSDSchedulers.Config(warmup_steps=125, total_steps=12404,
                                               tail_steps=2481, min_lr_factor=0.1,
                                               decay_type='sqrt', decay_ratio=0.2)
    return WSDTrainer.Config(**args)
