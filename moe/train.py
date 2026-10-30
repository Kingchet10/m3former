"""Megatron Core model + explicit DDP/AdamW training; no scheduler submission.

This trainer intentionally uses only DP, with TP/PP/EP=1. All expert parameters
are replicated and synchronized by torch DDP, including residual router params.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from architecture import Architecture, build_model
from data import MegatronPairStream, file_sha256
from recipe import learning_rate


def packed(row, device):
    from megatron.core.packed_seq_params import PackedSeqParams
    if isinstance(row, list):
        # Each original 2048-token sequence starts with position zero, so this
        # concatenation preserves both document and original sequence barriers.
        row={key:np.concatenate([r[key] for r in row]) for key in ('input','label','position')}
    starts = np.flatnonzero(row['position'] == 0)
    cu = np.append(starts, len(row['position'])).astype(np.int32)
    assert cu[0] == 0 and (np.diff(cu) > 0).all()
    cu = torch.from_numpy(cu).to(device)
    maxlen = int(np.diff(np.append(starts, len(row['position']))).max())
    inputs, labels, positions = (torch.from_numpy(row[k].copy()).to(device).unsqueeze(0)
                                 for k in ('input', 'label', 'position'))
    params = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                            max_seqlen_q=maxlen, max_seqlen_kv=maxlen)
    return inputs, labels, positions, params


def row_at(stream, block):
    stream.next_block = block
    return stream.next()


def append(path, record):
    with path.open('a') as f:
        f.write(json.dumps(record)+'\n')
        f.flush()


@torch.no_grad()
def evaluate(model, stream, r, device, rank, world):
    model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    microbatch=r['micro_batch_sequences']
    for block in range(rank*microbatch, r['validation_sequences'], world*microbatch):
        rows=[row_at(stream,b) for b in range(block,min(block+microbatch,r['validation_sequences']))]
        x, y, pos, params = packed(rows, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model(x, pos, None, labels=y, packed_seq_params=params)
        total[0] += losses.double().sum()
        total[1] += y.numel()
    dist.all_reduce(total)
    model.train()
    assert int(total[1]) == r['validation_tokens']
    return {'loss': (total[0]/total[1]).item(), 'tokens': int(total[1])}


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        if tensor is None:  # TE extra_state with FP8 disabled.
            h.update(b'null')
        else:
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--benchmark-steps', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    r = json.loads(Path(args.config).read_text())
    if 'performance' in r:
        perf=r['performance']
        for key,env in [('packed_experts','M3_MOE_PACKED'),('operator_fusion','M3_MOE_FUSION'),('compile_rmsnorm','M3_MOE_COMPILE_NORM')]:
            os.environ[env]=str(int(perf[key]))
        os.environ['M3_MOE_BUCKET_MB']=str(perf['ddp_bucket_mb'])
        torch.set_num_threads(perf['cpu_threads'])
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                            pipeline_model_parallel_size=1,
                                            expert_model_parallel_size=1)
    microbatch=r['micro_batch_sequences']
    assert r['global_batch_sequences'] % (world*microbatch) == 0
    accum = r['global_batch_sequences'] // (world*microbatch)
    out = Path(args.output)
    if rank == 0:
        # A run is immutable; resume is intentionally unavailable under final-only retention.
        out.mkdir(parents=True, exist_ok=False)
        (out/'config.json').write_text(json.dumps(r, indent=2))
        (out/'data-manifest.json').write_bytes((Path(r['data_root'])/'manifest.json').read_bytes())
        source_root=Path(__file__).resolve().parent
        if (source_root/'source-manifest.json').exists():
            inventory=json.loads((source_root/'source-manifest.json').read_text())
            for relative,expected in inventory.items():
                assert file_sha256(source_root/relative)==expected, f'Source changed: {relative}'
            (out/'source-manifest.json').write_bytes((source_root/'source-manifest.json').read_bytes())
        versions={}
        for package in ('torch','transformer_engine','flash_attn','triton','numpy'):
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package]=None
        (out/'runtime.json').write_text(json.dumps(dict(versions=versions,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(local),source_root=str(source_root),
            performance_environment={k:os.environ.get(k) for k in ['M3_MOE_PACKED','M3_MOE_FUSION','M3_MOE_COMPILE_NORM','M3_MOE_BUCKET_MB']},
            cpu_threads=torch.get_num_threads()),indent=2))
    dist.barrier()
    torch.manual_seed(r['seed']); model_parallel_cuda_manual_seed(r['seed'])
    router = r['router']
    model = build_model(variant=r['variant'], architecture=Architecture(**r['architecture']),
                        load_balancing_type=router['load_balancing_type'],
                        aux_loss_coeff=router['aux_loss_coeff'], z_loss_coeff=router['z_loss_coeff']).to(device)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True,
              bucket_cap_mb=int(os.environ.get('M3_MOE_BUCKET_MB','25')))
    initial_hash = model_digest(model)
    hashes = [None] * world
    dist.all_gather_object(hashes, initial_hash)
    assert len(set(hashes)) == 1, 'Initial parameters differ across ranks'
    optimizer = torch.optim.AdamW(model.parameters(), lr=r['peak_lr'], betas=tuple(r['betas']),
                                   eps=r['eps'], weight_decay=r['weight_decay'], fused=True)
    train = MegatronPairStream(r['data_root'], 'train', r['sequence_length'], verify=False)
    valid = MegatronPairStream(r['data_root'], 'validation', r['sequence_length'], verify=False)
    if rank == 0:
        (out/'initialization.json').write_text(json.dumps({'sha256': initial_hash,
            'parameters': sum(p.numel() for p in model.parameters()), 'world_size': world,
            'gradient_accumulation': accum, 'data_manifest_sha256': file_sha256(Path(r['data_root'])/'manifest.json')}, indent=2))
    start = time.monotonic()
    model.train()
    profiler = None
    if args.profile and rank == 0:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10,warmup=2,active=3,repeat=1))
        profiler.start()
    for step in range(1, (args.benchmark_steps or r['train_steps'])+1):
        lr = learning_rate(step, r)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1./accum, device=device))
        lm_sum = torch.zeros((), device=device)
        for micro in range(accum):
            block = (step-1)*r['global_batch_sequences'] + (rank*accum+micro)*microbatch
            x, y, pos, params = packed([row_at(train,block+j) for j in range(microbatch)], device)
            sync = ddp.no_sync() if micro < accum-1 else nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = ddp(x, pos, None, labels=y, packed_seq_params=params)
                    loss = losses.float().mean()
                (loss/accum).backward()
            lm_sum += loss.detach()/accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), r['gradient_clip'], error_if_nonfinite=True)
        if not torch.isfinite(lm_sum):
            raise FloatingPointError('Nonfinite language-model loss')
        optimizer.step()
        if step == 1 or step % r['log_every'] == 0 or args.benchmark_steps:
            dist.all_reduce(lm_sum); lm_sum /= world
            if rank == 0:
                entry = dict(step=step, tokens=step*r['global_batch_sequences']*r['sequence_length'], loss=lm_sum.item(), lr=lr,
                             grad_norm=grad_norm.item(), elapsed_seconds=time.monotonic()-start)
                append(out/'train.jsonl', entry); print(json.dumps(entry), flush=True)
        if profiler is not None:
            profiler.step()
            if step == 15:
                profiler.stop()
                (out/'profile.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
                profiler.export_chrome_trace(str(out/'profile.json'))
                profiler = None
        if args.benchmark_steps:
            continue
        if step in r['validation_steps']:
            result = evaluate(model, valid, r, device, rank, world)
            if rank == 0:
                append(out/'validation.jsonl', dict(step=step, final=step==r['train_steps'], **result))
    if args.benchmark_steps:
        torch.cuda.synchronize()
        verified_sync = None
        if r.get('verify_final_sync',False):
            dist.all_gather_object(hashes,model_digest(model))
            assert len(set(hashes))==1, 'Benchmark ranks diverged'
            verified_sync = True
        if rank == 0:
            (out/'benchmark.json').write_text(json.dumps(dict(status='complete',steps=args.benchmark_steps,
                max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved(),
                fusion=os.environ.get('M3_MOE_FUSION','0'),bucket_mb=os.environ.get('M3_MOE_BUCKET_MB','25'),
                packed=os.environ.get('M3_MOE_PACKED','0'),compile_norm=os.environ.get('M3_MOE_COMPILE_NORM','0'),
                all_ranks_identical=verified_sync),indent=2))
        dist.destroy_process_group()
        return
    # Final-only model checkpoint. Keep every log/config; never write intermediate Adam states.
    final_hash = model_digest(model)
    dist.all_gather_object(hashes, final_hash)
    assert len(set(hashes)) == 1, 'Final model differs across data-parallel ranks'
    if rank == 0:
        weights = {k: v.detach().cpu() if torch.is_tensor(v) else v for k,v in model.state_dict().items()}
        torch.save({'model': weights, 'config': r, 'step': r['train_steps']}, out/'final.pt.tmp')
        os.replace(out/'final.pt.tmp', out/'final.pt')
        restored = torch.load(out/'final.pt', map_location='cpu', weights_only=True)['model']
        assert weights.keys() == restored.keys()
        assert all(torch.equal(weights[k], restored[k]) if torch.is_tensor(weights[k]) else weights[k] == restored[k] for k in weights)
        (out/'research-summary.json').write_text(json.dumps(dict(
            status='completed', final_validation=result, model_sha256=final_hash,
            checkpoint_sha256=file_sha256(out/'final.pt'), checkpoint_reload='bitwise_passed',
            all_ranks_identical=True, train_tokens=r['train_tokens'],
            elapsed_seconds=time.monotonic()-start, optimizer_saved=False), indent=2))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
