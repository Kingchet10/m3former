"""CPU-only, post-completion export. Never changes training or WSD anchors."""
import argparse
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(-1).view(__import__('torch').uint8).numpy()).hexdigest()


def export_model(source, destination):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    torch.set_num_threads(2)
    reader = dcp.FileSystemReader(source)
    metadata = reader.read_metadata()
    entries = {}
    nested_model = any(tuple(path)[0] == 'model' for path in metadata.planner_data.values())
    for key, value in metadata.state_dict_metadata.items():
        path = tuple(metadata.planner_data[key])
        is_model = path[0] == 'model' if nested_model else len(path) == 1
        if is_model:
            assert isinstance(value, TensorStorageMetadata), key
            model_key = path[-1]
            assert path == (('model', model_key) if nested_model else (model_key,)), key
            entries[model_key] = value
        else:
            assert path[0] in {'optimizer', 'dataloader', 'lr_scheduler', 'train_state', 'wsd_rng'}, (key, path)
    assert entries, 'No model tensors in checkpoint'
    destination.mkdir()
    groups, group, count = [], [], 0
    for key, value in sorted(entries.items()):
        size = value.size.numel() * torch.empty((), dtype=value.properties.dtype).element_size()
        if group and count + size > 256 * 1024**2:
            groups.append(group)
            group, count = [], 0
        group.append(key)
        count += size
    if group:
        groups.append(group)
    mapping, tensors, files = {}, {}, {}
    for i, keys in enumerate(groups, 1):
        model = {key: torch.empty(entries[key].size, dtype=entries[key].properties.dtype) for key in keys}
        dcp.load({'model': model} if nested_model else model, storage_reader=reader)
        expected = {key: tensor_sha(value) for key, value in model.items()}
        name = f'pytorch_model-{i:05d}-of-{len(groups):05d}.bin'
        path = destination / name
        torch.save(model, path)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        assert loaded.keys() == model.keys()
        for key in keys:
            value = loaded[key]
            assert value.dtype == model[key].dtype and value.shape == model[key].shape
            assert tensor_sha(value) == expected[key], key
            mapping[key] = name
            tensors[key] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': expected[key]}
        files[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        del loaded, model
        gc.collect()
    assert set(mapping) == set(entries)
    total = sum(entries[k].size.numel() * torch.empty((), dtype=entries[k].properties.dtype).element_size() for k in entries)
    write(destination / 'pytorch_model.bin.index.json', {'metadata': {'total_size': total, 'format': 'native_torchtitan_state_dict'}, 'weight_map': mapping})
    proof = {'status': 'verified_model_only', 'at': stamp(), 'source': str(source),
             'source_metadata_sha256': sha(source / '.metadata'), 'tensor_count': len(entries),
             'total_tensor_bytes': total, 'files': files, 'tensors': tensors,
             'exact_dtype_preserved': True, 'contains_optimizer': False,
             'full_training_resume': False, 'load': 'Load each bin with torch.load(weights_only=True); merge state dicts using the index. Keys are native model keys, not HF architecture conversion.'}
    write(destination / 'model-only.json', proof)
    return proof


def verify_export(path):
    proof = json.loads((path / 'model-only.json').read_text())
    assert proof['status'] == 'verified_model_only'
    index = json.loads((path / 'pytorch_model.bin.index.json').read_text())
    assert set(index['weight_map']) == set(proof['tensors'])
    assert set(index['weight_map'].values()) == set(proof['files'])
    for name, record in proof['files'].items():
        assert Path(name).name == name
        assert (path / name).stat().st_size == record['bytes'] and sha(path / name) == record['sha256']
    return proof


def retain(item):
    run, ck = Path(item['run']), Path(item['checkpoint'])
    assert ck == run / 'checkpoint' / f"step-{item['step']}"
    receipt = run / 'checkpoint-retention.json'
    temp = ck.with_name(ck.name + '.model-only-export')
    old = ck.with_name(ck.name + '.full-state-to-delete')
    if receipt.exists() and not old.exists():
        assert (ck / 'model-only.json').exists()
        return 'already_retained'
    complete = Path(item['completion'])
    if not complete.exists():
        return 'pending_training'
    evidence = json.loads(complete.read_text())
    assert evidence['status'] == 'complete' and evidence['run'] == str(run)
    assert evidence['checkpoint'] == str(ck) and evidence['step'] == item['step']
    config = json.loads((run / 'config.json').read_text())
    assert item['step'] not in config['wsd_anchor_steps'], 'Never convert branch sources'
    assert config['training']['steps'] == item['step']
    summary = json.loads((run / 'research-summary.json').read_text())
    assert summary['status'] == 'complete' and summary['step'] == item['step']
    assert (run / 'validation.jsonl').exists()
    # Recover either side of the two atomic renames without losing the original.
    if not ck.exists():
        assert old.is_dir() and temp.is_dir()
        verify_export(temp)
        temp.rename(ck)
    if not (ck / 'model-only.json').exists():
        assert not old.exists()
        if temp.exists():
            shutil.rmtree(temp)  # Only our incomplete export; never a training checkpoint.
        original_files = {p.name: p.stat().st_size for p in ck.iterdir() if p.is_file()}
        assert '.metadata' in original_files
        export_model(ck, temp)
        verify_export(temp)
        write(run / 'checkpoint-export-verified.json', {'at': stamp(), 'source': str(ck), 'files': original_files})
        ck.rename(old)
        temp.rename(ck)
    proof = verify_export(ck)
    write(receipt, {'at': stamp(), 'status': 'model_only_verified', 'checkpoint': str(ck),
                    'proof': str(ck / 'model-only.json'), 'tensor_count': proof['tensor_count'],
                    'branch_sources': 'untouched; existing node runner removes only after dependent endpoint completes',
                    'original_training_config': 'preserved unchanged; full checkpoint was exported after completion',
                    'full_state_cleanup': 'pending' if old.exists() else 'complete'})
    if old.exists():
        assert (old / '.metadata').exists()
        shutil.rmtree(old)
    data = json.loads(receipt.read_text())
    data['full_state_cleanup'] = 'complete'
    write(receipt, data)
    return 'model_only_verified'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    home = args.manifest.parent
    with (home / 'retention.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(args.manifest.read_text())
        while True:
            states = {}
            for item in manifest['endpoints']:
                try:
                    state = retain(item)
                    states[item['name']] = {'status': state}
                except Exception as error:
                    states[item['name']] = {'status': 'export_error_full_state_preserved', 'error': repr(error)}
                    print(stamp(), item['name'], repr(error), flush=True)
            done = all(v['status'] in ('model_only_verified', 'already_retained') for v in states.values())
            write(home / 'state.json', {'pid': os.getpid(), 'at': stamp(), 'status': 'complete' if done else 'active', 'endpoints': states})
            if done or args.once:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
