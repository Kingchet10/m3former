"""Export the model weights of a dense TorchTitan (DCP) checkpoint for evaluation.

    python scripts/export_dense_checkpoint.py RUN/checkpoint/step-N RUN/checkpoint/step-N-model

Writes pytorch_model-*.bin shards with native M3Former state-dict keys, pytorch_model.bin.index.json and
model-only.json (tensor and file hashes), the format that evaluation/worker.py loads. Keep the output inside
RUN/checkpoint/: the worker reads the run configuration RUN/config.json two levels above the weights. CPU only; the
checkpoint is read, never modified.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='DCP checkpoint directory, e.g. RUN/checkpoint/step-98013')
    parser.add_argument('output', type=Path, help='new directory for the exported weights')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    export_model(args.checkpoint, args.output)
    print(json.dumps({k: v for k, v in verify_export(args.output).items() if k not in ('files', 'tensors')}, indent=2))


if __name__ == '__main__':
    main()
