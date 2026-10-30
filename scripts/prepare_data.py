"""Build a training corpus in the `megatron_document_pairs_v1` format that the dense and MoE trainers read.

    python scripts/prepare_data.py --output /path/to/corpus --tokenizer /path/to/tokenizer.model \
        --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-350BT --train-tokens 30e9

Documents are tokenized with the TinyLlama SentencePiece model (32,000 tokens, EOS = 2) and stored as int32
Megatron indexed files, one sequence per document. Each document contributes the next-token pairs
(ids[:-1], ids[1:]); a model never predicts across documents. The first documents of the stream form the validation
split (99,876,864 label tokens, as in the paper), the following ones the training split. Training reads the splits in
order, so shuffle the source beforehand if it is ordered.

`--jsonl FILE` reads documents from a local JSON-lines file with a "text" field instead of a Hugging Face dataset.
This builds the same format as the corpus of the paper, not the same documents.
"""
import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np

EOS = 2
VOCAB = 32000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class SplitWriter:
    """Megatron indexed dataset (version 1, int32 tokens) plus the cumulative pair offsets of its documents."""

    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        self.bin = (self.root / f'{name}.bin').open('wb')
        self.lengths, self.pairs = [], [0]

    def add(self, ids):
        a = np.asarray(ids, dtype='<i4')
        self.bin.write(a.tobytes())
        self.lengths.append(len(a))
        self.pairs.append(self.pairs[-1] + len(a) - 1)

    def close(self):
        self.bin.close()
        lengths = np.asarray(self.lengths, dtype='<i4')
        pointers = np.concatenate(([0], np.cumsum(lengths[:-1].astype(np.int64)) * 4)).astype('<i8')
        documents = np.arange(len(lengths) + 1, dtype='<i8')
        with (self.root / f'{self.name}.idx').open('wb') as f:
            f.write(b'MMIDIDX\x00\x00' + struct.pack('<QBQQ', 1, 4, len(lengths), len(lengths) + 1))
            f.write(lengths.tobytes())
            f.write(pointers.tobytes())
            f.write(documents.tobytes())
        np.asarray(self.pairs, dtype='<i8').tofile(self.root / f'{self.name}.pairs.i64')
        return self.pairs[-1]


def documents(args):
    if args.jsonl:
        with open(args.jsonl) as f:
            for line in f:
                yield json.loads(line)[args.text_field]
    else:
        from datasets import load_dataset
        stream = load_dataset(args.dataset, args.dataset_config, split='train', streaming=True)
        for row in stream:
            yield row[args.text_field]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tokenizer', required=True, type=Path, help='TinyLlama SentencePiece tokenizer.model')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-350BT')
    p.add_argument('--jsonl', type=Path)
    p.add_argument('--text-field', default='text')
    p.add_argument('--validation-tokens', type=float, default=99_876_864)
    p.add_argument('--train-tokens', type=float, required=True,
                   help='label tokens of the training split; 20N of the largest model you train, or more')
    p.add_argument('--no-eos', action='store_true', help='do not append EOS to every document')
    args = p.parse_args()

    import sentencepiece as spm
    tok = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if tok.get_piece_size() != VOCAB or tok.eos_id() != EOS:
        raise ValueError('Expected the TinyLlama tokenizer: 32,000 tokens, EOS = 2')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'tokenizer').mkdir()
    shutil.copyfile(args.tokenizer, args.output / 'tokenizer' / 'tokenizer.model')

    targets = {'validation': int(args.validation_tokens), 'train': int(args.train_tokens)}
    counts = {}
    source = documents(args)
    for split in ('validation', 'train'):
        writer = SplitWriter(args.output, split)
        for text in source:
            ids = tok.encode(text, out_type=int) + ([] if args.no_eos else [EOS])
            if len(ids) < 2:
                continue
            writer.add(ids)
            if writer.pairs[-1] >= targets[split]:
                break
        counts[split] = writer.close()
        if counts[split] < targets[split]:
            raise ValueError(f'The source ran out after {counts[split]} {split} label tokens')
        print(split, len(writer.lengths), 'documents,', counts[split], 'label tokens', flush=True)

    files = [f'{s}.{ext}' for s in ('train', 'validation') for ext in ('bin', 'idx', 'pairs.i64')]
    manifest = {
        'format': 'megatron_document_pairs_v1',
        'splits': {s: {'prefix': s, 'pair_offsets': f'{s}.pairs.i64', 'label_tokens': counts[s]}
                   for s in ('train', 'validation')},
        'file_sizes': {f: (args.output / f).stat().st_size for f in files},
        'files': {f: sha256(args.output / f) for f in files},
        'tokenizer': {'filename': 'tokenizer.model', 'vocab_size': VOCAB,
                      'tokenizer_sha256': sha256(args.output / 'tokenizer' / 'tokenizer.model')},
        'source': {'dataset': str(args.jsonl) if args.jsonl else f'{args.dataset}/{args.dataset_config}',
                   'append_eos': not args.no_eos},
        'source_binary_hashes_verified': True,
    }
    manifest['artifact_id'] = identity(manifest)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'DATASET_READY').write_text(sha256(args.output / 'manifest.json') + '\n')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
