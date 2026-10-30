"""Read int32 Megatron indexed documents as next-token pairs (format megatron_document_pairs_v1, docs/DATA.md).

Each document contributes (ids[:-1], ids[1:]); never predict across documents.
Only a cumulative pair-offset index is added, not a second copy of the corpus.
"""
import struct
import errno
import os
from pathlib import Path

import numpy as np

from .artifacts import file_sha256, read_manifest
from .stream import PackedStream


def index_array(path, dtype, offset, count):
    try:
        return np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=(count,))
    except OSError as error:
        if error.errno != errno.ENODEV:
            raise
        # Some read-only shared mounts support read/pread but not mmap.
        with Path(path).open('rb') as f:
            f.seek(offset)
            return np.frombuffer(f.read(count * np.dtype(dtype).itemsize), dtype=dtype)


class Int32Tokens:
    """Memory-map when supported, otherwise bounded positioned reads (never load 120GB)."""
    def __init__(self, path):
        self.path, self.fd = Path(path), None
        self.nbytes = self.path.stat().st_size
        if self.nbytes % 4:
            raise ValueError('Invalid int32 token file size')
        try:
            self.array = np.memmap(path, mode='r', dtype='<i4')
        except OSError as error:
            if error.errno != errno.ENODEV:
                raise
            self.array = None
            self.fd = os.open(path, os.O_RDONLY)

    def __len__(self):
        return self.nbytes // 4

    def __getitem__(self, bounds):
        if self.array is not None:
            return self.array[bounds]
        start, end, stride = bounds.indices(len(self))
        if stride != 1:
            raise ValueError('Only contiguous token reads are supported')
        size, offset = max(end - start, 0) * 4, start * 4
        chunks = []
        while size:
            chunk = os.pread(self.fd, size, offset)
            if not chunk:
                raise ValueError('Truncated token file')
            chunks.append(chunk)
            size, offset = size - len(chunk), offset + len(chunk)
        return np.frombuffer(b''.join(chunks), dtype='<i4')

    def __del__(self):
        if self.fd is not None:
            os.close(self.fd)


class ShardedInt32Tokens:
    """Logical concatenation of immutable source binaries, without copying tokens."""
    def __init__(self, shards):
        self.shards = shards
        self.ends = np.cumsum([s['tokens'] for s in shards], dtype=np.int64)
        self.nbytes = int(self.ends[-1]) * 4
        self.readers = {}
        for s in shards:
            if Path(s['bin']).stat().st_size != s['tokens'] * 4:
                raise ValueError('Source shard size mismatch: ' + s['bin'])

    def __getitem__(self, bounds):
        start, end, stride = bounds.indices(self.nbytes // 4)
        if stride != 1:
            raise ValueError('Only contiguous slices supported')
        pieces = []
        while start < end:
            i = int(np.searchsorted(self.ends, start, side='right'))
            base = int(self.ends[i-1]) if i else 0
            stop = min(end, int(self.ends[i]))
            if i not in self.readers:
                self.readers[i] = Int32Tokens(self.shards[i]['bin'])
            pieces.append(self.readers[i][slice(start-base, stop-base)])
            start = stop
        if not pieces:
            return np.empty(0, dtype='<i4')
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


class DocumentIndex:
    def __init__(self, prefix, binary_shards=None):
        prefix = str(prefix)
        path = Path(prefix + '.idx')
        with path.open('rb') as f:
            header = f.read(34)
        if len(header) != 34 or header[:9] != b'MMIDIDX\x00\x00':
            raise ValueError('Invalid Megatron index header')
        version, dtype, count, ndoc = struct.unpack('<QBQQ', header[9:])
        if version != 1 or dtype != 4 or ndoc != count + 1 or count < 1:
            raise ValueError('Expected int32 Megatron index with one sequence per document')
        if path.stat().st_size != 34 + count * 12 + ndoc * 8:
            raise ValueError('Invalid Megatron index size')
        self.lengths = index_array(path, '<i4', 34, count)
        self.pointers = index_array(path, '<i8', 34 + count * 4, count)
        self.documents = index_array(path, '<i8', 34 + count * 12, ndoc)
        self.tokens = ShardedInt32Tokens(binary_shards) if binary_shards else Int32Tokens(prefix + '.bin')

    def validate(self):
        """Bounded-memory validation of all document boundaries and byte pointers."""
        offset = 0
        for start in range(0, len(self.lengths), 1_000_000):
            end = min(start + 1_000_000, len(self.lengths))
            lengths = self.lengths[start:end].astype(np.int64)
            if (lengths < 1).any():
                raise ValueError('Empty/negative document length')
            expected = offset + np.concatenate(([0], np.cumsum(lengths[:-1]))) * 4
            if not np.array_equal(self.pointers[start:end], expected):
                raise ValueError('Noncontiguous document pointers')
            if not np.array_equal(self.documents[start:end], np.arange(start, end)):
                raise ValueError('Multiple sequences per document are unsupported')
            offset += int(lengths.sum()) * 4
        if self.documents[-1] != len(self.lengths) or offset != self.tokens.nbytes:
            raise ValueError('Document/token count mismatch')


class MegatronPairStream(PackedStream):
    def __init__(self, root, split, seq_len, rank=0, world_size=1, verify=True):
        self.root = Path(root)
        self.manifest = read_manifest(root)
        if self.manifest.get('format') != 'megatron_document_pairs_v1':
            raise ValueError('Wrong corpus format')
        self.split, self.seq_len = split, seq_len
        self.rank, self.world_size = rank, world_size
        if seq_len < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid sequence length or data-parallel rank')
        info = self.manifest['splits'][split]
        relative_paths = [info['prefix'] + '.idx', info['pair_offsets']]
        if not info.get('binary_shards'):
            relative_paths.append(info['prefix'] + '.bin')
        elif verify:
            for shard in info['binary_shards']:
                if file_sha256(shard['bin']) != shard['sha256']:
                    raise ValueError('Source shard checksum mismatch: ' + shard['bin'])
        for relative in relative_paths:
            path = self.root / relative
            if path.stat().st_size != self.manifest['file_sizes'][relative]:
                raise ValueError(f'File size mismatch: {relative}')
            if verify and file_sha256(path) != self.manifest['files'][relative]:
                raise ValueError(f'Artifact checksum mismatch: {relative}')
        self.index = DocumentIndex(self.root / info['prefix'], info.get('binary_shards'))
        offset_path = self.root / info['pair_offsets']
        self.offsets = index_array(offset_path, '<i8', 0, offset_path.stat().st_size // 8)
        count = info['label_tokens']
        if len(self.offsets) != len(self.index.lengths) + 1 or self.offsets[0] != 0 or self.offsets[-1] != count:
            raise ValueError('Pair index length/count mismatch')
        self.block_count, self.dropped_tail_labels = divmod(count, seq_len)
        self.next_block, self.epoch = rank, 0

    def next(self):
        if self.next_block >= self.block_count:
            raise StopIteration
        cursor, filled = self.next_block * self.seq_len, 0
        result = {k: np.empty(self.seq_len, dtype=np.int64) for k in ('input', 'label', 'position')}
        while filled < self.seq_len:
            doc = int(np.searchsorted(self.offsets, cursor, side='right')) - 1
            within = cursor - int(self.offsets[doc])
            take = min(self.seq_len - filled, int(self.offsets[doc + 1]) - cursor)
            if take <= 0:
                raise ValueError('Invalid pair offsets')
            source = int(self.index.pointers[doc]) // 4 + within
            ids = self.index.tokens[source:source + take + 1]
            if len(ids) != take + 1 or (ids < 0).any() or (ids >= self.manifest['tokenizer']['vocab_size']).any():
                raise ValueError('Out-of-range or truncated document tokens')
            target = slice(filled, filled + take)
            result['input'][target], result['label'][target] = ids[:-1], ids[1:]
            # Reset both at new documents and at a block starting mid-document.
            result['position'][target] = np.arange(take)
            cursor, filled = cursor + take, filled + take
        self.next_block += self.world_size
        return result
