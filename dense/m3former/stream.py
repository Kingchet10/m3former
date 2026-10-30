# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic fixed-length views over document-aligned cached pairs."""

from pathlib import Path

import numpy as np

from .artifacts import file_sha256, read_manifest


class PackedStream:
    def __init__(self, root, split, seq_len, rank=0, world_size=1, verify=True):
        self.root = Path(root)
        self.manifest = read_manifest(root)
        self.split, self.seq_len = split, seq_len
        self.rank, self.world_size = rank, world_size
        if seq_len < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid sequence length or data-parallel rank")
        count = self.manifest["splits"][split]["label_tokens"]
        self.arrays = {}
        for kind, dtype in (("input", "<u2"), ("label", "<u2"), ("position", "<u4")):
            relative = f"tokens/{split}.{kind}.bin"
            path = self.root / relative
            if path.stat().st_size != count * np.dtype(dtype).itemsize:
                raise ValueError(f"Invalid token file size: {relative}")
            if verify and file_sha256(path) != self.manifest["files"][relative]:
                raise ValueError(f"Token file checksum mismatch: {relative}")
            self.arrays[kind] = np.memmap(path, dtype=dtype, mode="r")
        self.block_count = count // seq_len
        self.dropped_tail_labels = count % seq_len
        self.next_block = rank
        self.epoch = 0

    def next(self):
        if self.next_block >= self.block_count:
            raise StopIteration
        start = self.next_block * self.seq_len
        end = start + self.seq_len
        result = {
            k: np.asarray(v[start:end], dtype=np.int64).copy()
            for k, v in self.arrays.items()
        }
        positions = result["position"]
        if positions[0] > 0:
            boundaries = np.flatnonzero(positions == 0)
            first_end = int(boundaries[0]) if len(boundaries) else len(positions)
            positions[:first_end] -= positions[0]
        self.next_block += self.world_size
        return result

    def reloop(self):
        self.next_block = self.rank
        self.epoch += 1

    def state_dict(self):
        return {
            "artifact_id": self.manifest["artifact_id"],
            "split": self.split,
            "seq_len": self.seq_len,
            "rank": self.rank,
            "world_size": self.world_size,
            "next_block": self.next_block,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("artifact_id", "split", "seq_len", "rank", "world_size"):
            if state[key] != expected[key]:
                raise ValueError(f"Incompatible stream resume: {key}")
        index = state["next_block"]
        if index < self.rank or (index - self.rank) % self.world_size:
            raise ValueError("Invalid stream cursor")
        self.next_block, self.epoch = index, state["epoch"]
