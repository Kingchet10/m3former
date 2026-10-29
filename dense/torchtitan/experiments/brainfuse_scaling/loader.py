# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan dataloader extension for immutable pre-tokenized FineWeb."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX

from .artifacts import file_sha256, read_manifest
from .stream import PackedStream


def partition_global_batch(inputs, labels, rank, world_size, remaining=None):
    """Mask the canonical target prefix before splitting complete sequences."""
    if labels.shape[0] % world_size or not 0 <= rank < world_size:
        raise ValueError("Global batch must split evenly across data-parallel ranks")
    if remaining is not None:
        valid_indices = (labels != IGNORE_INDEX).flatten().nonzero().flatten()
        if len(valid_indices) > max(remaining, 0):
            labels = labels.clone()
            labels.flatten()[valid_indices[max(remaining, 0) :]] = IGNORE_INDEX
    size = labels.shape[0] // world_size
    bounds = slice(rank * size, (rank + 1) * size)
    return {key: value[bounds] for key, value in inputs.items()}, labels[bounds], labels


class PackedTokenDataset(IterableDataset, Stateful):
    def __init__(self, stream, infinite=False):
        super().__init__()
        self.stream, self.infinite = stream, infinite

    def __iter__(self):
        while True:
            try:
                row = self.stream.next()
            except StopIteration:
                if not self.infinite:
                    return
                self.stream.reloop()
                continue
            yield (
                {
                    "input": torch.from_numpy(row["input"]),
                    "positions": torch.from_numpy(row["position"]),
                },
                torch.from_numpy(row["label"]),
            )

    def state_dict(self):
        return self.stream.state_dict()

    def load_state_dict(self, state):
        self.stream.load_state_dict(state)


class FrozenTokenLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "fineweb_frozen"
        split: str = "train"
        infinite: bool = False
        verify_files: bool = True

    def __init__(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        seq_len,
        local_batch_size,
        snapshot_every_n_steps=None,
    ):
        if config.dataset_path is None:
            raise ValueError("Set dataloader.dataset_path to a frozen corpus directory")
        if config.num_workers != 0:
            raise ValueError(
                "FrozenTokenLoader currently requires num_workers=0 for exact cursor semantics"
            )
        self.execution_rank = dp_rank
        self.execution_world_size = dp_world_size
        self.remaining_labels = None
        self.audit_global_batch = False
        self.last_global_batch_sha256 = None
        stream_type = PackedStream
        manifest = read_manifest(config.dataset_path)
        if manifest.get("format") == "megatron_document_pairs_v1":
            from .megatron_stream import MegatronPairStream
            stream_type = MegatronPairStream
        stream = stream_type(
            config.dataset_path,
            config.split,
            seq_len,
            rank=0,
            world_size=1,
            verify=config.verify_files,
        )
        token_path = Path(tokenizer.tokenizer_path) / stream.manifest["tokenizer"].get(
            "filename", "tokenizer.json"
        )
        if file_sha256(token_path) != stream.manifest["tokenizer"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer differs from the frozen token stream")
        if tokenizer.get_vocab_size() != stream.manifest["tokenizer"]["vocab_size"]:
            raise ValueError("Vocabulary size differs from the frozen token stream")
        if stream.block_count < dp_world_size * local_batch_size:
            raise ValueError("Frozen split is too small for one global batch")
        super().__init__(
            PackedTokenDataset(stream, config.infinite),
            0,
            1,
            batch_size=local_batch_size * dp_world_size,
            num_workers=0,
            drop_last=True,
            pin_memory=config.pin_memory,
            snapshot_every_n_steps=snapshot_every_n_steps or 1,
        )

    def __iter__(self):
        for inputs, labels in super().__iter__():
            if self.execution_world_size == 1:
                yield inputs, labels
                continue
            local_inputs, local_labels, global_labels = partition_global_batch(
                inputs,
                labels,
                self.execution_rank,
                self.execution_world_size,
                self.remaining_labels,
            )
            # Consume the canonical prefix once across accumulated global
            # microbatches, including a partial final optimizer step.
            if self.remaining_labels is not None:
                self.remaining_labels = max(
                    self.remaining_labels - int((global_labels != IGNORE_INDEX).sum()), 0
                )
            if self.audit_global_batch:
                digest = hashlib.sha256()
                for tensor in (inputs["input"], inputs["positions"], global_labels):
                    digest.update(tensor.numpy().astype("<i8", copy=False).tobytes())
                self.last_global_batch_sha256 = digest.hexdigest()
            yield local_inputs, local_labels
