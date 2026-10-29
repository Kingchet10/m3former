# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter-name sub-seeds isolate unchanged weights from FFN shape changes."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class PairedInitializer:
    initializer: Callable
    seed: int
    name: str
    gain: float = 1.0

    def to_dict(self):
        return {
            "initializer": repr(self.initializer),
            "seed": self.seed,
            "name": self.name,
            "gain": self.gain,
        }

    def __call__(self, parameter):
        seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{self.name}".encode()).digest()[:8],
            "little",
        ) % (2**63 - 1)
        devices = [parameter.device.index] if parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.initializer(parameter)
            if self.gain != 1.0:
                with torch.no_grad():
                    parameter.mul_(self.gain)


def configure_paired_initializers(config, seed, reference_width):
    """Keep original marginal distributions; normalize down-projection fan-in.

    The gain sqrt(reference_width / actual_down_input_width) makes the
    initial residual variance equal in expectation, assuming independent
    zero-mean branches. Actual RMS values are verified separately.
    """
    from torchtitan.protocols.module import Module

    for name, child, _, _ in config.traverse(Module.Config, recurse=True):
        if child.param_init is None:
            continue
        gain = 1.0
        if name.endswith("feed_forward.w2"):
            gain = (reference_width / child.in_features) ** 0.5
        child.param_init = {
            key: PairedInitializer(value, seed, f"{name}.{key}", gain)
            for key, value in child.param_init.items()
        }
