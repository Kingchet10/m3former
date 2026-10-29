# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measured validation with actual label counts, using the upstream validator."""

import json
import time
from dataclasses import dataclass

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.validate import Validator


class ResearchValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        pass

    def post_dataloading_process(self, input_dict, labels, model_parts):
        self.measured_labels += int((labels != IGNORE_INDEX).sum())
        return super().post_dataloading_process(input_dict, labels, model_parts)

    def validate(self, model_parts, step):
        self.measured_labels = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        super().validate(model_parts, step)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if self.dp_world_size > 1:
            count = torch.tensor(self.measured_labels, dtype=torch.int64, device="cuda")
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            self.measured_labels = int(count.item())
            seconds = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
            torch.distributed.all_reduce(seconds, op=torch.distributed.ReduceOp.MAX)
            elapsed = float(seconds.item())
        if torch.distributed.get_rank() != 0:
            return
        with (self.metrics_processor.output / "validation-cost.jsonl").open(
            "a"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "step": step,
                        "seconds": elapsed,
                        "valid_labels": self.measured_labels,
                    }
                )
                + "\n"
            )
