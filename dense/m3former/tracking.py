# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorBoard views aligned by optimizer step, effective labels, and train time."""

import json
import math
import time
from pathlib import Path

from .artifacts import write_json

INT64_MAX = 2**63 - 1


class ResearchTracker:
    def __init__(self, root, metadata, *, step, labels, training_seconds, flops=None):
        from torch.utils.tensorboard import SummaryWriter

        self.root = Path(root)
        path = self.root / "run-identity.json"
        if path.exists():
            previous = json.loads(path.read_text())
            for name in ("protocol_id", "artifact_id", "seed"):
                if previous[name] != metadata[name]:
                    raise ValueError(
                        "Tracking directory belongs to a different run: " + name
                    )
        else:
            write_json(path, metadata)
        # TensorBoard sorts event filenames lexicographically. Its process-wide
        # writer counter is not padded, so .10 sorts before .9 in the same second.
        # Give resumed segments a strictly later timestamp on every axis; otherwise
        # restart purging may read the new segment before the stale old segment.
        timestamps = [
            int(event.name.split(".")[3])
            for event in self.root.glob("by_*/events.out.tfevents.*")
        ]
        if timestamps:
            delay = max(timestamps) + 1.01 - time.time()
            if delay > 2:
                raise ValueError("Event timestamps are in the future; check host clock")
            if delay > 0:
                time.sleep(delay)
        self.writers = {}
        for axis, cursor in self.coordinates(
            step, labels, training_seconds, flops
        ).items():
            self.writers[axis] = SummaryWriter(
                str(self.root / axis),
                purge_step=min(cursor + 1, INT64_MAX),
                max_queue=1000,
                flush_secs=30,
            )
        segment = {
            "restored_step": step,
            "restored_labels": labels,
            "restored_training_seconds": training_seconds,
            "restored_training_flops": flops,
            **metadata,
        }
        with (self.root / "segments.jsonl").open("a") as stream:
            stream.write(json.dumps(segment, default=str) + "\n")
        self.writers["by_step"].add_text(
            "run/configuration", json.dumps(metadata, indent=2, default=str), step
        )

    @staticmethod
    def coordinates(step, labels, training_seconds, flops=None):
        result = {
            "by_step": int(step),
            "by_labels": int(labels),
            "by_training_milliseconds": round(training_seconds * 1000),
        }
        if flops is not None:
            # TensorBoard event steps are int64; cumulative FLOPs exceed 2**63 late in large runs.
            result["by_training_flops"] = min(int(flops), INT64_MAX)
        return result

    def log(self, metrics, *, step, labels, training_seconds, flops=None):
        coordinates = self.coordinates(step, labels, training_seconds, flops)
        if coordinates.keys() != self.writers.keys():
            raise ValueError("Tracking axes changed during a run")
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                for axis, writer in self.writers.items():
                    # Legacy scalar summaries honor restart purging in both TB
                    # event readers. Exact counters remain in JSON/checkpoints;
                    # labels on the horizontal axis use the int64 event step.
                    writer.add_scalar(name, value, coordinates[axis])

    def flush(self):
        for writer in self.writers.values():
            writer.flush()

    def close(self):
        for writer in self.writers.values():
            writer.close()
