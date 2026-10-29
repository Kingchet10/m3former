# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize measured runs without substituting estimates for missing results."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import write_json


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def summarize(run, warmup=10, minimum=20):
    run = Path(run)
    summary = json.loads((run / "research-summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError(f"Run did not complete: {run}")
    rows = read_rows(run / "research-steps.jsonl")
    if len({r["step"] for r in rows}) != len(rows):
        raise ValueError("Use separate directories for benchmark processes")
    measured = rows[warmup:]
    if len(measured) < minimum:
        raise ValueError(f"Need at least {minimum} measured optimizer steps")
    if any(
        not math.isfinite(r["loss"]) or not math.isfinite(r["grad_norm"]) for r in rows
    ):
        raise ValueError("Non-finite training metrics")
    seconds = np.array([r["training_seconds"] for r in measured])
    tokens = np.array([r["valid_labels"] for r in measured])
    if np.any(seconds <= 0) or np.any(tokens <= 0):
        raise ValueError("Invalid measured duration or label count")
    rate = float(tokens.sum() / seconds.sum())
    return {
        "run": str(run),
        "protocol_id": summary["protocol_id"],
        "environment": {
            "gpu": summary.get("gpu"),
            "torch": summary.get("torch"),
            "cuda": summary.get("cuda"),
        },
        "warmup_steps_excluded": warmup,
        "measured_steps": len(measured),
        "valid_labels": int(tokens.sum()),
        "training_seconds": float(seconds.sum()),
        "aggregate_valid_labels_per_second": rate,
        "step_rate_p10_p50_p90": np.quantile(
            tokens / seconds, [0.1, 0.5, 0.9]
        ).tolist(),
        "training_gpu_hours_per_billion_labels": 1e9 / rate / 3600,
        "end_to_end_valid_labels_per_second": summary[
            "end_to_end_valid_labels_per_second"
        ],
        "max_allocated_bytes": max(r["max_allocated_bytes"] for r in measured),
        "max_reserved_bytes": max(r["max_reserved_bytes"] for r in measured),
        "max_forward_retained_increment_bytes": max(
            r["forward_retained_increment_bytes"] for r in measured
        ),
        "nominal_training_tflops_per_second": sum(
            r["nominal_training_flops"] for r in measured
        )
        / seconds.sum()
        / 1e12,
        "model": summary["model"],
        "limitations": (
            "Single-GPU steady training estimate; formal budgets must add measured "
            "validation, checkpoints, startup and retries. FLOPs are analytical "
            "useful work, not hardware instruction counts. GPU hours apply only "
            "to the recorded hardware; this report does not certify absence of "
            "resource contention or suitability for a formal budget."
        ),
    }


def compare_resume(reference, resumed):
    expected = {
        r["step"]: r for r in read_rows(Path(reference) / "research-steps.jsonl")
    }
    rows = read_rows(Path(resumed) / "research-steps.jsonl")
    if not rows:
        raise ValueError("No resumed optimizer steps")
    keys = ("batch_sha256", "valid_labels", "valid_labels_seen", "loss", "grad_norm")
    for row in rows:
        if row["step"] not in expected:
            raise ValueError(f"Missing reference step {row['step']}")
        for key in keys:
            if row[key] != expected[row["step"]][key]:
                raise AssertionError(f"Resume mismatch at step {row['step']}: {key}")
    return {
        "passed": True,
        "compared_steps": [r["step"] for r in rows],
        "exact_fields": list(keys),
        "scope": "Batch hashes and logged scalar equality; final checkpoint tensor equality must be checked separately.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--minimum", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    write_json(a.output, [summarize(r, a.warmup, a.minimum) for r in a.runs])


if __name__ == "__main__":
    main()
