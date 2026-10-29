# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serial, bounded GPU acceptance; run inside the cluster resource guard."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import write_json


def commands(output):
    """Build commands without initializing CUDA, for review and CLI checks."""
    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "torchtitan.train",
    ]
    c4 = launch + [
        "--module",
        "llama3",
        "--config",
        "llama3_debugmodel_ce_loss",
        "--training.seq-len",
        "256",
        "--training.local-batch-size",
        "1",
        "--training.global-batch-size",
        "1",
        "--training.steps",
        "10",
        "--lr-scheduler.warmup-steps",
        "2",
        "--parallelism.data-parallel-shard-degree",
        "1",
        "--training.disable-cuda-graphs",
        "--checkpoint.enable",
        "--checkpoint.interval",
        "5",
        "--checkpoint.no-last-save-model-only",
        "--validator.no-enable",
        "--debug.seed",
        "42",
        "--debug.deterministic",
        "--debug.save-config-file",
        "config.json",
    ]
    dense = launch + [
        "--module",
        "torchtitan.experiments.brainfuse_scaling",
        "--config",
        "dense_30m",
        "--training.seq-len",
        "1024",
        "--training.local-batch-size",
        "2",
        "--training.global-batch-size",
        "16",
        "--training.steps",
        "40",
        "--lr-scheduler.warmup-steps",
        "4",
        "--checkpoint.interval",
        "20",
        "--validator.freq",
        "20",
        "--validator.steps",
        "2",
        "--audit-batches",
        "--debug.deterministic",
    ]

    def into(command, name):
        return command + ["--dump-folder", str(output / name)]

    hundred = dense.copy()
    hundred[hundred.index("dense_30m")] = "dense_100m"
    hundred[hundred.index("--training.local-batch-size") + 1] = "1"
    hundred[hundred.index("--training.steps") + 1] = "30"
    return [
        ("c4-full", into(c4, "c4-full")),
        (
            "c4-resumed",
            into(c4, "c4-resumed")
            + [
                "--checkpoint.initial-load-path",
                str(output / "c4-full/checkpoint/step-5"),
                "--checkpoint.no-initial-load-model-only",
            ],
        ),
        ("dense-full", into(dense, "dense-full")),
        (
            "dense-interrupted",
            into(dense, "dense-interrupted") + ["--stop-after-step", "20"],
        ),
        (
            "dense-resumed",
            into(dense, "dense-resumed")
            + [
                "--checkpoint.initial-load-path",
                str(output / "dense-interrupted/checkpoint/step-20"),
                "--checkpoint.no-initial-load-model-only",
            ],
        ),
        ("dense-100m", into(hundred, "dense-100m")),
    ]


def compare_all_tensors(reference, resumed):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    def load(path):
        metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        state = {
            key: torch.empty(value.size, dtype=value.properties.dtype)
            for key, value in metadata.items()
            if isinstance(value, TensorStorageMetadata)
        }
        dcp.load(state, checkpoint_id=path)
        return state

    expected, actual = load(reference), load(resumed)
    if not expected or expected.keys() != actual.keys():
        raise AssertionError("Checkpoint tensor keys differ or are empty")
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor, atol=0, rtol=0, msg=key)
    return {
        "passed": True,
        "tensors_compared": len(expected),
        "atol": 0,
        "rtol": 0,
        "scope": "All checkpoint tensor leaves, including model and optimizer; non-tensor state excluded",
    }


def check_recovery(output):
    from .report import compare_resume

    report = {
        "c4_tensors": compare_all_tensors(
            output / "c4-full/checkpoint/step-10",
            output / "c4-resumed/checkpoint/step-10",
        ),
        "dense_interrupted_prefix": compare_resume(
            output / "dense-full", output / "dense-interrupted"
        ),
        "dense_resumed_steps": compare_resume(
            output / "dense-full", output / "dense-resumed"
        ),
        "dense_tensors": compare_all_tensors(
            output / "dense-full/checkpoint/step-40",
            output / "dense-resumed/checkpoint/step-40",
        ),
    }
    write_json(output / "recovery-validation.json", report)


def run_phase(command, log, env, timeout):
    def interrupted(signum, frame):
        raise InterruptedError(f"Acceptance runner received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return child.wait(timeout=timeout)
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase-timeout", type=int, default=1200)
    parser.add_argument("--check-recovery-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    planned = commands(output)
    if args.dry_run:
        print(json.dumps(dict(planned), indent=2))
        return
    if args.check_recovery_only:
        check_recovery(output)
        return
    if args.phase_timeout <= 0:
        raise ValueError("phase-timeout must be positive")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    write_json(output / "commands.json", dict(planned))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Commit the acceptance code before GPU execution")
    record = {
        "git_commit": source,
        "scope": "engineering_only",
        "status": "running",
        "phases": [],
    }
    write_json(output / "acceptance.json", record)
    try:
        for name, command in planned:
            started = time.monotonic()
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.log").open("x") as log:
                returncode = run_phase(command, log, env, args.phase_timeout)
            record["phases"].append(
                {
                    "name": name,
                    "returncode": returncode,
                    "seconds": time.monotonic() - started,
                }
            )
            write_json(output / "acceptance.json", record)
            if returncode:
                raise RuntimeError(f"{name} failed; inspect its retained log")
            if name == "dense-resumed":
                check_recovery(output)
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        write_json(output / "acceptance.json", record)


if __name__ == "__main__":
    main()
