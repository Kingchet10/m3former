# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit checkpoint lineage for changing only the execution layout."""

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from .artifacts import file_sha256, identity, write_json


def scientific_signature(config):
    training = dict(config["training"])
    training.pop("local_batch_size")
    signature = {
        "architecture": config["architecture"],
        "training": training,
        "token_budget": config["token_budget"],
        "optimizer": config["optimizer"],
        "lr_scheduler": config["lr_scheduler"],
        "loss": config["loss"],
        "compile": config["compile"],
        "activation_checkpoint": config["activation_checkpoint"],
        "seed": config["debug"]["seed"],
        "deterministic": config["debug"]["deterministic"],
        "infinite": config["dataloader"]["infinite"],
    }
    # Config objects use tuples; saved JSON uses lists for the same values.
    return json.loads(json.dumps(signature))


def checkpoint_bytes(path):
    metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    result = {
        key: None
        for key, value in metadata.items()
        if isinstance(value, BytesStorageMetadata)
        and key.startswith(("train_state.", "dataloader.", "lr_scheduler."))
    }
    dcp.load(result, checkpoint_id=path)
    return result


def make_lineage(parent_run, checkpoint, output):
    parent_run, checkpoint, output = map(
        lambda p: Path(p).resolve(), (parent_run, checkpoint, output)
    )
    if output.exists():
        raise FileExistsError("A lineage record is immutable")
    config = json.loads((parent_run / "config.json").read_text())
    tracking = json.loads(
        (parent_run / "research-tensorboard/run-identity.json").read_text()
    )
    state = checkpoint_bytes(checkpoint)
    step = state["train_state.step"]
    if checkpoint != parent_run / "checkpoint" / f"step-{step}":
        raise ValueError("Checkpoint does not belong to the stated source run")
    if state["train_state.research_protocol_id"] != tracking["protocol_id"]:
        raise ValueError("Checkpoint protocol differs from its source tracking record")
    value = {
        "parent_run": str(parent_run),
        "parent_checkpoint": str(checkpoint),
        "parent_commit": tracking["git_commit"],
        "parent_protocol_id": tracking["protocol_id"],
        "artifact_id": tracking["artifact_id"],
        "tokenizer_sha256": tracking["tokenizer_sha256"],
        "parent_config_sha256": file_sha256(parent_run / "config.json"),
        "scientific_signature": scientific_signature(config),
        "checkpoint_state": {
            key: val for key, val in state.items() if key.startswith("train_state.")
        },
        "checkpoint_files": {
            str(p.relative_to(checkpoint)): file_sha256(p)
            for p in sorted(checkpoint.rglob("*"))
            if p.is_file()
        },
        "policy": "Full optimizer/scheduler/canonical loader restore; global batch and scientific recipe unchanged",
    }
    value["lineage_id"] = identity(value)
    write_json(output, value)
    return value


def validate_lineage(config, manifest):
    path = Path(config.resume_lineage)
    lineage = json.loads(path.read_text())
    declared = lineage.pop("lineage_id")
    if identity(lineage) != declared:
        raise ValueError("Resume lineage content identity mismatch")
    lineage["lineage_id"] = declared
    parent = Path(lineage["parent_run"])
    checkpoint = Path(lineage["parent_checkpoint"])
    if (
        config.checkpoint.initial_load_model_only
        or config.checkpoint.initial_load_path is None
    ):
        raise ValueError("Layout migration requires a full-state initial checkpoint")
    if Path(config.checkpoint.initial_load_path).resolve() != checkpoint:
        raise ValueError("Resume checkpoint differs from the lineage record")
    if (
        lineage["scientific_signature"] != scientific_signature(config.to_dict())
        or lineage["artifact_id"] != manifest["artifact_id"]
        or lineage["tokenizer_sha256"] != manifest["tokenizer"]["tokenizer_sha256"]
    ):
        raise ValueError(
            "Scientific configuration or data changed during layout migration"
        )
    if file_sha256(parent / "config.json") != lineage["parent_config_sha256"]:
        raise ValueError("Source configuration was modified")
    current = {
        str(p.relative_to(checkpoint)): file_sha256(p)
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file()
    }
    if current != lineage["checkpoint_files"]:
        raise ValueError("Source checkpoint content changed")
    return lineage


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def verify_restored(parent, restored, output):
    from .engineering import compare_all_tensors

    tensors = compare_all_tensors(Path(parent), Path(restored))
    old, new = checkpoint_bytes(parent), checkpoint_bytes(restored)
    if old.keys() != new.keys():
        raise ValueError("Restored auxiliary state keys differ")
    checked = []
    for key in old:
        if key == "train_state.research_protocol_id":
            continue
        left, right = old[key], new[key]
        if key.startswith("dataloader.dp_rank_"):
            left, right = pickle.loads(left), pickle.loads(right)
        if not nested_equal(left, right):
            raise ValueError("Restored auxiliary state differs: " + key)
        checked.append(key)
    proof = {
        "status": "passed",
        "all_model_optimizer_tensors": tensors,
        "auxiliary_keys_equal": checked,
        "parent_protocol_id": old["train_state.research_protocol_id"],
        "resumed_protocol_id": new["train_state.research_protocol_id"],
        "step": old["train_state.step"],
        "valid_labels": old["train_state.research_valid_tokens"],
    }
    write_json(output, proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("lineage")
    build.add_argument("--parent-run", required=True)
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--parent", required=True)
    verify.add_argument("--restored", required=True)
    verify.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "lineage":
        print(
            json.dumps(
                make_lineage(args.parent_run, args.checkpoint, args.output), indent=2
            )
        )
    else:
        print(
            json.dumps(
                verify_restored(args.parent, args.restored, args.output), indent=2
            )
        )


if __name__ == "__main__":
    main()
