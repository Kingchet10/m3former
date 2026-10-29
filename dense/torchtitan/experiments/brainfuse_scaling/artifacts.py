# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Content identities shared by preparation, training, and evaluation."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_manifest(root, *, verify=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    declared = manifest.pop("artifact_id")
    if identity(manifest) != declared:
        raise ValueError("Manifest identity mismatch")
    manifest["artifact_id"] = declared
    if verify:
        for relative, expected in manifest["files"].items():
            if file_sha256(root / relative) != expected:
                raise ValueError(f"Artifact checksum mismatch: {relative}")
    return manifest
