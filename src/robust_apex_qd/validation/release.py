"""Verify repeat output bytes against recorded runs and frozen effective inputs."""

import json
from pathlib import Path
from typing import Any

from robust_apex_qd.features.embeddings import file_sha256

ARTIFACTS = ("library.fasta", "top.fasta", "ranking.tsv")
IDENTITY_FIELDS = (
    "seed",
    "raw_count",
    "library_count",
    "top_count",
    "config_sha256",
    "checkpoint_sha256",
    "challenge_reference_sha256",
    "training_fasta_sha256",
    "inference_policy",
    "inference_assets_sha256",
    "sampling_steps",
)


def verify_run_identity(first: Path, second: Path, inputs: dict[str, str]) -> dict[str, Any]:
    if first.resolve() == second.resolve():
        raise ValueError("Require distinct actual run directories")
    for name, digest in inputs.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in [first, second]]
    for field in IDENTITY_FIELDS:
        if field not in manifests[0] or manifests[0][field] != manifests[1].get(field):
            raise ValueError(f"Repeated generation differs in {field}")
    actual = []
    for root, manifest in zip([first, second], manifests, strict=True):
        hashes = {name: file_sha256(root / name) for name in ARTIFACTS}
        if hashes != manifest["output_sha256"]:
            raise ValueError("Run output differs from its manifest")
        for name, digest in manifest["inference_assets_sha256"].items():
            if inputs.get(name) != digest:
                raise ValueError("Run inference asset is not in the frozen inputs")
        actual.append(hashes)
    if actual[0] != actual[1]:
        raise ValueError("Repeated output bytes differ")
    return dict(byte_identical=True, output_sha256=actual[0], inputs_verified=len(inputs))
