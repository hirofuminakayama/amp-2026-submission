import json
from pathlib import Path

import pytest

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.validation.release import verify_run_identity


def test_repeat_identity_checks_actual_files_and_frozen_inputs(tmp_path: Path) -> None:
    asset = tmp_path / "weights"
    asset.write_bytes(b"frozen")
    frozen = {str(asset): file_sha256(asset)}
    runs = []
    for name in ["first", "second"]:
        root = tmp_path / name
        root.mkdir()
        for artifact in ["library.fasta", "top.fasta", "ranking.tsv"]:
            (root / artifact).write_text("same\n")
        manifest = dict(
            seed=42,
            raw_count=120000,
            library_count=50000,
            top_count=100,
            config_sha256="config",
            checkpoint_sha256="checkpoint",
            challenge_reference_sha256="challenge",
            training_fasta_sha256="training",
            inference_policy={"policy": "ddim-lref-rankmean"},
            sampling_steps=250,
            inference_assets_sha256=frozen,
            output_sha256={p.name: file_sha256(p) for p in root.iterdir()},
        )
        (root / "manifest.json").write_text(json.dumps(manifest))
        runs.append(root)
    assert verify_run_identity(runs[0], runs[1], frozen)["byte_identical"]
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input"):
        verify_run_identity(runs[0], runs[1], frozen)
    asset.write_bytes(b"frozen")
    (runs[1] / "ranking.tsv").write_text("different\n")
    with pytest.raises(ValueError, match="output"):
        verify_run_identity(runs[0], runs[1], frozen)


def test_repeat_identity_rejects_reusing_the_same_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        verify_run_identity(tmp_path, tmp_path, {})
