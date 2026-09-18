import json
import subprocess
import sys
from pathlib import Path

import yaml

from robust_apex_qd.io.fasta import read_fasta

ROOT = Path(__file__).resolve().parents[2]


def test_baseline_config_preserves_upstream_defaults() -> None:
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    assert config["generation"]["raw_pool_size"] == 50_000
    assert config["generation"]["length_policy"] == "uniform"
    assert config["ranking"]["official_challenge_similarity_max"] == 0.80


def test_lfs_assets_are_materialized() -> None:
    assert (ROOT / "checkpoint/model.pt").stat().st_size == 132_526_179
    weights = sorted((ROOT / "apex/APEX_pathogen_models").iterdir())
    assert len(weights) == 8
    assert all(path.stat().st_size > 1_000_000 for path in weights)


def test_generate_cli_has_no_required_arguments() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "robust_apex_qd.cli", "generate", "--show-defaults"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    defaults = json.loads(result.stdout)
    assert defaults["raw_pool_size"] == 120_000
    assert defaults["library_size"] == 50_000
    assert defaults["top_k"] == 100


def test_smoke_writes_four_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robust_apex_qd.cli",
            "smoke",
            "--raw-pool-size",
            "40",
            "--library-size",
            "32",
            "--top-k",
            "8",
            "--batch-size",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    assert {path.name for path in output_dir.iterdir() if path.is_file()} == {
        "library.fasta",
        "top.fasta",
        "ranking.tsv",
        "manifest.json",
    }
    assert len(read_fasta(output_dir / "library.fasta")) == 32
    assert len(read_fasta(output_dir / "top.fasta")) == 8
