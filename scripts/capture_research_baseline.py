"""Verify saved inputs and exercise the official checkpoint in a fresh research run."""

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import torch.version
from audit_submission_readiness import freeze_audit

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.generation.sampler import make_official_backend, set_process_determinism
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.validation.compliance import validate_records, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [Path("data"), Path("checkpoint"), Path("reports")])
    start = time.monotonic()
    freeze_audit(args.output)
    freeze = json.loads(Path("reports/final_freeze.json").read_text())
    run = Path(freeze["representative_full_run"]["run_path"])
    references = set(read_fasta_sequences(Path("data/antibacterial.fasta")))
    validation = validate_submission(run, references)
    if not validation.is_valid:
        raise ValueError(f"Saved submission validation failed: {validation.issues}")
    candidates = pd.read_csv(run / "work/candidates.csv.gz")
    library = set(read_fasta_sequences(run / "library.fasta"))
    top = set(read_fasta_sequences(run / "top.fasta"))
    if len(candidates) != 60_000 or not top <= library <= set(candidates.sequence):
        raise ValueError("Saved pool, library and Top do not match")
    paths = [
        Path("reports/final_freeze.json"),
        Path("uv.lock"),
        Path("pyproject.toml"),
        *sorted(Path("configs").glob("*.yaml")),
        Path("checkpoint/model.pt"),
        *sorted(Path("apex/APEX_pathogen_models").glob("*")),
        *sorted(path for path in run.rglob("*") if path.is_file()),
    ]
    lfs_paths = [Path("checkpoint/model.pt"), *sorted(Path("apex/APEX_pathogen_models").glob("*"))]
    for path in lfs_paths:
        pointer = subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True)
        expected = next(
            line.removeprefix("oid sha256:")
            for line in pointer.splitlines()
            if line.startswith("oid sha256:")
        )
        if file_sha256(path) != expected:
            raise ValueError(f"Weight does not match the Git LFS object: {path}")
    manifest = {
        "schema_version": 1,
        "research_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "capture_script_sha256": file_sha256(Path(__file__)),
        "freeze_commit": freeze["freeze_commit"],
        "policy": {"ranker": "B1", "library": "L2", "calibration": "C0"},
        "pool_rows": len(candidates),
        "library_rows": len(library),
        "top_rows": len(top),
        "saved_submission_validation": "passed",
        "lfs_weights_verified": len(lfs_paths),
        "sha256": {str(path): file_sha256(path) for path in paths if path.is_file()},
    }
    (args.output / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "pandas", "biopython", "fair-esm", "pydantic"]
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "seed": 42,
        "batch_size": 1,
        "design_length": 20,
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("Official checkpoint smoke requires CUDA; baseline hashes are saved")
    set_process_determinism(42)
    torch.cuda.reset_peak_memory_stats()
    backend = make_official_backend(Path("checkpoint/model.pt"), "cuda")
    sequences = backend(design_length=20, batch_size=1, round_seed=42)
    records = [FastaRecord(f"smoke_{i}", seq) for i, seq in enumerate(sequences)]
    write_fasta(records, args.output / "checkpoint_smoke.fasta")
    report = validate_records(records, references, check_similarity=True)
    environment.update(
        {
            "gpu": torch.cuda.get_device_name(),
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "runtime_seconds": time.monotonic() - start,
            "checkpoint_smoke_sha256": file_sha256(args.output / "checkpoint_smoke.fasta"),
            "smoke_sequence_validation": "passed" if report.is_valid else "failed",
            "validation_issues": [issue.message for issue in report.issues],
        }
    )
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if not report.is_valid:
        raise ValueError("Checkpoint smoke sequence validation failed; see environment.json")
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
