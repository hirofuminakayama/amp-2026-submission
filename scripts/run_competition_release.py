"""Freeze effective local inputs and execute two independently generated default-size outputs."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.pipeline import ROOT
from robust_apex_qd.validation.compliance import require_valid_submission
from robust_apex_qd.validation.release import verify_run_identity


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_inputs(inputs: dict[str, str]) -> None:
    for name, digest in inputs.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/final.yaml"
    config = yaml.safe_load(config_path.read_text())
    sources = sorted(
        set(ROOT.glob("src/**/*.py"))
        | set(ROOT.glob("scripts/*.py"))
        | {
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            config_path,
            ROOT / "configs/inference_encoders.json",
        }
    )
    assets = list((ROOT / config["inference"]["predictor_assets"]).rglob("*"))
    assets += list((ROOT / "apex/APEX_pathogen_models").glob("*"))
    assets += list((ROOT / "apex").glob("*.py"))
    assets += [
        ROOT / config["generation"]["checkpoint"],
        ROOT / config["references"]["known_amp_fasta"],
        ROOT / config["references"]["challenge_fasta"],
    ]
    assets += list((Path(torch.hub.get_dir()) / "checkpoints").glob("esm2_t6_8M_UR50D*.pt"))
    inputs = {
        str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): file_sha256(p)
        for p in [*sources, *assets]
        if p.is_file()
    }
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    freeze = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=inputs,
        config=config,
        scope="effective local source and assets; not a remote clean-clone validation",
        entrypoint="uv run --frozen generate; only output directory differs between runs",
    )
    write_json(output / "input_freeze.json", freeze)
    executions = []
    references = set(read_fasta_sequences(ROOT / config["references"]["challenge_fasta"]))
    for index in [1, 2]:
        verify_inputs(inputs)
        destination = output / f"full_run_{index}"
        command = ["uv", "run", "--frozen", "generate", "--output-dir", str(destination)]
        started = time.monotonic()
        record: dict[str, Any] = dict(
            run=index,
            command=command,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        executions.append(record)
        write_json(output / "execution.json", dict(runs=executions, status="running"))
        print(f"Starting independent full run {index}", flush=True)
        with (output / f"full_run_{index}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, wall_seconds=time.monotonic() - started)
        if result.returncode:
            record["status"] = "failed"
            write_json(output / "execution.json", dict(runs=executions, status="failed"))
            raise RuntimeError(f"Full run {index} failed; inspect its preserved log")
        verify_inputs(inputs)
        manifest = json.loads((destination / "manifest.json").read_text())
        expected = dict(
            seed=config["seed"],
            raw_count=config["generation"]["raw_pool_size"],
            library_count=50000,
            top_count=100,
            config_sha256=inputs["configs/final.yaml"],
            sampling_steps=config["generation"]["sampling_steps"],
            inference_policy=config["inference"],
        )
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Actual run differs from the frozen default configuration")
        require_valid_submission(destination, references, library_size=50000, top_k=100)
        validator = [
            sys.executable,
            "scripts/verify_existing_output.py",
            "--output-dir",
            str(destination),
            "--antibacterial-fasta",
            config["references"]["challenge_fasta"],
        ]
        with (output / f"official_{index}.log").open("w") as log:
            subprocess.run(validator, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        record.update(status="passed", local_validator=True, official_validator=True)
        write_json(output / "execution.json", dict(runs=executions, status="running"))
    repeated = verify_run_identity(output / "full_run_1", output / "full_run_2", inputs)
    write_json(output / "validation_report.json", dict(**repeated, runs=executions))
    write_json(output / "execution.json", dict(runs=executions, status="passed"))
    write_json(
        output / "release_manifest.json",
        dict(
            status="local_reproduction_passed; external publication and submission not performed",
            input_freeze_sha256=file_sha256(output / "input_freeze.json"),
            validation_sha256=file_sha256(output / "validation_report.json"),
            new_generation_runs=2,
            output_sha256=repeated["output_sha256"],
            remote_clean_clone=False,
            full_tier="source disclosure and repository access remain separately assessed",
        ),
    )


if __name__ == "__main__":
    main()
