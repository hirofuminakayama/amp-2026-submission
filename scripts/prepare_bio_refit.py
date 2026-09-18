"""Prepare corrected endpoint refits without changing historical features or observations."""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from run_competition_bioaccuracy import (
    archive_sources,
    checked_manifest,
    finish_stage,
    read_observations,
    write_json,
)

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.bio_followup import chemistry_status, refit_mic_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path("work/competition_bioaccuracy/20260913-b")
    fresh_output(args.output, [base, args.audit, Path("data"), Path("checkpoint")])
    started = time.monotonic()
    inputs = archive_sources(
        args.output,
        [
            Path(__file__),
            Path("src/robust_apex_qd/research/bio_followup.py"),
            Path("configs/mic_research.json"),
            Path("configs/competition_bioaccuracy.json"),
            Path("uv.lock"),
        ],
    )
    inputs.update(checked_manifest(args.audit / "manifest.json"))
    inputs.update(checked_manifest(base / "split/manifest.json"))
    corrected = read_observations(args.audit / "corrected_endpoint_observations.jsonl")
    split = json.loads((args.audit / "split_manifest.json").read_text())
    original = pd.read_json(base / "split/rows.jsonl", lines=True)
    mic = refit_mic_rows(original, corrected, split)
    prior = Path(json.loads(Path("configs/mic_research.json").read_text())["prior_models"])
    fasta = prior / "prepare/sequences.fasta"
    inputs[str(fasta)] = file_sha256(fasta)
    feature_sequences = read_fasta_sequences(fasta)
    if any(
        feature_sequences[int(index)] != sequence
        for index, sequence in zip(mic.sequence_index, mic.sequence, strict=True)
    ):
        raise ValueError("Original MIC feature row mapping differs from the sequence")
    split_root = args.output / "split"
    split_root.mkdir()
    mic.to_json(split_root / "rows.jsonl", orient="records", lines=True)
    shutil.copyfile(args.audit / "split_manifest.json", split_root / "split_manifest.json")
    finish_stage(split_root, inputs, started)
    prepare = args.output / "prepare"
    prepare.mkdir()
    retained = [r for r in corrected if chemistry_status(r) != "known_modified"]
    for name, rows in [
        ("endpoint_observations", retained),
        ("hc50_observations", [r for r in retained if r.endpoint.endswith("hc50")]),
    ]:
        (prepare / f"{name}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
    write_json(
        prepare / "cohort.json",
        dict(
            mic_rows=len(mic),
            mic_exact_rows=int(mic.exact_regression.sum()),
            human_hc50=sum(
                r.endpoint == "measured_hc50" and r.rbc_species == "human" for r in retained
            ),
            scope=(
                "Endpoint refit diagnostics retain unknown stereochemistry; "
                "primary selection remains separate"
            ),
            unknown_chemistry_mic=int(mic.reviewed_chemistry.eq("unknown").sum()),
            removed_mic_ids=sorted(
                set(original[original.objective.eq("measured_mic")].observation_id)
                - set(mic.observation_id)
            ),
        ),
    )
    finish_stage(prepare, inputs, started)
    for stage in ["features", "hc50-embeddings"]:
        inputs.update(checked_manifest(base / stage / "manifest.json"))
        shutil.copytree(base / stage, args.output / stage)
    for seed in [42, 43, 44]:
        config = json.loads(Path("configs/mic_research.json").read_text())
        config.update(seeds=[seed], device="cpu")
        write_json(args.output / f"mic-s{seed}.json", config)
    write_json(
        args.output / "refit_protocol.json",
        dict(
            scope="Corrected endpoint refits, not complete nested joint procedure selection",
            device="cpu",
            cpu_threads=2,
            epochs=80,
            seeds=[42, 43, 44],
            primary_coverage=json.loads((args.audit / "primary_coverage.json").read_text()),
            gpu_hours_required=0,
            existing_fit_reuse=False,
        ),
    )
    finish_stage(args.output, inputs, started)
    print(json.dumps(dict(root=str(args.output), mic_rows=len(mic))))


if __name__ == "__main__":
    main()
