import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from robust_apex_qd.apex.ensemble import load_prediction_archive
from robust_apex_qd.calibration.model import (
    build_calibration_rows,
    evaluate_calibration,
    fit_calibration_artifact,
    load_measurements,
)
from robust_apex_qd.io.fasta import FastaRecord, write_fasta

ROOT = Path(__file__).resolve().parents[3]


def _load_calibration_config(path: Path) -> dict[str, int]:
    payload = yaml.safe_load(path.read_text())
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Configuration must contain a calibration mapping")
    return {
        "seed": int(calibration["seed"]),
        "folds": int(calibration["folds"]),
        "bootstrap_iterations": int(calibration["bootstrap_iterations"]),
    }


def _write_experimental_fasta(measurements: pd.DataFrame, path: Path) -> None:
    peptides = measurements[["peptide_id", "sequence"]].drop_duplicates()
    if peptides["peptide_id"].duplicated().any() or peptides["sequence"].duplicated().any():
        raise ValueError("Each experimental peptide ID and sequence must map one-to-one")
    write_fasta(
        [
            FastaRecord(str(row.peptide_id), str(row.sequence))
            for row in peptides.itertuples(index=False)
        ],
        path,
    )


def run_calibration(
    *,
    config_path: Path,
    measurements_path: Path,
    apex_predictions_path: Path,
    oof_path: Path,
    summary_path: Path,
    artifact_path: Path,
) -> dict[str, object]:
    config = _load_calibration_config(config_path)
    measurements = load_measurements(measurements_path)
    fasta_path = apex_predictions_path.with_name("calibration_peptides.fasta")
    _write_experimental_fasta(measurements, fasta_path)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "apex/APEX_predict_ensemble.py"),
            "--input",
            str(fasta_path),
            "--output",
            str(apex_predictions_path),
            "--aggregates",
            str(apex_predictions_path.with_name("calibration_apex_mean.csv")),
            "--manifest",
            str(apex_predictions_path.with_name("calibration_apex_manifest.json")),
        ],
        cwd=ROOT,
        check=True,
    )
    rows = build_calibration_rows(measurements, load_prediction_archive(apex_predictions_path))
    evaluation = evaluate_calibration(rows, **config)
    variants = evaluation.summary["variants"]
    adopted = evaluation.summary["adopted_variant"]
    candidate_variant = adopted or min(
        ("C1", "C2"),
        key=lambda name: variants[name]["metrics"]["brier"],
    )
    artifact = fit_calibration_artifact(rows, evaluation, variant=str(candidate_variant))
    summary = dict(evaluation.summary)
    summary["fitted_candidate"] = {
        "variant": artifact.variant,
        "feature_order": list(artifact.feature_order),
        "coefficients": list(artifact.coefficients),
        "coefficient_signs": [
            "positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in artifact.coefficients
        ],
        "intercept": artifact.intercept,
    }
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.oof.to_csv(oof_path, index=False, float_format="%.10g", lineterminator="\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(artifact.model_dump_json(indent=2) + "\n")
    return summary
