"""Audit saved submission evidence without regenerating or replacing submission files."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import aggregate_predictions, load_prediction_archive
from robust_apex_qd.calibration.model import STRAIN_TO_PATHOGEN, load_measurements
from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.evaluation.seqme_eval import run_seqme_evaluation, validated_embedding_lookup
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.ranking.audit import paired_audit
from robust_apex_qd.ranking.objectives import percentile_score

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze_audit(output: Path) -> None:
    freeze = json.loads((ROOT / "reports/final_freeze.json").read_text())
    expected = {}
    for key in ("representative_full_run", "reproducibility_run"):
        run = freeze[key]
        for name, field in (
            ("library.fasta", "library_fasta_sha256"),
            ("top.fasta", "top_fasta_sha256"),
            ("ranking.tsv", "ranking_tsv_sha256"),
            ("manifest.json", "manifest_sha256"),
        ):
            expected[str(ROOT / run["run_path"] / name)] = run[field]
        manifest = json.loads((ROOT / run["run_path"] / "manifest.json").read_text())
        if manifest["commit"] != freeze["freeze_commit"]:
            raise ValueError("Saved run does not match freeze commit")
        for name, field in (
            ("configs/final.yaml", "config_sha256"),
            ("checkpoint/model.pt", "checkpoint_sha256"),
            ("checkpoint/calibration.json", "calibration_sha256"),
            ("data/training/training.fasta", "training_fasta_sha256"),
            ("data/antibacterial.fasta", "challenge_reference_sha256"),
        ):
            expected[str(ROOT / name)] = manifest[field]
    expected[str(ROOT / freeze["clean_clone_log_path"])] = freeze["clean_clone_log_sha256"]
    config_bytes = subprocess.check_output(
        ["git", "show", f"{freeze['freeze_commit']}:configs/final.yaml"], cwd=ROOT
    )
    if hashlib.sha256(config_bytes).hexdigest() != freeze["final_config_sha256"]:
        raise ValueError("Frozen configuration hash mismatch")
    write_json(
        output / "freeze_verification.json",
        {"freeze_commit": freeze["freeze_commit"], "verified_sha256": verify_hashes(expected)},
    )


def measured_audit(output: Path, iterations: int) -> None:
    path = ROOT / "experimental/mic.csv"
    data = load_measurements(path)
    archive_path = ROOT / "work/calibration_apex_predictions.npz"
    apex_manifest = json.loads((ROOT / "work/calibration_apex_manifest.json").read_text())
    verify_hashes(
        {
            str(archive_path): apex_manifest["predictions_sha256"],
            str(ROOT / "work/calibration_peptides.fasta"): apex_manifest["input_sha256"],
        }
    )
    archive = load_prediction_archive(archive_path)
    ids = data[["peptide_id", "sequence"]].drop_duplicates().sort_values("peptide_id")
    if set(ids.sequence) != set(archive.sequences) or len(set(archive.sequences)) != len(ids):
        raise ValueError("Measured and prediction sequences must match one-to-one")
    panel = data.assign(pathogen=data.strain.map(STRAIN_TO_PATHOGEN))
    if any(
        set(group.pathogen) != set(archive.pathogens) for _, group in panel.groupby("peptide_id")
    ):
        raise ValueError("Each peptide must have the complete prediction pathogen panel")
    aggregates = aggregate_predictions(archive.mic_u_m)
    positions = {sequence: index for index, sequence in enumerate(archive.sequences)}
    index = [positions[sequence] for sequence in ids.sequence]
    base = ids.copy()
    base["score_B0"] = percentile_score(
        aggregates.official_broad_mean_mic_u_m, higher_is_better=False
    )[index]
    base["score_B1"] = percentile_score(aggregates.median_log2_mic, higher_is_better=False)[index]
    results = {}
    rows = []
    for label in ("all", "gram+", "gram-"):
        subset = data if label == "all" else data[data.strain_type == label]
        target = subset.groupby("peptide_id").active.mean()
        frame = base.assign(measured_success_rate_16=base.peptide_id.map(target))
        frame.to_csv(
            output
            / f"peptide_scores_{label.replace('+', 'positive').replace('-', 'negative')}.csv",
            index=False,
            float_format="%.10g",
        )
        results[label] = paired_audit(frame, seed=42, iterations=iterations)
        rows.extend(
            {"group": label, "ranker": ranker, **metrics}
            for ranker, metrics in results[label]["point"].items()
        )
    pd.DataFrame(rows).to_csv(output / "measured_metrics.csv", index=False, float_format="%.10g")
    write_json(output / "paired_bootstrap.json", results)
    strain = panel.groupby(["strain", "pathogen", "strain_type"]).agg(
        measurements=("mic", "size"),
        active=("active", "sum"),
        censored=("mic_relation", lambda values: int((values == ">").sum())),
    )
    strain.to_csv(output / "strain_audit.csv")
    training = set(read_fasta_sequences(ROOT / "data/training/training.fasta"))
    write_json(
        output / "measurement_audit.json",
        {
            "measurements": len(data),
            "peptides": len(ids),
            "strains": data.strain.nunique(),
            "measurement_sha256": file_sha256(path),
            "apex_sha256": file_sha256(archive_path),
            "right_censored": int((data.mic_relation == ">").sum()),
            "censoring_bounds": sorted(data.loc[data.mic_relation == ">", "mic"].unique().tolist()),
            "exact_at_ceiling": int(((data.mic_relation == "=") & (data.mic == 64)).sum()),
            "active": int(data.active.sum()),
            "duplicate_pairs": int(panel.duplicated(["peptide_id", "pathogen"]).sum()),
            "missing_by_column": data.isna().sum().to_dict(),
            "modifications_recorded": sorted(data.modification.dropna().unique().tolist()),
            "medium": sorted(data.medium.dropna().unique().tolist()),
            "exact_diffusion_training_overlap": len(set(ids.sequence) & training),
            "apex_training_overlap": "unknown: exact checkpoint training sequences not verified",
            "limitations": [
                "APEX-selected cohort; post-selection evaluation, not independent validation",
                "Missing assay fields and blank modifications are unknown, not confirmed absent",
                "Censored MIC values are lower bounds; no uncensored MIC quantile claim",
                "B0/B1 full-panel scores held fixed for gram-group target diagnostics",
                "B3-B6 legacy calibrated scores evaluated on calibrator fitting cohort",
                "Undefined correlations remain null; intervals use finite paired replicates",
            ],
        },
    )


def library_audit(run: Path, output: Path) -> None:
    work = run / "work"
    freeze = json.loads((ROOT / "reports/final_freeze.json").read_text())
    if run.resolve() != (ROOT / freeze["representative_full_run"]["run_path"]).resolve():
        raise ValueError("Library comparison requires the frozen representative run")
    freeze_audit(output)
    paths = [path for path in run.rglob("*") if path.is_file()]
    before = {str(path): file_sha256(path) for path in paths}
    write_json(output / "inputs_before.json", before)
    inputs = dict(
        candidates_path=work / "candidates.csv.gz",
        candidate_embeddings_path=work / "candidate_embeddings.npy",
        reference_fasta_path=ROOT / "data/training/training.fasta",
        reference_embeddings_path=work / "reference_embeddings.npy",
        embedding_manifest_path=work / "embedding_manifest.json",
    )
    validated_embedding_lookup(**inputs)
    candidates = pd.read_csv(inputs["candidates_path"])
    for name in ("candidate_physchem.csv.gz", "candidate_embedding_diagnostics.csv.gz"):
        features = pd.read_csv(work / name)
        if features.candidate_id.duplicated().any() or set(features.candidate_id) != set(
            candidates.candidate_id
        ):
            raise ValueError("Feature candidate IDs differ from the frozen pool")
        if (
            "sequence" in features
            and not features.set_index("candidate_id")
            .sequence.reindex(candidates.candidate_id)
            .tolist()
            == candidates.sequence.tolist()
        ):
            raise ValueError("Feature sequence alignment differs from the frozen pool")
    apex = json.loads((work / "apex_manifest.json").read_text())
    verify_hashes(
        {
            str(work / "apex_mean.csv"): apex["aggregates_sha256"],
            str(work / "apex_predictions.npz"): apex["predictions_sha256"],
        }
    )
    variant_paths = {}
    for variant in ("L0", "L1", "L2"):
        destination = output / variant
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/select_library.py"),
                "--variant",
                variant,
                "--output-dir",
                str(destination),
                "--candidates",
                str(work / "candidates.csv.gz"),
                "--physchem",
                str(work / "candidate_physchem.csv.gz"),
                "--embeddings",
                str(work / "candidate_embedding_diagnostics.csv.gz"),
                "--apex",
                str(work / "apex_mean.csv"),
                "--report",
                str(output / "selection.csv"),
            ],
            check=True,
            cwd=ROOT,
        )
        variant_paths[variant] = destination / "library.fasta"
    if file_sha256(variant_paths["L2"]) != file_sha256(run / "library.fasta"):
        raise ValueError("Reselected L2 differs from the frozen library")
    challenge = set(read_fasta_sequences(ROOT / "data/antibacterial.fasta"))
    compliance = {}
    for variant, path in variant_paths.items():
        sequences = read_fasta_sequences(path)
        overlap = len(set(sequences) & challenge)
        if len(sequences) != 50_000 or len(set(sequences)) != 50_000 or overlap:
            raise ValueError("Library full-count, uniqueness or reference overlap failed")
        compliance[variant] = {
            "count": len(sequences),
            "unique": len(set(sequences)),
            "challenge_overlap": overlap,
            "sha256": file_sha256(path),
        }
    write_json(output / "full_library_checks.json", compliance)
    with threadpool_limits(limits=1):
        run_seqme_evaluation(
            variant_paths=variant_paths,
            **inputs,
            seed=42,
            subset_size=1000,
            csv_path=output / "library_comparison.csv",
            markdown_path=output / "library_comparison.md",
            subset_ids_path=output / "subset_ids.tsv",
            adoption_path=output / "library_adoption.json",
        )
    write_json(output / "inputs_after.json", verify_hashes(before))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "measured", "libraries"))
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "work/phase11-clean-0f2f600/full_run_1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protected = [
        args.run_dir,
        ROOT / "experimental",
        ROOT / "data",
        ROOT / "checkpoint",
        ROOT / "work/phase11-clean-0f2f600",
        ROOT / "work/calibration_apex_predictions.npz",
    ]
    fresh_output(output, protected)
    write_json(
        output / "execution.json",
        {
            "mode": args.mode,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "implementation_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in [
                    Path(__file__),
                    ROOT / "src/robust_apex_qd/ranking/audit.py",
                    ROOT / "src/robust_apex_qd/calibration/model.py",
                    ROOT / "scripts/select_library.py",
                    ROOT / "src/robust_apex_qd/evaluation/seqme_eval.py",
                ]
            },
        },
    )
    if args.mode == "freeze":
        freeze_audit(output)
    elif args.mode == "measured":
        measured_audit(output, args.iterations)
    else:
        library_audit(args.run_dir.resolve(), output)
    print(f"Completed {args.mode}: {output}", flush=True)


if __name__ == "__main__":
    main()
