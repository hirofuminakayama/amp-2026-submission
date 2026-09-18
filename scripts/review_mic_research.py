"""Verify MIC experiment artifacts and summarize measured results and outstanding work."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from run_mic_research import checked_manifest, finish_stage, write_json

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    # The report may be a fresh child of the run, but never overwrite an input stage.
    if args.output.exists() or args.output.resolve() == args.root.resolve():
        parser.error("A fresh report directory is required")
    stages = [
        "audit-r2",
        "prepare",
        "models",
        "repeat-s43",
        "repeat-s44",
        "baselines-r2",
        "handoff",
        "selection",
        "extensions",
    ]
    fresh_output(args.output, [args.root / stage for stage in stages])
    inputs = {}
    manifests = []
    for stage in stages:
        if not (args.root / stage / "manifest.json").is_file():
            raise ValueError(f"Incomplete stage: {stage}")
        for manifest in sorted((args.root / stage).rglob("*manifest.json")):
            payload = json.loads(manifest.read_text())
            if "artifacts_sha256" in payload:
                inputs.update(checked_manifest(manifest))
                manifests.append(str(manifest))
    frames = [
        pd.read_csv(args.root / stage / "model_comparison.csv")
        for stage in ["models", "repeat-s43", "repeat-s44"]
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(args.output / "all_seed_metrics.csv", index=False)
    summary = (
        metrics.groupby("family")
        .agg(
            seeds=("seed", "nunique"),
            macro_mae_mean=("macro_mae", "mean"),
            macro_mae_std=("macro_mae", "std"),
            top20_mean=("macro_top20", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "model_summary.csv", index=False)
    baselines = pd.read_csv(args.root / "baselines-r2/baseline_metrics.csv")
    baseline = baselines[(baselines.model == "linear8") & (baselines.cohort == "all")].iloc[0]
    variants = []
    for path in sorted((args.root / "selection/tops").glob("*/manifest.json")):
        row = json.loads(path.read_text())
        if row["status"] != "complete":
            raise ValueError(f"Incomplete selection: {path}")
        variants.append(row["id"])
    write_json(
        args.output / "verification.json",
        dict(
            verified_manifests=len(manifests),
            verified_files=len(inputs),
            neural_fits=sum(
                len(list((args.root / stage / "fits").rglob("fit.json")))
                for stage in ["models", "repeat-s43", "repeat-s44"]
            ),
            baseline_fits=len(list((args.root / "baselines-r2/fits").glob("*/manifest.json"))),
            validated_candidates=variants,
            adopted=False,
            excluded_run="baselines; only completed baselines-r2 is evaluated",
        ),
    )
    splits = json.loads((args.root / "prepare/split_manifest.json").read_text())
    handoff = json.loads((args.root / "handoff/handoff.json").read_text())
    bounds = metrics.coverage90_exact.dropna()
    pool_count = pd.read_csv(args.root / "handoff/ensemble_sweep.csv").sequences.iloc[0]
    (args.output / "report.md").write_text(
        "# MIC research results\n\n"
        "Status: initial screen complete; overall phased plan in progress.\n\n"
        f"Development evaluation: {len(splits['outer'])} sequences, "
        f"{len(set(splits['groups'].values()))} groups, "
        f"identity threshold {splits['threshold']}. "
        "Prior holdout reuse and unknown APEX training "
        "overlap remain limitations. This is not official QMAP reproduction.\n\n"
        "## Measured outcomes\n\n"
        f"Nested linear8 macro MAE: {baseline.macro_mae:.6f} log2 uM; "
        f"Top20: {baseline.macro_top20:.6f}.\n\n"
        "```csv\n" + summary.to_csv(index=False) + "```\n\n"
        f"Observed nominal 90% interval coverage range: {bounds.min():.3f}-{bounds.max():.3f}. "
        "Compare quantitative error, Top20 and interval calibration separately. "
        "These results do not prove new-candidate wet-lab benefit.\n\n"
        "## Handoff\n\n"
        f"{len(handoff['selected'])} numeric bundles and {pool_count} sequences are in `handoff/`. "
        f"The selection pipeline validated {len(variants)} library/Top candidates. "
        "APEX-only ordered Tops reproduce the control. "
        "No new method is adopted and no full generation reproducibility is claimed.\n\n"
        "## Outstanding work\n\n"
        "- Rerun the fine-tuned baseline under OOD60; old 80% results are not interchangeable.\n"
        "- Validate the official QMAP engine and predefined tests separately.\n"
        "- Curate assay-to-paper linkage for publication sensitivity and strict delta pairs. "
        "Current exports have no verified study mapping; reference tokens need validation.\n"
        "- Acquire and validate strain-genome accessions before the pair/genome pilot.\n"
        "- Evaluate missing-assay inference before deploying assay-aware heads.\n"
        "- Complete HC50/joint-hit comparison, final adoption and two full reproducibility runs "
        "through the biological-accuracy finalization workflow.\n"
    )
    inputs[str(Path(__file__))] = file_sha256(Path(__file__))
    finish_stage(args.output, inputs, start)


if __name__ == "__main__":
    main()
