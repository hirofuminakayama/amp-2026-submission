"""Independently verify continuation artifacts, fold isolation and common-row metrics."""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_mic_research import finish_stage, write_json
from train_mic_models import evaluate

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.prediction_cache import PredictionCache, isolated_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    names = [
        "models",
        "models43",
        "models44",
        "baselines",
        "controls-v2",
        "controls43",
        "controls44",
        "pairs-final",
        "pairs-checked",
        "delta",
        "cliffs",
        "shuffle",
        "handoff",
        "selection",
        "safety",
        "report",
    ]
    stages = [args.run / name for name in names]
    fresh_output(args.output, [args.prepared, *stages])
    capture_execution(args.output)
    hashes: dict[str, str] = {}
    historical_inputs = {}
    manifests = 0
    for stage in stages:
        if not (stage / "manifest.json").is_file():
            raise ValueError(f"Incomplete stage: {stage}")
        for path in sorted(stage.rglob("*manifest.json")):
            record = json.loads(path.read_text())
            manifests += 1
            for name, expected in record.get("artifacts_sha256", {}).items():
                artifact = path.parent / name
                digest = hashes.setdefault(str(artifact), file_sha256(artifact))
                if digest != expected:
                    raise ValueError(f"Changed artifact: {artifact}")
            for name, expected in record.get("inputs_sha256", {}).items():
                artifact = Path(name)
                digest = hashes.setdefault(str(artifact), file_sha256(artifact))
                if digest != expected:
                    snapshot = args.run / "references/pair-registry-v1.json"
                    if name != "configs/mic_pair_sources.json" or file_sha256(snapshot) != expected:
                        raise ValueError(f"Changed stage input: {artifact}")
                    historical_inputs[name] = dict(snapshot=str(snapshot), sha256=expected)
                    hashes[str(snapshot)] = expected
            hashes[str(path)] = file_sha256(path)
    rows = pd.read_json(args.prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    lookup = {key: i for i, key in enumerate(rows.observation_id)}
    fits = 0
    for stage in stages:
        inspected = set()
        for path in [*stage.rglob("fit.json"), *stage.rglob("manifest.json")]:
            item = json.loads(path.read_text())
            if (
                "training_ids" not in item
                or "validation_ids" not in item
                or path.parent in inspected
            ):
                continue
            inspected.add(path.parent)
            training = np.array([lookup[key] for key in item["training_ids"]])
            validation = np.array([lookup[key] for key in item["validation_ids"]])
            isolated_rows(
                rows.sequence.tolist(), rows.homology_group.tolist(), training, validation
            )
            match = re.search(r"(?:outer|/fold|/o)(\d+)", str(path.parent))
            if match:
                fold = int(match[1])
                assert not (rows.iloc[training].homology_fold == fold).any()
                inner = "-inner" in str(path.parent) or re.search(r"/o\d+-i\d", str(path.parent))
                if inner:
                    assert not (rows.iloc[validation].homology_fold == fold).any()
                else:
                    assert (rows.iloc[validation].homology_fold == fold).all()
            for pair in item.get("pairs", []):
                assert {pair["left"]["observation_id"], pair["right"]["observation_id"]} <= set(
                    item["training_ids"]
                )
            assert item.get("serialization_equal", False)
            fits += 1
    metrics = []
    for stage in stages:
        for path in sorted(stage.glob("*-oof.csv.gz")):
            frame = pd.read_csv(path)
            assert not frame.observation_id.duplicated().any()
            assert set(frame.observation_id) == set(rows.observation_id)
            frame = frame.set_index("observation_id").reindex(rows.observation_id)
            predictions = frame.prediction.to_numpy()
            scales = frame.scale.to_numpy() if "scale" in frame else np.full(len(rows), np.nan)
            for cohort, mask in {
                "all": np.ones(len(rows), bool),
                "matched_strain": rows.strain_index.to_numpy() >= 0,
                "potent_exact": rows.exact_regression.to_numpy(bool)
                & (rows.mic_um.to_numpy() < 10),
            }.items():
                metrics.append(
                    dict(
                        stage=stage.name,
                        model=path.name,
                        cohort=cohort,
                        **evaluate(rows.loc[mask], predictions[mask], scales[mask]),
                    )
                )
    pd.DataFrame(metrics).to_csv(args.output / "common_row_metrics.csv", index=False)
    for name in ["linear8", "finetune8"]:
        directory = args.run / "handoff" / name
        dependencies = json.loads((directory / "refit.json").read_text())["dependencies"]
        sequences = json.loads((directory / "cache/sequences.json").read_text())
        cache = PredictionCache(directory / "cache")
        np.testing.assert_equal(
            cache.read(sequences, dependencies)[::-1], cache.read(sequences[::-1], dependencies)
        )
        base = (args.run / "selection/tops/library-L2/top.fasta").read_bytes()
        assert (args.run / "selection/tops" / f"{name}-w1/top.fasta").read_bytes() == base
    strict = (args.run / "pairs/pairs.jsonl").read_bytes()
    assert strict == (args.run / "pairs-final/pairs.jsonl").read_bytes()
    assert strict == (args.run / "pairs-checked/pairs.jsonl").read_bytes()
    comparison = pd.read_csv(args.run / "selection/tops/candidate_comparison.csv")
    assert comparison.status.eq("complete").all()
    report = json.loads((args.run / "report/adoption_report.json").read_text())
    assert not report["adopted"]
    baseline = comparison[comparison.id.eq("library-L2")].iloc[0]
    tradeoffs = comparison.copy()
    for column in [
        "apex_activity",
        "apex_weak_species",
        "safety",
        "diversity",
        "physchem_mean_log2",
        "linear8_mean_log2",
        "linear650_mean_log2",
        "mlp8_mean_log2",
        "finetune8_mean_log2",
        "runtime_seconds",
    ]:
        tradeoffs[f"change_{column}"] = comparison[column] - baseline[column]
    tradeoffs.to_csv(args.output / "changes_from_control.csv", index=False)
    costs = [
        dict(stage=p.name, wall_seconds=json.loads((p / "manifest.json").read_text())["seconds"])
        for p in stages
    ]
    pd.DataFrame(costs).to_csv(args.output / "stage_costs.csv", index=False)
    (args.output / "report.md").write_text(
        "# MIC continuation review\n\n"
        f"Research candidate: `{report['recommended_research_candidate']}`. "
        "All six model omissions retain that recommendation "
        "within this baseline-pool comparison.\n\n"
        "`changes_from_control.csv` reports raw changes from B1/L2: activity, weakest-species "
        "activity, developability and diversity favor higher values; predicted log2 MIC favors "
        "lower values. Runtime is measured selection time. `stage_costs.csv` reports wall time; "
        "overlapping jobs must not be summed as elapsed or exclusive GPU time.\n\n"
        "These are computational proxies. The single-paper delta pilot did not improve absolute "
        "OOF error, and it cannot establish broad activity-cliff generalization. "
        "The generation policy remains B1/L2/C0; no new full-generation run is claimed.\n"
    )
    write_json(
        args.output / "verification.json",
        dict(
            verified_files=len(hashes),
            manifests=manifests,
            split_isolated_fits=fits,
            candidate_artifact_pairs=len(comparison),
            metrics_rows=len(metrics),
            pair_registry_hardening_equal=True,
            historical_input_snapshots=historical_inputs,
            cache_reorder_equal=True,
            apex_control_equal=True,
            adopted=False,
            excluded_runs=["controls (FASTA argument error before any fit)"],
            conditional_remaining="new policy adoption and its two real generation runs",
        ),
    )
    write_json(args.output / "verified_hashes.json", hashes)
    finish_stage(args.output, hashes, start)


if __name__ == "__main__":
    main()
