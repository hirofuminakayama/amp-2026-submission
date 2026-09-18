"""Freeze distinct completed candidates before recomputing common adoption scenarios."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_competition_pool import FAMILIES
from report_competition_scale import rank_comparisons
from run_mic_research import checked_manifest, write_json

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.adoption import candidate_identity, common_coverage

SCALE = Path("work/competition_exploration/20260913-scale")
MIC = Path("work/mic_prediction/20260913-phase24-a")
BIO = Path("work/competition_bioaccuracy/20260913-b")
BIO_POOLS = {
    "pool-baseline-r2": "baseline",
    "pool-baseline-mlp-r2": "baseline",
    "pool-ddim120k-r2": "ddim120k",
    "pool-ddim120k-mlp": "ddim120k",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = checked_manifest(SCALE / "comparison-v2/manifest.json")
    frozen = json.loads((SCALE / "comparison-v2/comparison_set.json").read_text())
    config = frozen["config"]
    tables = []
    artifacts = dict(frozen["candidates"])
    for source in frozen["inputs_sha256"]:
        manifest = Path(source)
        if file_sha256(manifest) != frozen["inputs_sha256"][source]:
            raise ValueError("Changed frozen comparison input")
        inputs.update(checked_manifest(manifest))
        table = pd.read_csv(manifest.parent / "scenario_scores_by_subset.csv")
        table["id"] = manifest.parent.parent.name + "--" + table.id
        tables.append(table)
    inputs.update(checked_manifest(MIC / "report/manifest.json"))
    table = pd.read_csv(MIC / "report/scenario_scores.csv")
    table = table[table.omitted.eq("none")].drop(columns="omitted")
    for name in table.id.unique():
        artifacts["mic--" + name] = dict(path=str(MIC / "selection/tops" / name))
    table["id"] = "mic--" + table.id
    tables.append(table)
    for directory, pool_name in BIO_POOLS.items():
        root = BIO / directory
        pool = SCALE / "processed" / pool_name
        inputs.update(checked_manifest(root / "manifest.json"))
        table = pd.read_csv(root / "tops/candidate_comparison.csv")
        if not table.status.eq("complete").all():
            raise ValueError("Biological candidate is incomplete")
        sequences = pd.read_csv(pool / "models/pool.csv.gz").sequence.tolist()
        index = {s: i for i, s in enumerate(sequences)}
        for family in FAMILIES:
            path = pool / "models" / f"{family}.npz"
            inputs[str(path)] = file_sha256(path)
            predictions = np.load(path)["species"]
            for i, row in table.iterrows():
                top = pd.read_csv(root / "tops" / row.id / "ranking.csv")
                positions = [index[s] for s in top.sequence]
                table.loc[i, f"{family}_weak_log2"] = float(
                    np.sort(predictions[positions], axis=1)[:, -3:].mean()
                )
        # The stateful supplied HC50 extractor must not be reconstructed by merging batches.
        table["hc50_complete_median"] = np.nan
        table["hc50_coverage"] = 0
        table["hc50_missing_reason"] = "not scored in the registered common oracle batches"
        for name in table.id:
            artifacts[f"bio-{directory}--{name}"] = dict(path=str(root / "tops" / name))
        table["id"] = "bio-" + directory + "--" + table.id
        metrics_path = pool / "metrics/library_metrics.csv"
        inputs[str(metrics_path)] = file_sha256(metrics_path)
        metrics = pd.read_csv(metrics_path).rename(columns={"diversity": "library_diversity"})
        tables.append(
            table.merge(
                metrics[["library", "seed", "subset_size", "fbd", "library_diversity"]],
                on="library",
                validate="many_to_many",
            )
        )
    identities: dict[str, str] = {}
    aliases = []
    for name, record in artifacts.items():
        directory = Path(record["path"])
        inputs.update(checked_manifest(directory / "manifest.json"))
        identity = candidate_identity(directory)
        canonical = identities.setdefault(identity, name)
        aliases.append(dict(id=name, canonical=canonical, identity=identity, **record))
    canonical_ids = set(identities.values())
    frame = pd.concat(tables, ignore_index=True)
    frame = frame[frame.id.isin(canonical_ids)].copy()
    # Freeze membership and exact source hashes before any rank calculations.
    write_json(
        args.output / "comparison_set.json",
        dict(candidates=aliases, config=config, inputs_sha256=inputs),
    )
    frame.to_csv(args.output / "measurements.csv", index=False)
    ranked = rank_comparisons(frame, config)
    ranked.to_csv(args.output / "scenario_ranking.csv")
    common, omitted = common_coverage(frame, ["hc50_complete_median"])
    shared = rank_comparisons(common, config)
    shared.to_csv(args.output / "common_coverage_ranking.csv")
    sensitivity = []
    for family in ["APEX", *FAMILIES]:
        sensitivity.append(
            rank_comparisons(frame, config, family).reset_index().assign(omitted=family)
        )
    pd.concat(sensitivity, ignore_index=True).to_csv(
        args.output / "model_sensitivity.csv", index=False
    )
    write_json(
        args.output / "comparison_summary.json",
        dict(
            registered=len(artifacts),
            distinct=len(canonical_ids),
            recommended=ranked.index[0],
            common_coverage_recommended=shared.index[0],
            common_coverage_omitted=omitted,
            top_ties=ranked[ranked.mean_rank.eq(ranked.mean_rank.min())].index.tolist(),
            adoption_status="pending inference and asset review",
            limitations=[
                "public development and selection reuse; no independent external MIC observations",
                "supplied HC50 batch context preserved; biological additions lack common HC50",
                "library subset seeds are not repeated generator or training seeds",
                "biological paired endpoint evidence retained separately from proxy scenarios",
                "identical library membership and ordered Top counted once",
            ],
        ),
    )
    write_json(
        args.output / "manifest.json",
        dict(artifacts_sha256={p.name: file_sha256(p) for p in args.output.iterdir()}),
    )


if __name__ == "__main__":
    main()
