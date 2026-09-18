"""Verify biological research artifacts and write bounded findings and restart evidence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_competition_bioaccuracy import archive_sources, checked_manifest, finish_stage, write_json

from robust_apex_qd.evaluation.developability import evaluate_developability
from robust_apex_qd.evaluation.readiness import fresh_output


def run(root: Path, pools: list[Path], output: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    stage_records = []
    for stage in [
        "inventory",
        "prepare",
        "rescore",
        "features",
        "split",
        "joint",
        "hc50-embeddings",
        "hc50-train",
        "hc50-measured",
        "nested-selection",
        "joint-heads",
        "hc50-oracle",
        "bundle-parity-r2",
        "shared-mic-models",
        "shared-mic-models-s43",
        "shared-mic-models-s44",
        "nested-esm8-exact",
        "nested-esm8-interval",
        "nested-esm8-normal",
        "nested-esm8-exact-s43",
        "nested-esm8-exact-s44",
        "adaptive-nested",
        "bundle-parity-mlp",
    ]:
        path = root / stage / "manifest.json"
        verified = checked_manifest(path)
        inputs.update(verified)
        stage_records.append(
            dict(
                stage=stage,
                files=len(verified) - 1,
                seconds=json.loads(path.read_text())["seconds"],
            )
        )
    fit_manifests = sorted(root.glob("shared-mic-models*/fits/*/*/manifest.json"))
    for path in fit_manifests:
        inputs.update(checked_manifest(path))
    sequences = json.loads((root / "features/sequences.json").read_text())
    reasons = {s: set(evaluate_developability(s).hard_filter_reasons) for s in sequences}
    rules = sorted(set.union(set(), *reasons.values()))
    labels = pd.read_csv(root / "joint/molecular_joint_labels.csv")
    mapping = (
        pd.read_csv(root / "prepare/observation_mapping.csv.gz")
        .drop_duplicates("molecule_id")
        .set_index("molecule_id")
        .sequence
    )
    labels["sequence"] = labels.molecule_id.map(mapping)
    labels = labels[
        (labels.evidence == "measured_mic+measured_hc50")
        & labels.rbc_species.eq("human")
        & labels.ratio.eq(8)
    ]
    filter_records = []
    for removed in ["none", "all", *rules]:
        keep = {s: (removed == "all" or not (r - {removed})) for s, r in reasons.items()}
        frame = labels.assign(retained=labels.sequence.map(keep))
        for species, cohort in frame.groupby("species"):
            active = cohort[cohort.hit_lower.eq(1)]
            inactive = cohort[cohort.hit_upper.eq(0)]
            filter_records.append(
                dict(
                    removed_rule=removed,
                    species=species,
                    available=len(cohort),
                    retained=int(cohort.retained.sum()),
                    joint_positive=len(active),
                    positive_retained=int(active.retained.sum()),
                    positive_retention=float(active.retained.mean()) if len(active) else None,
                    joint_negative=len(inactive),
                    negative_retained=int(inactive.retained.sum()),
                    negative_retention=float(inactive.retained.mean()) if len(inactive) else None,
                )
            )
    pd.DataFrame(filter_records).to_csv(output / "joint_filter_retention.csv", index=False)
    write_json(
        output / "ablation_registry.json",
        dict(
            rules=rules,
            scope="single-rule removal, observed paired human molecules",
            inference="retention alone does not establish selectivity gain; no filter adoption",
        ),
    )
    hc = pd.read_csv(root / "hc50-measured/measured_hc50_oof.csv")
    hc = hc[hc.relation.eq("=")].copy()
    distance = pd.read_csv(root / "split/cross_split_identity.csv")
    hc = hc.merge(distance, on=["sequence", "fold"], validate="many_to_one")
    hc["absolute_error"] = abs(hc.prediction_log2_um - np.log2(hc.value_um))
    hc["identity_bin"] = pd.cut(
        hc.max_train_identity,
        [-0.001, 0.3, 0.45, 0.600001],
        labels=["0-0.3", "0.3-0.45", "0.45-0.6"],
    )
    balanced = (
        hc.groupby(["arm", "sequence", "identity_bin"], observed=True)
        .absolute_error.mean()
        .reset_index()
    )
    balanced.groupby(["arm", "identity_bin"], observed=True).agg(
        molecules=("sequence", "size"), mae=("absolute_error", "mean")
    ).reset_index().to_csv(output / "nearest_train_distance_bins.csv", index=False)
    comparisons = []
    for pool in pools:
        inputs.update(checked_manifest(pool / "manifest.json"))
        table = pd.read_csv(pool / "tops/candidate_comparison.csv")
        table["artifact_root"] = str(pool)
        table["scope"] = "fixed L2 library; official and local validators; no full generation rerun"
        comparisons.append(table)
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(
            output / "final_candidate_comparison.csv", index=False
        )
    comparisons = []
    for path in [root / "nested-selection", *sorted(root.glob("nested-esm8-*"))]:
        protocol = json.loads((path / "protocol.json").read_text())
        table = pd.read_csv(path / "selector_comparison.csv")
        table["mic_family"] = protocol.get("mic_family", "linear8")
        table["training_seed"] = protocol.get("mic_training_seed", 42)
        comparisons.append(table)
    scores = pd.concat(comparisons, ignore_index=True)
    scores.to_csv(output / "mic_family_selection_comparison.csv", index=False)
    supported = scores[scores.metric.eq("top20pct") & scores.species.eq(7)]
    # Cluster-cap selectors with missing folds are not comparable to complete-fold procedures.
    maximum_rows = supported.assessed_rows.max()
    supported = supported[supported.assessed_rows == maximum_rows]
    aggregate = (
        supported.groupby(["mic_family", "selector"])
        .agg(
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            training_seeds=("training_seed", "nunique"),
        )
        .reset_index()
    )
    eligible = aggregate[(aggregate.mic_family == "linear8") | (aggregate.training_seeds == 3)]
    winner = eligible.sort_values(
        ["lower", "mic_family", "selector"], ascending=[False, True, True]
    ).iloc[0]
    aggregate.to_csv(output / "three_seed_selector_comparison.csv", index=False)
    write_json(
        output / "adoption_decision.json",
        dict(
            status="development recommendation; final adoption and release pending",
            primary_selector=winner.selector,
            primary_mic_family=winner.mic_family,
            primary_lower=float(winner.lower),
            primary_upper=float(winner.upper),
            primary="human paired molecular macro top20%; five outer folds",
            candidates=aggregate.to_dict("records"),
            hc50="conditional on MIC family; regression alone does not establish selection gain",
            filters="unchanged pending full selected-set evidence",
            multitask="shared-trunk expansion requires supporting comparison evidence",
            limitations=[
                "public development reuse",
                "unknown chemical stereochemistry",
                "unknown pretrained training overlap",
                "few paired molecules and incomplete P25/P100",
                "exact-only residual calibration can bias censored tails",
                "generated molecules unmeasured",
            ],
            production_config_changed=False,
            remaining=[
                "completed DDPM/union/donor comparison",
                "final pool and ranker adoption",
                "portable bundle is verified; submission entrypoint integration pending",
                "real-model generation smoke",
                "two independent full generation runs and release manifest",
            ],
            minimum="candidate artifact checks only; release readiness not established",
            full="accessibility, source disclosure and full reproduction remain separate checks",
        ),
    )
    write_json(
        output / "verification.json",
        dict(
            stages=stage_records,
            verified_paths=len(inputs),
            native_mic_fit_manifests=len(fit_manifests),
            pools=[str(p) for p in pools],
            all_hashes_valid=True,
            excluded_invalid_runs=["pool-baseline", "pool-ddim120k"],
            excluded_reason="original linear-pool score signs were reversed; use corrected outputs",
        ),
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pools", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.root, *args.pools])
    started = time.monotonic()
    inputs = archive_sources(
        args.output,
        [
            Path(__file__),
            Path("scripts/run_competition_bioaccuracy.py"),
            Path("src/robust_apex_qd/evaluation/developability.py"),
            Path("uv.lock"),
        ],
    )
    inputs.update(run(args.root, args.pools, args.output))
    finish_stage(args.output, inputs, started)


if __name__ == "__main__":
    main()
