"""Apply existing competition scenarios and model-omission sensitivity to MIC candidates."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_competition_pool import FAMILIES, safety, score_scenarios
from run_mic_research import checked_manifest, finish_stage, write_json
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_lineage import capture_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("work/competition_exploration/20260913-scale/processed/baseline"),
    )
    parser.add_argument("--safety", type=Path)
    parser.add_argument("--stage", choices=["safety", "report"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    inputs = checked_manifest(args.selection / "manifest.json")
    fresh_output(args.output, [args.selection, args.pool])
    capture_execution(args.output)
    if args.stage == "safety":
        safety(args.selection, args.output, ["tops"])
        finish_stage(args.output, inputs, start)
        return
    if args.safety is None:
        parser.error("report requires --safety")
    inputs.update(checked_manifest(args.safety / "manifest.json"))
    for stage in ["models", "metrics"]:
        inputs.update(checked_manifest(args.pool / stage / "manifest.json"))
    config_path = Path("configs/competition_scale.json")
    config = json.loads(config_path.read_text())
    inputs[str(config_path)] = file_sha256(config_path)
    candidates = pd.read_csv(args.selection / "tops/candidate_comparison.csv")
    feasible = candidates[candidates.status.eq("complete")].copy()
    pool = pd.read_csv(args.pool / "models/pool.csv.gz")
    index = {s: i for i, s in enumerate(pool.sequence)}
    predictions = {
        name: np.load(args.pool / "models" / f"{name}.npz")["species"] for name in FAMILIES
    }
    values = pd.read_csv(args.safety / "predictions.csv").set_index("sequence").hc50
    control = set(pd.read_csv(args.selection / "tops/library-L2/ranking.csv").sequence)
    for i, row in feasible.iterrows():
        top = pd.read_csv(args.selection / "tops" / row.id / "ranking.csv")
        positions = [index[s] for s in top.sequence]
        for family, prediction in predictions.items():
            feasible.loc[i, f"{family}_weak_log2"] = float(
                np.sort(prediction[positions], axis=1)[:, -3:].mean()
            )
        hc50 = values.reindex(top.sequence).to_numpy()
        feasible.loc[i, "hc50_coverage"] = int(np.isfinite(hc50).sum())
        feasible.loc[i, "hc50_complete_median"] = (
            float(np.median(hc50)) if np.isfinite(hc50).all() else np.nan
        )
        feasible.loc[i, "control_overlap"] = len(control & set(top.sequence))
    library = pd.read_csv(args.pool / "metrics/library_metrics.csv")
    scored = []
    for _, group in library.groupby(["seed", "subset_size"]):
        seed, size = int(group.seed.iloc[0]), int(group.subset_size.iloc[0])
        merged = feasible.merge(
            group[["library", "fbd", "diversity"]].rename(
                columns={"diversity": "library_diversity"}
            ),
            on="library",
            validate="many_to_one",
        )
        for omit in [None, "APEX", *FAMILIES]:
            scored.append(
                score_scenarios(merged, config, omit).assign(
                    seed=seed, subset_size=size, omitted=omit or "none"
                )
            )
    combined = pd.concat(scored, ignore_index=True)
    combined.to_csv(args.output / "scenario_scores.csv", index=False)
    columns = [f"scenario_{name}" for name in config["scenarios"]]
    rankings = []
    for omitted, group in combined.groupby("omitted"):
        summary = group.groupby("id")[columns].mean()
        summary["mean_rank"] = summary.rank(ascending=False, method="min").mean(1)
        table = feasible.merge(summary, on="id", validate="one_to_one").sort_values(
            ["mean_rank", "runtime_seconds", "id"]
        )
        rankings.append(table.assign(omitted=omitted))
    ranked = pd.concat(rankings, ignore_index=True)
    ranked.to_csv(args.output / "scenario_ranking.csv", index=False)
    winners = {name: group.iloc[0].id for name, group in ranked.groupby("omitted", sort=True)}
    write_json(
        args.output / "adoption_report.json",
        dict(
            recommended_research_candidate=winners["none"],
            omission_winners=winners,
            adopted=False,
            current_policy="B1/L2/C0",
            reason=(
                "handoff to competition selection and joint MIC/HC50 workflow; "
                "proxy comparison does not implement a new full-generation policy"
            ),
            evidence="saved baseline pool and existing scenario weights; no new significance gate",
            coverage=feasible[["id", "hc50_coverage", "control_overlap"]].to_dict("records"),
            full_reproduction="conditional on later adoption; no new policy adopted here",
        ),
    )
    (args.output / "report.md").write_text(
        "# MIC candidate handoff\n\n"
        + f"Scenario recommendation: `{winners['none']}`.\n\n"
        + "See scenario_ranking.csv for all candidates and model omissions. "
        + "Activity and HC50 are computational proxies; missing HC50 is not safety evidence. "
        + "The existing L2 library is shared across candidates. "
        + "The handoff preserves B1/L2/C0 pending the downstream adoption decision.\n"
    )
    finish_stage(args.output, inputs, start)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
