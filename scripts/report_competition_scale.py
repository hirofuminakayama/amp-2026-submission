"""Freeze the expanded comparison set and rank actual complete candidate artifacts."""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from evaluate_competition_pool import FAMILIES, score_scenarios
from run_competition_models import write_json

from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.features.embeddings import file_sha256


def rank_comparisons(
    frame: pd.DataFrame, config: dict[str, Any], omit: str | None = None
) -> pd.DataFrame:
    keys = ["id", "seed", "subset_size"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate candidate/subset measurements")
    conditions = set(map(tuple, frame[["seed", "subset_size"]].drop_duplicates().to_numpy()))
    for _, group in frame.groupby("id"):
        if set(map(tuple, group[["seed", "subset_size"]].to_numpy())) != conditions:
            raise ValueError("Candidate subset coverage differs")
    scores = pd.concat(
        [
            score_scenarios(group, config, omit)
            for _, group in frame.groupby(["seed", "subset_size"])
        ],
        ignore_index=True,
    )
    columns = [f"scenario_{name}" for name in config["scenarios"]]
    means = scores.groupby("id")[columns].mean()
    for column in columns:
        means[column + "_rank"] = means[column].rank(ascending=False, method="min")
    means["mean_rank"] = means[[c + "_rank" for c in columns]].mean(1)
    return means.sort_values(["mean_rank", "id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_scale.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-pool", type=Path, action="append", default=[])
    parser.add_argument("--report-directory", default="report-combined")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tables, inventory, inputs, artifacts = [], [], {}, {}
    for root in [*[args.root / p for p in config["pools"]], *args.extra_pool]:
        pool = root.name
        report = root / args.report_directory
        manifest = json.loads((report / "manifest.json").read_text())
        verify_hashes({str(report / k): v for k, v in manifest["artifacts_sha256"].items()})
        table = pd.read_csv(report / "scenario_scores_by_subset.csv")
        table["arm_id"] = table.id
        table["id"] = pool + "--" + table.id
        tables.append(table)
        inputs[str(report / "manifest.json")] = file_sha256(report / "manifest.json")
        inventory.append(pd.read_csv(root / "prepare/pool_inventory.csv").assign(pool=pool))
        for _, row in table.drop_duplicates("id").iterrows():
            path = root / row.stage / row.arm_id
            manifest = json.loads((path / "manifest.json").read_text())
            verify_hashes({str(path / k): v for k, v in manifest["artifacts_sha256"].items()})
            artifacts[row.id] = dict(
                path=str(path),
                manifest_sha256=file_sha256(path / "manifest.json"),
                library=row.library,
                ranker=row.ranker,
                constraint=row.constraint,
            )
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.concat(tables, ignore_index=True)
    # Lock actual candidates and input identities before computing any aggregate recommendation.
    write_json(
        args.output / "comparison_set.json",
        dict(candidates=artifacts, inputs_sha256=inputs, config=config),
    )
    ranked = rank_comparisons(frame, config)
    raw = frame.groupby("id").first().drop(columns=[c for c in ranked if c in frame])
    for column in ["fbd", "library_diversity"]:
        raw[column] = frame.groupby("id")[column].mean()
    result = raw.join(ranked).sort_values(["mean_rank", "id"])
    result.to_csv(args.output / "full_candidate_comparison.csv")
    pd.concat(inventory, ignore_index=True).to_csv(args.output / "pool_inventory.csv", index=False)
    sensitive = []
    for name in ["APEX", *FAMILIES]:
        sensitive.append(rank_comparisons(frame, config, name).reset_index().assign(omitted=name))
    pd.concat(sensitive, ignore_index=True).to_csv(
        args.output / "model_sensitivity.csv", index=False
    )
    best = result[result.mean_rank == result.mean_rank.min()]
    write_json(
        args.output / "screen_decision.json",
        dict(
            recommended=best.index[0] if len(best) == 1 else None,
            top_ties=best.index.tolist(),
            tie_action="measure end-to-end runtime for tied methods before adoption"
            if len(best) > 1
            else "no tie",
            candidate_count=len(result),
            evidence="computational public-development proxies",
            adoption_status="pending integration and asset/tier review",
            claim_limitations=[
                "hyperparameter/selection reuse of development data",
                "pretrained training overlap partly unknown",
                "HC50 depends on preserved batch context",
                "alternate representation is diagnostic",
                "no experimental activity or safety improvement demonstrated",
            ],
        ),
    )
    columns = [
        "pool",
        "library",
        "ranker",
        "mean_rank",
        "apex_activity",
        "hc50_complete_median",
        "fbd",
    ]
    page = (
        "<!doctype html><meta charset='utf-8'><title>Expanded pool comparison</title>"
        "<style>body{font:16px system-ui;max-width:1400px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{padding:.5rem;border-bottom:1px solid #ddd}"
        "input{font:inherit;padding:.5rem}</style><h1>Expanded pool comparison</h1>"
        "<p>Computational development proxies; not an experimental activity or safety claim.</p>"
        "<label>Filter <input id='filter'></label>"
    )
    page += result[columns].to_html(escape=True, float_format=lambda x: f"{x:.5g}")
    page += (
        "<h2>Decision</h2><pre>"
        + html.escape((args.output / "screen_decision.json").read_text())
        + "</pre><script>document.getElementById('filter').oninput=e=>"
        "document.querySelectorAll('tbody tr').forEach(r=>r.hidden="
        "!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>"
    )
    (args.output / "tradeoffs.html").write_text(page)
    write_json(
        args.output / "manifest.json",
        dict(
            source_sha256=file_sha256(Path(__file__)),
            artifacts_sha256={p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
