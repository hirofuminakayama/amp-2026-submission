"""Aggregate registered exploration tradeoffs; proxy rankings are not an official score."""

import argparse
import html
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from run_competition_selection import requests

from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.features.embeddings import file_sha256


def scenario_scores(frame: pd.DataFrame, scenarios: dict[str, list[float]]) -> pd.DataFrame:
    result = frame.copy()
    families = {
        "activity": {"activity": True},
        "weak_species": {"weak_species": True},
        "safety": {"safety": True, "hc50_complete_median": True},
        "distribution": {"fbd": False},
        "diversity": {"diversity": True, "library_diversity": True},
    }
    for family, components in families.items():
        ranks = [
            frame[column].rank(pct=True, ascending=higher) for column, higher in components.items()
        ]
        result[f"family_{family}"] = pd.concat(ranks, axis=1).mean(axis=1)
    scores = result[[f"family_{f}" for f in families]].to_numpy()
    for name, weights in scenarios.items():
        weight = np.asarray(weights)
        result[f"scenario_{name}"] = np.nansum(scores * weight, axis=1) / np.sum(
            np.isfinite(scores) * weight, axis=1
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--esmc", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    config = json.loads((args.selection / "protocol.json").read_text())
    for stage in ["prepare", "oracles", "stability", "tops"]:
        manifest = json.loads((args.selection / stage / "stage_manifest.json").read_text())
        verify_hashes(manifest["artifacts_sha256"])
    tops = pd.read_csv(args.selection / "tops/top_comparison.csv")
    registry = pd.DataFrame(requests(config, args.selection))
    if set(tops.id) != set(registry.id) or tops.id.duplicated().any():
        raise ValueError("Incomplete arm accounting")
    registry.merge(tops[["id", "status"]], on="id", validate="one_to_one").to_csv(
        args.output / "selection_matrix.csv", index=False
    )
    tops.to_csv(args.output / "top_comparison.csv", index=False)
    libraries = []
    for seed in config["seeds"]:
        small = None
        for size in config["subset_sizes"]:
            directory = args.selection / "stability" / f"s{seed}-n{size}"
            subset = pd.read_csv(directory / "subset_ids.csv")
            if small is not None:
                for variant in subset.variant.unique():
                    a = small[small.variant == variant].sequence.tolist()
                    b = subset[subset.variant == variant].sequence.tolist()
                    if a != b[: len(a)]:
                        raise ValueError("Non-nested comparison subsets")
            small = subset
            table = pd.read_csv(directory / "metrics.csv")
            if set(table.variant) != set(
                p.stem for p in (args.selection / "prepare/libraries").glob("*.fasta")
            ):
                raise ValueError("Incomplete library comparison")
            libraries.append(table)
    library = pd.concat(libraries, ignore_index=True)
    library.to_csv(args.output / "library_comparison.csv", index=False)
    feasible = tops[tops.status == "complete"].copy()
    # Partial coverage medians are displayed but cannot masquerade as whole-Top safety metrics.
    feasible["hc50_complete_median"] = feasible.hc50_median.where(feasible.hc50_coverage == 100)
    scored = []
    for _, group in library.groupby(["seed", "subset_size"]):
        seed, size = int(group.seed.iloc[0]), int(group.subset_size.iloc[0])
        merged = feasible.merge(
            group[["variant", "fbd", "diversity"]].rename(
                columns={"variant": "library", "diversity": "library_diversity"}
            ),
            on="library",
            validate="many_to_one",
        )
        scored.append(
            scenario_scores(merged, config["selection"]["scenarios"]).assign(
                seed=seed, subset_size=size
            )
        )
    scores = pd.concat(scored, ignore_index=True)
    scores.to_csv(args.output / "scenario_scores_by_subset.csv", index=False)
    columns = [f"scenario_{name}" for name in config["selection"]["scenarios"]]
    summary = scores.groupby("id")[columns].mean()
    for column in columns:
        summary[f"{column}_rank"] = summary[column].rank(ascending=False, method="min")
    summary["mean_rank"] = summary[[f"{c}_rank" for c in columns]].mean(axis=1)
    summary = feasible.merge(summary, on="id", validate="one_to_one").sort_values(
        ["mean_rank", "runtime_seconds", "id"]
    )
    # Common-coverage sensitivity uses the same registered oracle cohort in every Top.
    common = []
    for row in feasible.itertuples():
        top = pd.read_csv(args.selection / "tops" / row.id / "ranking.csv")
        covered = top[top.hc50.notna()]
        common.append(
            dict(
                id=row.id,
                n=len(covered),
                activity=covered.species.mean(),
                weak_species=covered.worst3.mean(),
                hc50=covered.hc50.median(),
                developability=covered.dev_pass.mean(),
                scope=(
                    "Top intersect shared ranker-union oracle cohort; membership/count can differ"
                ),
            )
        )
    pd.DataFrame(common).to_csv(args.output / "common_coverage.csv", index=False)
    summary.to_csv(args.output / "scenario_ranking.csv", index=False)
    representation = pd.read_csv(args.representation / "representation_comparison.csv")
    if args.esmc is not None:
        esmc_manifest = json.loads((args.esmc / "manifest.json").read_text())
        verify_hashes({str(args.esmc / k): v for k, v in esmc_manifest["artifacts_sha256"].items()})
        esmc_protocol = json.loads((args.esmc / "protocol.json").read_text())
        reference_protocol = json.loads((args.representation / "protocol.json").read_text())
        if esmc_protocol["samples"] != reference_protocol["samples"]:
            raise ValueError("Alternate representation subsets differ")
        other = pd.read_csv(args.esmc / "representation_comparison.csv")
        representation = representation.merge(
            other, on=["variant", "seed", "subset_size"], validate="one_to_one"
        )
    representation.to_csv(args.output / "representation_comparison.csv", index=False)
    family_best = summary.groupby("family", sort=True).first().reset_index()[["family", "id"]]
    family_best.to_csv(args.output / "family_best.csv", index=False)
    family_scores = scores.groupby("id")[
        [f"family_{name}" for name in config["selection"]["groups"]]
    ].mean()
    nondominated = []
    for identifier, row in family_scores.iterrows():
        dominates = (family_scores.ge(row).all(axis=1) & family_scores.gt(row).any(axis=1)).any()
        if not dominates:
            nondominated.append(identifier)
    decision = dict(
        recommended_screen_candidate=summary.iloc[0].id,
        global_top3=summary.id.head(3).tolist(),
        family_best=family_best.to_dict("records"),
        pareto=nondominated,
        scientific_claim=(
            "computational proxy tradeoff; no measured superiority or official hidden score claim"
        ),
        minimum=(
            "artifact checks passed for feasible arms; read access and "
            "submission documentation still needed"
        ),
        full="public visibility/disclosure and two actual default generation runs still needed",
        tie_break=(
            "selection-stage measured runtime only; end-to-end generation "
            "runtime unchanged for fixed pool"
        ),
        submission_policy="unchanged B1/L2/C0; screen candidates for later scale/integration",
        other_predictors=(
            "new measured predictors not yet available on this pool; later predictor comparison"
        ),
        esmc=(
            "completed: same subsets, ESM-C 300M; diagnostic only"
            if args.esmc is not None
            else "not run; ESM2-650M alternate representation available"
        ),
    )
    (args.output / "screen_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    baseline = feasible[feasible.id == "library-L2"].iloc[0]
    view = summary[
        [
            "id",
            "library",
            "activity",
            "weak_species",
            "hc50_median",
            "hc50_coverage",
            "safety",
            "diversity",
            "changed_top",
            "random25_p05",
            "random25_p50",
            "random25_p95",
            "mean_rank",
        ]
    ].copy()
    view["activity_delta"] = view.activity - baseline.activity
    names = {
        "id": "比較案",
        "library": "library",
        "activity": "菌種等重み活性予測",
        "activity_delta": "対照との差",
        "weak_species": "弱い3菌種",
        "hc50_median": "予測HC50中央値 µM",
        "hc50_coverage": "HC50対象数/100",
        "safety": "合成容易性等filter通過率",
        "diversity": "Topクラスタ数",
        "changed_top": "対照から入替数",
        "random25_p05": "Random25 p05",
        "random25_p50": "Random25 p50",
        "random25_p95": "Random25 p95",
        "mean_rank": "3シナリオ平均順位",
    }
    table = view.rename(columns=names).to_html(
        index=False, float_format=lambda x: f"{x:.4f}", classes="results", border=0
    )
    distribution = (
        library.groupby("variant")[["fbd", "diversity", "conformity"]]
        .mean()
        .join(representation.set_index("variant").filter(regex="^fbd_"))
    )
    template = Path(__file__).parent / "templates/competition_tradeoffs.html"
    payload = template.read_text()
    replacements = {
        "__RECOMMEND__": html.escape(str(summary.iloc[0].id)),
        "__RUN_DATE__": datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d"),
        "__COUNT__": str(len(registry)),
        "__FEASIBLE__": str(len(feasible)),
        "__TOP_TABLE__": table,
        "__LIBRARY_TABLE__": distribution.to_html(float_format=lambda x: f"{x:.4f}", border=0),
    }
    for placeholder, value in replacements.items():
        payload = payload.replace(placeholder, value)

    (args.output / "tradeoffs.html").write_text(payload)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                runtime_seconds=time.monotonic() - began,
                code_sha256=file_sha256(Path(__file__)),
                artifacts_sha256={
                    p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary[["id", "mean_rank", "activity", "changed_top"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
