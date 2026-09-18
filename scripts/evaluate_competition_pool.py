"""Evaluate full-pool candidates with shared subsets and explicit oracle coverage."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evaluate_competition_generators import hemopi_batches
from report_competition_selection import scenario_scores
from run_competition_models import write_json
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.evaluation.seqme_eval import CachedEmbeddingLookup, _evaluate_variant
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.selection import keyed_subset

FAMILIES = ["physchem", "linear8", "linear650", "mlp8", "finetune8"]


def score_scenarios(
    frame: pd.DataFrame, config: dict[str, Any], omit: str | None = None
) -> pd.DataFrame:
    rows = frame.copy()
    components = [rows.apex_activity.rank(pct=True)] if omit != "APEX" else []
    components += [
        rows[f"{name}_mean_log2"].rank(pct=True, ascending=False)
        for name in FAMILIES
        if name != omit
    ]
    rows["activity"] = pd.concat(components, axis=1).mean(axis=1)
    rows["weak_species"] = rows.apex_weak_species
    result = scenario_scores(rows, config["scenarios"])
    result["family_activity"] = rows.activity
    weak = [rows.apex_weak_species.rank(pct=True)] if omit != "APEX" else []
    weak += [
        rows[f"{name}_weak_log2"].rank(pct=True, ascending=False)
        for name in FAMILIES
        if name != omit and f"{name}_weak_log2" in rows
    ]
    result["family_weak_species"] = pd.concat(weak, axis=1).mean(axis=1)
    scores = result[
        [f"family_{n}" for n in ["activity", "weak_species", "safety", "distribution", "diversity"]]
    ].to_numpy()
    for name, weights in config["scenarios"].items():
        weight = np.asarray(weights)
        result[f"scenario_{name}"] = np.nansum(scores * weight, axis=1) / np.sum(
            np.isfinite(scores) * weight, axis=1
        )
    return result


def metrics(config: dict[str, Any], root: Path, output: Path) -> None:
    pool = pd.read_csv(root / "prepare/candidates.csv.gz")
    sequences = pool.sequence.tolist()
    vectors = np.load(root / "features/candidate_embeddings.npy")
    reference = read_fasta_sequences(Path(config["reference"]))
    reference = [
        s for s in reference if 8 <= len(s) <= 50 and not set(s) - set("ACDEFGHIKLMNPQRSTVWY")
    ]
    refs = np.load(root / "features/reference_embeddings.npy")
    mapping = {s: v for s, v in zip(reference, refs, strict=True)}
    mapping.update(zip(sequences, vectors, strict=True))
    lookup = CachedEmbeddingLookup(
        tuple(mapping), np.asarray(list(mapping.values()), dtype=np.float32)
    )
    reference_unique = list(dict.fromkeys(reference))
    records, subsets = [], {}
    for path in sorted((root / "libraries").glob("*.fasta")):
        full = read_fasta_sequences(path)
        for seed in config["seeds"]:
            for size in config["subset_sizes"]:
                sample = keyed_subset(full, size, seed)
                reference_sample = keyed_subset(reference_unique, size, seed)
                result = _evaluate_variant(
                    full, sample, reference_unique, reference_sample, lookup, seed=seed
                )
                records.append(dict(library=path.stem, seed=seed, subset_size=size, **result))
                subsets[f"{path.stem}/{seed}/{size}"] = sample
                subsets[f"reference/{seed}/{size}"] = reference_sample
                print(f"{path.stem} seed{seed} n{size}: FBD={result['fbd']:.4f}", flush=True)
    pd.DataFrame(records).to_csv(output / "library_metrics.csv", index=False)
    write_json(output / "subsets.json", subsets)
    # Same recorded reference and seed as the earlier alternate-representation diagnostic.
    alt = Path("work/competition_exploration/20260912-a/representation")
    protocol = json.loads((alt / "protocol.json").read_text())
    alt_sequences = pd.read_csv(alt / "sequences.csv").sequence.tolist()
    alt_vectors = np.load(alt / "embeddings.npy")
    alt_index = {s: i for i, s in enumerate(alt_sequences)}
    reference_sample = protocol["samples"]["reference"]
    b = alt_vectors[[alt_index[s] for s in reference_sample]].astype(float)
    cb = np.cov(b, rowvar=False)
    if root.name == "baseline":
        old = Path(config["refits"])
        old_sequences = pd.read_csv(old / "pool_sequences.csv").sequence.tolist()
        old_index = {s: i for i, s in enumerate(old_sequences)}
        candidate_vectors = np.load(old / "esm650.npy")[[old_index[s] for s in sequences]]
        alternate_path = old / "esm650.npy"
    else:
        alternate_path = root / "models/esm650.npy"
        candidate_vectors = np.load(alternate_path)
    index = {s: i for i, s in enumerate(sequences)}
    alternate = []
    for path in sorted((root / "libraries").glob("*.fasta")):
        sample = keyed_subset(read_fasta_sequences(path), 1000, 42)
        a = candidate_vectors[[index[s] for s in sample]].astype(float)
        ca = np.cov(a, rowvar=False)
        fbd = max(
            0.0,
            float(
                np.sum((a.mean(0) - b.mean(0)) ** 2) + np.trace(ca + cb - 2 * sqrtm(ca @ cb).real)
            ),
        )
        alternate.append(dict(library=path.stem, seed=42, subset_size=1000, esm650_fbd=fbd))
        subsets[f"esm650/{path.stem}/42/1000"] = sample
    pd.DataFrame(alternate).to_csv(output / "alternate_representation.csv", index=False)
    write_json(
        output / "alternate_inputs.json",
        {
            str(p): file_sha256(p)
            for p in [
                alternate_path,
                alt / "protocol.json",
                alt / "sequences.csv",
                alt / "embeddings.npy",
            ]
        },
    )
    write_json(output / "subsets.json", subsets)


def safety(root: Path, output: Path, stages: list[str]) -> None:
    sequences = sorted(
        set(
            s
            for stage in stages
            for p in (root / stage).glob("*/top.fasta")
            for s in read_fasta_sequences(p)
        )
    )
    values = {}
    cache_inputs = {}
    for path, column in [
        (
            Path("work/competition_exploration/20260912-a/phase2/oracles/predictions.csv.gz"),
            "hemopi2_hc50_u_m",
        ),
        (
            Path(
                "work/competition_exploration/20260912-b/phase4-evaluation/safety/predictions.csv.gz"
            ),
            "hc50",
        ),
    ]:
        rows = pd.read_csv(path).dropna(subset=[column])
        values.update(zip(rows.sequence, rows[column], strict=True))
        cache_inputs[str(path)] = file_sha256(path)
    missing = [s for s in sequences if s not in values and 7 <= len(s) <= 40]
    write_json(
        output / "protocol.json",
        dict(
            ordered_missing=missing,
            cache_inputs_sha256=cache_inputs,
            batch_size=1000,
            oracle_sha256=file_sha256(Path("work/oracles/hemopi2/manifest.json")),
            extractor_sha256=file_sha256(Path("src/robust_apex_qd/evaluation/oracles.py")),
            scope="union of actual Tops; oracle is not used to prefilter rankers",
            limitation="frozen ordered batches; supplied RRI extractor retains row state",
        ),
    )
    values.update(hemopi_batches(missing, output))
    pd.DataFrame(dict(sequence=sequences, hc50=[values.get(s) for s in sequences])).to_csv(
        output / "predictions.csv", index=False
    )


def report(
    config: dict[str, Any], root: Path, output: Path, stages: list[str], safety_dir: Path
) -> None:
    candidates = pd.concat(
        [
            pd.read_csv(root / stage / "candidate_comparison.csv").assign(stage=stage)
            for stage in stages
        ],
        ignore_index=True,
    )
    candidates.to_csv(output / "candidate_inventory.csv", index=False)
    feasible = candidates[candidates.status == "complete"].copy()
    values = pd.read_csv(safety_dir / "predictions.csv").set_index("sequence").hc50
    pool_sequences = pd.read_csv(root / "prepare/candidates.csv.gz").sequence.tolist()
    pool_index = {s: i for i, s in enumerate(pool_sequences)}
    predictions = {name: np.load(root / "models" / f"{name}.npz")["species"] for name in FAMILIES}
    for i, row in feasible.iterrows():
        top = pd.read_csv(root / row.stage / row.id / "ranking.csv")
        positions = [pool_index[s] for s in top.sequence]
        for name, prediction in predictions.items():
            feasible.loc[i, f"{name}_weak_log2"] = float(
                np.sort(prediction[positions], axis=1)[:, -3:].mean()
            )
        hc50 = values.reindex(top.sequence).to_numpy()
        feasible.loc[i, "hc50_coverage"] = int(np.isfinite(hc50).sum())
        feasible.loc[i, "hc50_complete_median"] = (
            float(np.median(hc50)) if np.isfinite(hc50).all() else np.nan
        )
        indices = np.asarray(
            [
                np.random.default_rng(seed).choice(len(top), 25, replace=False)
                for seed in range(42, 1042)
            ]
        )
        draws = pd.DataFrame(
            dict(
                apex_activity=top.species.to_numpy()[indices].mean(1),
                hc50_median=np.median(hc50[indices], axis=1),
            )
        )
        draws.to_csv(output / f"{row.id}-random25.csv.gz", index=False)
        feasible.loc[i, "random25_activity_p05"] = draws.apex_activity.quantile(0.05)
        feasible.loc[i, "random25_activity_p95"] = draws.apex_activity.quantile(0.95)
    library = pd.read_csv(root / "metrics/library_metrics.csv")
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
        scored.append(score_scenarios(merged, config).assign(seed=seed, subset_size=size))
    by_subset = pd.concat(scored, ignore_index=True)
    by_subset.to_csv(output / "scenario_scores_by_subset.csv", index=False)
    columns = [f"scenario_{n}" for n in config["scenarios"]]
    summary = by_subset.groupby("id")[columns].mean()
    for column in columns:
        summary[column + "_rank"] = summary[column].rank(ascending=False, method="min")
    summary["mean_rank"] = summary[[c + "_rank" for c in columns]].mean(1)
    ranked = feasible.merge(summary, on="id", validate="one_to_one").sort_values(
        ["mean_rank", "runtime_seconds", "id"]
    )
    ranked.to_csv(output / "scenario_ranking.csv", index=False)
    top_libraries = ranked[ranked.factor == "library"].library.head(2).tolist()
    top_rankers = (
        ranked[(ranked.factor == "ranker") | ranked.id.eq("library-L2")].ranker.head(3).tolist()
    )
    specs = [
        dict(
            id=f"compound-{lib}-{ranker}",
            library=lib,
            ranker=ranker,
            constraint="challenge80",
            factor="compound",
        )
        for lib in top_libraries
        for ranker in top_rankers
    ]
    write_json(output / "compound_specs.json", specs)
    write_json(
        output / "interpretation.json",
        dict(
            evidence="public-development computational proxies, not independent measured activity",
            subset_protocol="same keyed seeds and sizes for all libraries",
            hc50="exact Top coverage; original cache/batch context retained",
            selection="mean of three registered scenario ranks",
            compound=(
                "two best single-factor libraries by three best single-factor rankers, "
                "with the screened challenge80 constraint"
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_scale.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--stage", choices=["metrics", "safety", "report"], required=True)
    parser.add_argument("--include-compound", action="store_true")
    parser.add_argument("--top-stage", default="tops")
    parser.add_argument("--compound-stage", default="compound")
    parser.add_argument("--safety-stage")
    parser.add_argument("--output-stage")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.root / args.pool
    suffix = "-combined" if args.include_compound else ""
    stages = [args.top_stage, args.compound_stage] if args.include_compound else [args.top_stage]
    safety_stage = args.safety_stage or "safety" + suffix
    dependencies = ["prepare", "features", "models", "libraries"]
    if args.stage != "metrics":
        dependencies += stages
    if args.stage == "report":
        dependencies += ["metrics", safety_stage]
    for stage in dependencies:
        directory = root / stage
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["config"] != config:
            raise ValueError("Evaluation input configuration changed")
        verify_hashes(
            {str(directory / name): digest for name, digest in manifest["artifacts_sha256"].items()}
        )
    output = root / (args.output_stage or args.stage + suffix)
    output.mkdir(exist_ok=False)
    source_hash = file_sha256(Path(__file__))
    (output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    with threadpool_limits(limits=4):
        if args.stage == "metrics":
            metrics(config, root, output)
        elif args.stage == "safety":
            safety(root, output, stages)
        else:
            report(config, root, output, stages, root / safety_stage)
    write_json(
        output / "manifest.json",
        dict(
            config=config,
            selection_stages=stages,
            safety_stage=safety_stage,
            source_sha256=source_hash,
            seconds=time.monotonic() - start,
            artifacts_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
