"""Run registered frozen-pool exploration without changing the adopted submission."""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_research_selection import embedding_lookup
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import aggregate_predictions, load_prediction_archive
from robust_apex_qd.calibration.model import (
    CalibrationArtifact,
    build_calibration_rows,
    evaluate_calibration,
    fit_calibration_artifact,
    load_measurements,
)
from robust_apex_qd.evaluation.developability import evaluate_developability
from robust_apex_qd.evaluation.oracles import run_hemopi2
from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.evaluation.seqme_eval import _evaluate_variant
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.evaluate import _ranker_scores
from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.research.exploration import (
    ExplorationConstraints,
    oracle_union,
    select_exploration_top,
    select_portfolio,
)
from robust_apex_qd.research.selection import bounded_quotas, keyed_subset
from robust_apex_qd.selection.top import maximum_levenshtein_ratio, maximum_local_similarity


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_library(frame: pd.DataFrame, path: Path) -> None:
    write_fasta(
        [FastaRecord(i, s) for i, s in zip(frame.candidate_id, frame.sequence, strict=True)], path
    )


def prepare(config: dict[str, Any], root: Path, output: Path) -> None:
    old = Path(config["old_selection"])
    old_config = json.loads((old / "prepare/protocol.json").read_text())
    verify_hashes(json.loads((old / "prepare/inputs.json").read_text()))
    pool = pd.read_csv(old / "prepare/pool.csv.gz")
    pool = pool[pool.valid].copy().reset_index(drop=True)
    archive = load_prediction_archive(Path(config["frozen_run"]) / "work/apex_predictions.npz")
    index = {s: i for i, s in enumerate(archive.sequences)}
    tensor = archive.mic_u_m[[index[s] for s in pool.sequence]]
    calibration_rows = build_calibration_rows(
        load_measurements(Path("experimental/mic.csv")),
        load_prediction_archive(Path("work/calibration_apex_predictions.npz")),
    )
    evaluation = evaluate_calibration(
        calibration_rows,
        seed=config["calibration_seed"],
        folds=config["calibration_folds"],
        bootstrap_iterations=config["bootstrap_iterations"],
    )
    evaluation.oof.to_csv(output / "calibration_oof.csv", index=False)
    write_json(output / "calibration_metrics.json", evaluation.summary)
    for c in ["C0", "C1", "C2"]:
        artifact = (
            CalibrationArtifact(
                schema_version=1,
                variant="C1",
                feature_order=("median_log2_predicted_mic",),
                coefficients=(0.0,),
                intercept=0.0,
                training_sha256="not_fitted",
                cv_metrics={},
                seed=42,
                folds=5,
            )
            if c == "C0"
            else fit_calibration_artifact(calibration_rows, evaluation, variant=c)
        )
        if c != "C0":
            write_json(output / f"{c}.json", artifact.model_dump())
        scores = _ranker_scores(
            tensor, artifact, pool.physchem_ood.to_numpy(), pool.embedding_ood.to_numpy()
        )
        if c == "C0":
            # The legacy function expects a logistic artifact; explicitly use raw votes for C0.
            from robust_apex_qd.ranking.objectives import (
                broad_objectives,
                conservative_mdr_proxy,
                weighted_quality,
            )

            votes = (tensor <= 16).mean(axis=1)
            mean, tail = broad_objectives(votes)
            mdr = conservative_mdr_proxy(votes)
            disagreement = aggregate_predictions(tensor).model_disagreement_mad_log2
            scores.update(
                B3=percentile_score(mean),
                B4=0.75 * percentile_score(mean) + 0.25 * percentile_score(tail),
                B5=0.6 * percentile_score(mean)
                + 0.2 * percentile_score(tail)
                + 0.2 * percentile_score(mdr)
                - 0.15 * percentile_score(disagreement),
                B6=weighted_quality(
                    broad_mean=mean,
                    broad_tail=tail,
                    mdr_proxy=mdr,
                    disagreement=disagreement,
                    physchem_ood=pool.physchem_ood.to_numpy(),
                    embedding_ood=pool.embedding_ood.to_numpy(),
                ),
            )
        for b in range(3, 7):
            pool[f"{c}-B{b}"] = scores[f"B{b}"]
    votes = (tensor <= 16).mean(axis=1)
    # Equal species weights: A. baumannii, E. coli, K. pneumoniae, P. aeruginosa,
    # S. aureus, E. faecalis, E. faecium; three/two strain panels each receive one vote.
    species = np.column_stack(
        [
            votes[:, 0],
            votes[:, 1:4].mean(1),
            votes[:, 4],
            votes[:, 5:7].mean(1),
            votes[:, 7:9].mean(1),
            votes[:, 9],
            votes[:, 10],
        ]
    )
    for i in range(7):
        pool[f"species_{i}"] = species[:, i]
    pool["species"] = species.mean(1)
    pool["worst3"] = np.sort(species, axis=1)[:, :3].mean(1)
    pool["consensus"] = np.mean(
        [percentile_score(pool[r].to_numpy()) for r in ["B1", "balanced", "tail90", "species"]],
        axis=0,
    )
    pool.to_csv(output / "pool.csv.gz", index=False)
    libraries = output / "libraries"
    libraries.mkdir()
    for name in ["L0", "L1", "L2", "Lref"]:
        (libraries / f"{name}.fasta").write_bytes((old / f"prepare/{name}.fasta").read_bytes())
    l2 = pool[pool.sequence.isin(read_fasta_sequences(libraries / "L2.fasta"))]
    reference = pd.read_csv(old / "prepare/reference_clusters.csv")
    for fraction in [0.25, 0.5, 0.75]:
        selected = []
        for length, count in sorted(l2.length.value_counts().items()):
            available = pool[pool.length == length]
            capacity = available.embedding_cluster.value_counts().to_dict()
            l2w = l2[l2.length == length].embedding_cluster.value_counts(normalize=True).to_dict()
            refw = (
                reference[reference.length == length].cluster.value_counts(normalize=True).to_dict()
            )
            weights = {
                k: (1 - fraction) * l2w.get(k, 0) + fraction * refw.get(k, 0) for k in capacity
            }
            for cluster, quota in bounded_quotas(int(count), weights, capacity).items():
                selected.append(
                    available[available.embedding_cluster == cluster]
                    .sort_values(["physchem_ood", "embedding_ood", "raw_order", "sequence"])
                    .head(quota)
                )
        write_library(
            pd.concat(selected).sort_values(["length", "raw_order", "sequence"]),
            libraries / f"mix{fraction}.fasta",
        )
    quality = pool.assign(
        quality=percentile_score(pool.physchem_ood.to_numpy(), higher_is_better=False)
        + percentile_score(pool.embedding_ood.to_numpy(), higher_is_better=False)
    )
    write_library(
        quality.sort_values(
            ["quality", "raw_order", "sequence"], ascending=[False, True, True]
        ).head(config["library_size"]),
        libraries / "lowOOD.fasta",
    )
    write_json(output / "legacy_config.json", old_config)
    union = oracle_union(
        pool,
        config["rankers"] + ["consensus"] + [f"species_{i}" for i in range(7)],
        config["oracle_prefilter"],
    )
    pool[pool.sequence.isin(union)][["candidate_id", "sequence"]].to_csv(
        output / "oracle_union.csv", index=False
    )
    print(f"Prepared {len(pool)} candidates, 8 libraries, {len(union)} oracle union", flush=True)


def oracles(config: dict[str, Any], root: Path, output: Path) -> None:
    cohort = pd.read_csv(root / "prepare/oracle_union.csv")
    old = Path(config["old_selection"])
    cache = pd.read_csv(old / "oracles/oracle_predictions.csv.gz")
    available = cache.dropna(subset=["hemopi2_hc50_u_m"])
    predictions = dict(zip(available.sequence, available.hemopi2_hc50_u_m, strict=True))
    missing = cohort[~cohort.sequence.isin(predictions) & cohort.sequence.str.len().le(40)]
    for start in range(0, len(missing), config["oracle_batch_size"]):
        batch = missing.iloc[start : start + config["oracle_batch_size"]]
        batch_path = output / f"batch-{start:05}.csv"
        if batch_path.exists():
            saved = pd.read_csv(batch_path)
            if saved.sequence.tolist() != batch.sequence.tolist():
                raise ValueError("Resumed oracle batch input mismatch")
        else:
            result = run_hemopi2(
                Path(config["oracle_dir"]),
                dict(zip(batch.candidate_id, batch.sequence, strict=True)),
            )
            saved = batch.copy()
            saved["hemopi2_hc50_u_m"] = [result[i].hc50_u_m for i in batch.candidate_id]
            saved.to_csv(batch_path, index=False)
        predictions.update(dict(zip(saved.sequence, saved.hemopi2_hc50_u_m, strict=True)))
        print(f"Oracle {start + len(batch)}/{len(missing)} new", flush=True)
    cohort["hemopi2_hc50_u_m"] = cohort.sequence.map(predictions)
    cohort.to_csv(output / "predictions.csv.gz", index=False)
    write_json(
        output / "coverage.json",
        dict(
            union=len(cohort),
            new=len(missing),
            known=int(cohort.hemopi2_hc50_u_m.notna().sum()),
            union_sha256=file_sha256(root / "prepare/oracle_union.csv"),
            maximum_length=40,
        ),
    )


def stability(config: dict[str, Any], root: Path, output: Path) -> None:
    old = Path(config["old_selection"])
    old_config = json.loads((root / "prepare/legacy_config.json").read_text())
    _, references, lookup = embedding_lookup(old_config)
    reference_unique = list(dict.fromkeys(references))
    for seed in config["seeds"]:
        for size in config["subset_sizes"]:
            destination = output / f"s{seed}-n{size}"
            if (destination / "complete.json").exists():
                verify_hashes(json.loads((destination / "complete.json").read_text()))
                continue
            destination.mkdir(exist_ok=False)
            source = old / f"stability-s{seed}-n{size}"
            metrics = pd.read_csv(source / "metrics.csv").to_dict("records")
            subsets = pd.read_csv(source / "subset_ids.csv").to_dict("records")
            ref = keyed_subset(reference_unique, size, seed)
            for path in sorted((root / "prepare/libraries").glob("*.fasta")):
                if path.stem in ["L0", "L1", "L2", "Lref"]:
                    continue
                sequences = read_fasta_sequences(path)
                sample = keyed_subset(sequences, size, seed)
                began = time.monotonic()
                result = _evaluate_variant(sequences, sample, references, ref, lookup, seed=seed)
                metrics.append(
                    dict(
                        seed=seed,
                        subset_size=size,
                        variant=path.stem,
                        **result,
                        runtime_seconds=time.monotonic() - began,
                    )
                )
                subsets.extend(
                    dict(variant=path.stem, position=i, sequence=s) for i, s in enumerate(sample)
                )
                print(
                    f"Stability seed={seed} n={size} {path.stem} FBD={result['fbd']:.4f}",
                    flush=True,
                )
            pd.DataFrame(metrics).to_csv(destination / "metrics.csv", index=False)
            pd.DataFrame(subsets).to_csv(destination / "subset_ids.csv", index=False)
            write_json(
                destination / "complete.json",
                {str(p): file_sha256(p) for p in destination.iterdir() if p.is_file()},
            )


def requests(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    result = [
        dict(
            id=f"library-{p.stem}",
            library=p.stem,
            ranker="B1",
            constraint="current",
            safety="none",
            family="E01",
        )
        for p in sorted((root / "prepare/libraries").glob("*.fasta"))
    ]
    result += [
        dict(
            id=f"rank-{r}",
            library="L2",
            ranker=r,
            constraint="current",
            safety="none",
            family="E02/E03",
        )
        for r in config["rankers"]
        if r != "B1"
    ]
    result += [
        dict(
            id=f"constraint-{c}",
            library="L2",
            ranker="B1",
            constraint=c,
            safety="none",
            family="E04",
        )
        for c in config["constraints"]
        if c != "current"
    ]
    result += [
        dict(
            id=f"safety-{s}",
            library="L2",
            ranker="B1",
            constraint="current",
            safety=s,
            family="E05",
        )
        for s in [
            "union-none",
            "hc50-hard",
            "hc50-soft",
            "dev-hard",
            "dev-soft",
            "both-hard",
            "both-soft",
        ]
    ]
    result += [
        dict(
            id=f"portfolio-{fraction}",
            library="L2",
            ranker="consensus",
            constraint="current",
            safety="none",
            family="E17",
            portfolio=fraction,
        )
        for fraction in config["portfolio_fractions"]
    ]
    return result


def tops(config: dict[str, Any], root: Path, output: Path) -> None:
    pool = pd.read_csv(root / "prepare/pool.csv.gz")
    predictions = pd.read_csv(root / "oracles/predictions.csv.gz").set_index("sequence")
    union = set(pd.read_csv(root / "prepare/oracle_union.csv").sequence)
    expected = oracle_union(
        pool,
        config["rankers"] + ["consensus"] + [f"species_{i}" for i in range(7)],
        config["oracle_prefilter"],
    )
    if union != expected or set(predictions.index) != union:
        raise ValueError("Oracle union differs from registered ranker cohort")
    oracle_old = pd.read_csv(
        Path(config["old_selection"]) / "oracles/oracle_predictions.csv.gz"
    ).set_index("sequence")
    pool["hc50"] = pool.sequence.map(predictions.hemopi2_hc50_u_m)
    pool["dev_pass"] = pool.sequence.map(oracle_old.hard_filter_pass)
    missing_dev = pool.dev_pass.isna()
    pool.loc[missing_dev, "dev_pass"] = pool.loc[missing_dev, "sequence"].map(
        lambda sequence: evaluate_developability(sequence).hard_filter_pass
    )
    pool["dev_pass"] = pool.dev_pass.astype(bool)
    pool[["candidate_id", "sequence", "dev_pass"]].to_csv(
        output / "developability_pool.csv.gz", index=False
    )
    challenge_refs = tuple(read_fasta_sequences(Path("data/antibacterial.fasta")))
    known_refs = tuple(read_fasta_sequences(Path("data/training/training.fasta")))
    cache_path = output / "similarity_cache.json"
    cache = (
        json.loads(cache_path.read_text())
        if cache_path.exists()
        else json.loads((Path(config["old_selection"]) / "tops/similarity_cache.json").read_text())
    )

    def challenge(s: str) -> float:
        if s not in cache["challenge"]:
            cache["challenge"][s] = maximum_levenshtein_ratio(s, challenge_refs)
        return cache["challenge"][s]

    def known(s: str) -> float:
        if s not in cache["known"]:
            cache["known"][s] = maximum_local_similarity(s, known_refs)
        return cache["known"][s]

    spec = importlib.util.spec_from_file_location(
        "official_validator", Path("scripts/verify_submission.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Official validator cannot load")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    baseline = set(read_fasta_sequences(Path(config["frozen_run"]) / "top.fasta"))
    all_results = []
    for request in requests(config, root):
        directory = output / request["id"]
        if (directory / "result.json").exists():
            all_results.append(json.loads((directory / "result.json").read_text()))
            continue
        directory.mkdir(exist_ok=False)
        began = time.monotonic()
        library = root / f"prepare/libraries/{request['library']}.fasta"
        frame = pool[pool.sequence.isin(read_fasta_sequences(library))].copy()
        frame["score"] = percentile_score(frame[request["ranker"]].to_numpy())
        constraints = ExplorationConstraints.model_validate(
            config["constraints"][request["constraint"]]
        )
        if constraints.physchem == "soft":
            frame["score"] -= config["soft_penalty"] * frame.hard_reject.astype(float)
        safety = request["safety"]
        if safety != "none":
            frame.loc[~frame.sequence.isin(union), "score"] = np.nan
        if "hc50" in safety or "both" in safety:
            frame.loc[frame.hc50.isna(), "score"] = np.nan
            if "hard" in safety:
                frame.loc[frame.hc50.lt(100), "score"] = np.nan
            else:
                ranks = frame.hc50.rank(pct=True)
                frame["score"] -= config["soft_penalty"] * (1 - ranks)
        if "dev" in safety or "both" in safety:
            if "hard" in safety:
                frame.loc[~frame.dev_pass, "score"] = np.nan
            else:
                frame["score"] -= config["soft_penalty"] * (~frame.dev_pass).astype(float)
        try:
            if "portfolio" in request:
                top = select_portfolio(
                    frame, request["portfolio"], constraints, config["top_k"], challenge, known
                )
                top["score"] = top.portfolio_score
            else:
                top = select_exploration_top(
                    frame, "score", constraints, config["top_k"], challenge, known
                )
        except ValueError as error:
            if not str(error).startswith("infeasible:"):
                raise
            result = {
                **request,
                "status": "infeasible",
                "reason": str(error),
                "runtime_seconds": time.monotonic() - began,
            }
        else:
            if request["id"] == "library-L2" and top.sequence.tolist() != read_fasta_sequences(
                Path(config["frozen_run"]) / "top.fasta"
            ):
                raise ValueError("Current B1 control differs from the frozen ranked Top")
            top["rank"] = np.arange(1, len(top) + 1)
            top["challenge_similarity"] = top.sequence.map(challenge)
            top.to_csv(directory / "ranking.csv", index=False)
            write_library(top, directory / "top.fasta")
            (directory / "library.fasta").symlink_to(library.resolve())
            full = official._verify_sequences(directory / "library.fasta")
            if len(full) != config["library_size"]:
                raise ValueError("Library count mismatch")
            official._verify_top(directory / "top.fasta", full, config["top_k"])
            official._verify_no_overlap(full, set(challenge_refs))
            official._veritfy_max_simularity(set(top.sequence), set(challenge_refs))
            write_json(
                directory / "official_validation.json",
                dict(
                    valid=True,
                    scope=(
                        "artifact sequence/count/containment/reference checks; "
                        "not generation reproducibility"
                    ),
                    validator_sha256=file_sha256(Path("scripts/verify_submission.py")),
                    library_sha256=file_sha256(library),
                    top_sha256=file_sha256(directory / "top.fasta"),
                ),
            )
            rng = np.random.default_rng(config["random25_seed"])
            draws = np.asarray(
                [rng.choice(100, 25, replace=False) for _ in range(config["random25_draws"])]
            )
            np.save(directory / "random25_indices.npy", draws, allow_pickle=False)
            activity = top.species.to_numpy()[draws].mean(1)
            safety_draw = np.median(top.hc50.to_numpy()[draws], axis=1)
            pd.DataFrame({"activity": activity, "hc50_median": safety_draw}).to_csv(
                directory / "random25.csv.gz", index=False
            )
            current = set(top.sequence)
            result = {
                **request,
                "status": "complete",
                "count": len(top),
                "changed_top": len(current - baseline),
                "activity": float(top.species.mean()),
                "weak_species": float(top.worst3.mean()),
                "safety": float(top.dev_pass.mean()),
                "hc50_median": float(top.hc50.median()) if top.hc50.notna().any() else None,
                "hc50_coverage": int(top.hc50.notna().sum()),
                "diversity": int(top.embedding_cluster.nunique()),
                "random25_p05": float(np.quantile(activity, 0.05)),
                "random25_p50": float(np.quantile(activity, 0.5)),
                "random25_p95": float(np.quantile(activity, 0.95)),
                "random25_complete_hc50": int(np.isfinite(safety_draw).sum()),
                "runtime_seconds": time.monotonic() - began,
                "evidence": "predicted, not measured",
            }
        write_json(directory / "result.json", result)
        write_json(cache_path, cache)
        all_results.append(result)
        print(f"{request['id']}: {result['status']}", flush=True)
    pd.DataFrame(all_results).to_csv(output / "top_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_exploration.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "oracles", "stability", "tops"], required=True
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError("Registered config changed; use a new experiment directory")
    if not protocol.exists():
        if any(args.output.iterdir()):
            raise ValueError(
                "Unregistered nonempty output directory; preserve it and use a new run"
            )
        write_json(protocol, config)
    destination = args.output / args.stage
    if (destination / "stage_manifest.json").exists():
        raise ValueError("Stage already completed")
    if args.stage == "prepare":
        destination.mkdir(exist_ok=False)
    else:
        manifest = json.loads((args.output / "prepare/stage_manifest.json").read_text())
        verify_hashes(manifest["artifacts_sha256"])
        destination.mkdir(exist_ok=True)
    began = time.monotonic()
    source_snapshot = destination / "runner_started.py"
    if source_snapshot.exists() and source_snapshot.read_bytes() != Path(__file__).read_bytes():
        raise ValueError("Runner changed since stage start; preserve this stage and use a new run")
    if not source_snapshot.exists():
        source_snapshot.write_bytes(Path(__file__).read_bytes())
    with threadpool_limits(limits=1):
        {"prepare": prepare, "oracles": oracles, "stability": stability, "tops": tops}[args.stage](
            config, args.output, destination
        )
    write_json(
        destination / "stage_manifest.json",
        dict(
            config_sha256=file_sha256(args.config),
            code_sha256=file_sha256(source_snapshot),
            runtime_seconds=time.monotonic() - began,
            artifacts_sha256={
                str(p): file_sha256(p) for p in destination.rglob("*") if p.is_file()
            },
        ),
    )


if __name__ == "__main__":
    main()
