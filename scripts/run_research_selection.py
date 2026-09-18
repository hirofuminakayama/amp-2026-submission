"""Compare frozen libraries, rankers and safety proxies without replacing submission artifacts."""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import load_prediction_archive
from robust_apex_qd.calibration.model import load_measurements
from robust_apex_qd.evaluation.oracles import run_hemopi2
from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.evaluation.seqme_eval import (
    _evaluate_variant,
    decide_library_adoption,
    validated_embedding_lookup,
)
from robust_apex_qd.features.embeddings import _array_sha256, file_sha256
from robust_apex_qd.generation.sampler import LengthPolicy, build_length_quotas
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.research.selection import (
    align_apex_scores,
    bounded_quotas,
    complete_panel_activity,
    keyed_subset,
    library_rows,
    selection_scores,
)
from robust_apex_qd.selection.library import LibraryCandidate, select_library
from robust_apex_qd.selection.top import (
    TopCandidate,
    TopSelectionConfig,
    maximum_levenshtein_ratio,
    maximum_local_similarity,
    select_top_with_fallback,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def embedding_lookup(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    work = Path(config["frozen_run"]) / "work"
    return validated_embedding_lookup(
        candidates_path=work / "candidates.csv.gz",
        candidate_embeddings_path=work / "candidate_embeddings.npy",
        reference_fasta_path=Path(config["training_reference"]),
        reference_embeddings_path=work / "reference_embeddings.npy",
        embedding_manifest_path=work / "embedding_manifest.json",
    )


def prepare(config: dict[str, Any], output: Path, config_path: Path) -> None:
    run = Path(config["frozen_run"])
    freeze = json.loads(Path("reports/final_freeze.json").read_text())
    if run.resolve() != Path(freeze["representative_full_run"]["run_path"]).resolve():
        raise ValueError("Only the frozen representative pool is supported")
    sources = [p for p in run.rglob("*") if p.is_file()]
    sources += [
        Path(config[k]) for k in ["training_reference", "challenge_reference", "oracle_cache"]
    ]
    sources += [
        Path("configs/final.yaml"),
        Path("experimental/mic.csv"),
        config_path,
        Path(config["oracle_dir"]) / "manifest.json",
    ]
    write_json(output / "inputs.json", {str(p): file_sha256(p) for p in sources})
    write_json(output / "protocol.json", config)
    work = run / "work"
    sequences, references, _ = embedding_lookup(config)
    rows = pd.read_csv(work / "candidates.csv.gz")
    for name in ["candidate_physchem.csv.gz", "candidate_embedding_diagnostics.csv.gz"]:
        frame = pd.read_csv(work / name)
        if "sequence" in frame:
            if (
                frame.set_index("candidate_id").sequence.reindex(rows.candidate_id).tolist()
                != sequences
            ):
                raise ValueError("Feature sequence identity mismatch")
            frame = frame.drop(columns="sequence")
        frame = frame.drop(columns=[c for c in frame if c in rows and c != "candidate_id"])
        rows = rows.merge(frame, on="candidate_id", validate="one_to_one", how="left")
    archive = load_prediction_archive(work / "apex_predictions.npz")
    for name, scores in align_apex_scores(
        list(archive.sequences), archive.mic_u_m, sequences, rows.valid.tolist()
    ).items():
        rows[name] = scores
    candidates = [
        LibraryCandidate.model_validate({key: row[key] for key in LibraryCandidate.model_fields})
        for row in rows.to_dict("records")
    ]
    quotas = build_length_quotas(
        config["library_size"],
        LengthPolicy.EMPIRICAL_TEMPERED,
        10,
        40,
        Path(config["training_reference"]),
        0.75,
    )
    variants = {}
    for variant in ["L0", "L1", "L2"]:
        selection = select_library(
            candidates, variant=variant, size=config["library_size"], target_quotas=quotas
        )
        variants[variant] = [c.sequence for c in selection.selected]
        write_fasta(
            [FastaRecord(c.candidate_id, c.sequence) for c in selection.selected],
            output / f"{variant}.fasta",
        )
    if file_sha256(output / "L2.fasta") != file_sha256(run / "library.fasta"):
        raise ValueError("Reselected L2 is not byte-identical to the freeze")
    # Reconstruct the frozen projection/clustering and prove labels before projecting references.
    em = json.loads((work / "embedding_manifest.json").read_text())
    embeddings = np.load(work / "candidate_embeddings.npy")
    reference_embeddings = np.load(work / "reference_embeddings.npy")
    rng = np.random.default_rng(em["pca_seed"])
    indices = np.sort(rng.choice(len(references), em["pca_reference_subset"], replace=False))
    pca = PCA(n_components=em["pca_components"], svd_solver="full").fit(
        reference_embeddings[indices]
    )
    if _array_sha256(pca.components_, pca.mean_) != em["pca_transform_sha256"]:
        raise ValueError("Reconstructed PCA hash differs from the frozen transform")
    projected = pca.transform(embeddings).astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=em["cluster_count"],
        random_state=em["clustering_seed"],
        n_init=10,
        batch_size=2048,
        reassignment_ratio=0.0,
    )
    labels = km.fit_predict(projected)
    if not np.array_equal(labels, rows.embedding_cluster.to_numpy()):
        raise ValueError("Reconstructed clusters differ from the frozen labels")
    reference_labels = km.predict(pca.transform(reference_embeddings).astype(np.float32))
    reference_rows = pd.DataFrame(
        {
            "sequence": references,
            "cluster": reference_labels,
            "length": [len(s) for s in references],
        }
    )
    reference_rows.to_csv(output / "reference_clusters.csv", index=False)
    all_clusters = range(em["cluster_count"])
    ref_frequency = reference_rows.cluster.value_counts(normalize=True).reindex(
        all_clusters, fill_value=0
    )
    l2 = library_rows(rows, variants["L2"])
    l2_frequency = l2.embedding_cluster.value_counts(normalize=True).reindex(
        all_clusters, fill_value=0
    )
    tv = float((ref_frequency - l2_frequency).abs().sum() / 2)
    if tv > config["reference_cluster_trigger_tv"]:
        selected_frames = []
        for length, quota in sorted(l2.length.value_counts().to_dict().items()):
            available = rows[rows.valid & (rows.length == length)]
            capacity = available.embedding_cluster.value_counts().to_dict()
            weights = (
                reference_rows[reference_rows.length == length].cluster.value_counts().to_dict()
            )
            allocation = bounded_quotas(quota, weights, capacity)
            for cluster, count in allocation.items():
                selected_frames.append(
                    available[available.embedding_cluster == cluster]
                    .sort_values(["physchem_ood", "embedding_ood", "raw_order", "sequence"])
                    .head(count)
                )
        matched = pd.concat(selected_frames).sort_values(["length", "raw_order", "sequence"])
        variants["Lref"] = matched.sequence.tolist()
        write_fasta(
            [FastaRecord(r.candidate_id, r.sequence) for r in matched.itertuples()],
            output / "Lref.fasta",
        )
    challenge = set(read_fasta_sequences(Path(config["challenge_reference"])))
    summaries = []
    composition = []
    for name, selected in variants.items():
        if (
            len(selected) != config["library_size"]
            or len(set(selected)) != len(selected)
            or set(selected) & challenge
        ):
            raise ValueError("Full library count/uniqueness/novelty contract failed")
        subset = library_rows(rows, selected)
        frequency = subset.embedding_cluster.value_counts(normalize=True).reindex(
            all_clusters, fill_value=0
        )
        summaries.append(
            {
                "variant": name,
                "count": len(selected),
                "cluster_coverage": int((frequency > 0).sum()),
                "reference_cluster_tv": float((frequency - ref_frequency).abs().sum() / 2),
            }
        )
        for dimension in ["length", "embedding_cluster"]:
            for value, count in subset[dimension].value_counts().items():
                composition.append(
                    {"variant": name, "dimension": dimension, "value": value, "count": count}
                )
    pd.DataFrame(summaries).to_csv(output / "library_composition_summary.csv", index=False)
    pd.DataFrame(composition).to_csv(output / "library_composition.csv", index=False)
    rows.to_csv(output / "pool.csv.gz", index=False)
    write_json(
        output / "selection_manifest.json",
        {
            "variants": list(variants),
            "l2_reference_cluster_tv": tv,
            "reference_variant_triggered": "Lref" in variants,
            "pca_and_cluster_reconstruction": "exact_match",
        },
    )
    print(f"Prepared {list(variants)}; L2/reference cluster TV={tv:.4f}", flush=True)


def oracles(config: dict[str, Any], root: Path, output: Path) -> None:
    rows = pd.read_csv(root / "prepare/pool.csv.gz")
    l2 = set(read_fasta_sequences(root / "prepare/L2.fasta"))
    prefilter = (
        library_rows(rows, sorted(l2))
        .sort_values(["B1", "raw_order", "sequence"], ascending=[False, True, True])
        .head(config["oracle_prefilter"])
    )
    cached = pd.read_csv(config["oracle_cache"])
    if (
        cached.candidate_id.duplicated().any()
        or cached.set_index("candidate_id").sequence.reindex(prefilter.candidate_id).tolist()
        != prefilter.sequence.tolist()
    ):
        raise ValueError("Oracle cache does not match L2 sequence IDs")
    usable = cached[cached.hemopi2_hc50_u_m.notna()]
    predictions = dict(zip(usable.sequence, usable.hemopi2_hc50_u_m, strict=True))
    missing = prefilter[
        (~prefilter.sequence.isin(predictions))
        & (prefilter.length <= config["maximum_oracle_length"])
    ]
    for start in range(0, len(missing), config["oracle_batch_size"]):
        batch = missing.iloc[start : start + config["oracle_batch_size"]]
        result = run_hemopi2(
            Path(config["oracle_dir"]), dict(zip(batch.candidate_id, batch.sequence, strict=True))
        )
        batch_rows = [
            {
                "candidate_id": r.candidate_id,
                "sequence": r.sequence,
                "hemopi2_hc50_u_m": result[r.candidate_id].hc50_u_m,
            }
            for r in batch.itertuples()
        ]
        pd.DataFrame(batch_rows).to_csv(output / f"batch-{start:05}.csv", index=False)
        predictions.update({r["sequence"]: r["hemopi2_hc50_u_m"] for r in batch_rows})
        print(f"HemoPI2 completed {start + len(batch)}/{len(missing)} new sequences", flush=True)
    cached["hemopi2_hc50_u_m"] = cached.sequence.map(predictions)
    cached.to_csv(output / "oracle_predictions.csv.gz", index=False)
    prefilter[["candidate_id", "sequence"]].to_csv(output / "prefilter.csv", index=False)
    write_json(
        output / "coverage.json",
        {
            "prefilter": len(prefilter),
            "new_predictions": len(missing),
            "known_predictions": int(cached.hemopi2_hc50_u_m.notna().sum()),
            "scope": "fixed B1 first 10000; outside cohort remains missing; HC50 is predicted",
        },
    )


def tops(config: dict[str, Any], root: Path, output: Path, oracle_path: Path | None = None) -> None:
    pool = pd.read_csv(root / "prepare/pool.csv.gz")
    pilot = oracle_path is not None
    oracle_path = oracle_path or root / "oracles/oracle_predictions.csv.gz"
    expected_prefilter = (
        library_rows(pool, read_fasta_sequences(root / "prepare/L2.fasta"))
        .sort_values(["B1", "raw_order", "sequence"], ascending=[False, True, True])
        .head(config["oracle_prefilter"])[["candidate_id", "sequence"]]
        .reset_index(drop=True)
    )
    prefilter_path = root / "oracles/prefilter.csv"
    prefilter = expected_prefilter if pilot else pd.read_csv(prefilter_path)
    if not prefilter.equals(expected_prefilter):
        raise ValueError("Saved prefilter differs from fixed B1 candidates")
    prefilter_sequences = set(prefilter.sequence)
    oracle = pd.read_csv(oracle_path).set_index("sequence")
    challenge = tuple(read_fasta_sequences(Path(config["challenge_reference"])))
    known = tuple(read_fasta_sequences(Path(config["training_reference"])))
    cache_path = root / "top_pilot/similarity_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    challenge_cache, known_cache = cache.get("challenge", {}), cache.get("known", {})
    write_json(
        output / "oracle_input.json",
        {
            "prefilter_sha256": None if pilot else file_sha256(prefilter_path),
            "path": str(oracle_path),
            "sha256": file_sha256(oracle_path),
            "available_hc50": int(oracle.hemopi2_hc50_u_m.notna().sum()),
        },
    )

    def challenge_similarity(sequence: str, references: Any) -> float:
        if sequence not in challenge_cache:
            challenge_cache[sequence] = maximum_levenshtein_ratio(sequence, references)
        return challenge_cache[sequence]

    def known_similarity(sequence: str, references: Any) -> float:
        if sequence not in known_cache:
            known_cache[sequence] = maximum_local_similarity(sequence, references)
        return known_cache[sequence]

    variants = json.loads((root / "prepare/selection_manifest.json").read_text())["variants"]
    requests = [(v, "B1", "B1") for v in variants]
    requests += [("L2", r, "B1") for r in config["rankers"] if r != "B1"]
    requests += [
        ("L2", "B1", condition) for condition in config["filter_conditions"] if condition != "B1"
    ]
    results, random_summaries, draws = [], [], []
    baseline_top = set(read_fasta_sequences(Path(config["frozen_run"]) / "top.fasta"))
    for variant, ranker, condition in requests:
        label = f"{variant}-{ranker}-{condition}"
        sequences = read_fasta_sequences(root / "prepare" / f"{variant}.fasta")
        frame = library_rows(pool, sequences)
        frame["hc50"] = frame.sequence.map(oracle.hemopi2_hc50_u_m)
        frame["dev_pass"] = frame.sequence.map(oracle.hard_filter_pass)
        rejected = np.zeros(len(frame), dtype=bool)
        if condition != "B1":
            rejected |= ~frame.sequence.isin(prefilter_sequences).to_numpy()
        if condition in ["hemopi2", "both"]:
            rejected |= (
                ~frame.hc50.ge(config["hc50_boundary_um"]) | (frame.length > 40)
            ).to_numpy()
        if condition in ["developability", "both"]:
            rejected |= ~frame.dev_pass.fillna(False).to_numpy(bool)
        support = {
            "library_rows": len(frame),
            "oracle_overlength_rows": int((frame.length > config["maximum_oracle_length"]).sum()),
            "oracle_missing_rows": int(frame.hc50.isna().sum()),
            "external_filter_rejected_rows": int(rejected.sum()),
            "oracle_scope": "fixed B1 first 10000; not whole-library HC50 coverage",
        }
        candidates = [
            TopCandidate(
                candidate_id=r["candidate_id"],
                sequence=r["sequence"],
                raw_order=r["raw_order"],
                final_score=float(r[ranker]),
                embedding_cluster=r["embedding_cluster"],
                physchem_hard_reject=bool(r["hard_reject"]),
                external_hard_reject=bool(rejected[i]),
                median_log2_mic=-r["B1"],
            )
            for i, r in enumerate(frame.to_dict("records"))
        ]
        try:
            selected = select_top_with_fallback(
                candidates,
                challenge_references=challenge,
                known_references=known,
                top_k=config["top_k"],
                config=TopSelectionConfig(),
                challenge_similarity=challenge_similarity,
                known_similarity=known_similarity,
            )
        except RuntimeError as error:
            results.append(
                {
                    "label": label,
                    "library": variant,
                    "ranker": ranker,
                    "condition": condition,
                    "status": "cannot_collect_top100",
                    **support,
                    "reason": str(error),
                    "evidence": "proxy",
                }
            )
            print(f"{label}: cannot collect Top-100", flush=True)
            continue
        selected_sequences = [r.candidate.sequence for r in selected.selected]
        top = frame.set_index("sequence").loc[selected_sequences].reset_index()
        if not set(selected_sequences) <= set(sequences) or len(set(selected_sequences)) != 100:
            raise ValueError("Top containment/count failed")
        for name, values in {
            "challenge_similarity": [r.challenge_similarity for r in selected.selected],
            "known_similarity": [r.known_similarity for r in selected.selected],
            "pairwise_similarity": [r.pairwise_similarity for r in selected.selected],
        }.items():
            top[name] = values
        top.to_csv(output / f"{label}.csv", index=False)
        write_fasta(
            [
                FastaRecord(r.candidate.candidate_id, r.candidate.sequence)
                for r in selected.selected
            ],
            output / f"{label}.fasta",
        )
        current = set(selected_sequences)
        if variant == "L2" and ranker == condition == "B1" and current != baseline_top:
            raise ValueError("L2 B1 Top differs from the frozen Top")
        results.append(
            {
                "label": label,
                "library": variant,
                "ranker": ranker,
                "condition": condition,
                "status": "complete",
                **support,
                "evidence": "proxy",
                "count": len(top),
                "changed_from_frozen_top": len(current - baseline_top),
                "top_jaccard_frozen": len(current & baseline_top) / len(current | baseline_top),
                "apex_vote16": float(top.B2.mean()),
                "apex_mean_mic_um": float(-top.B0.mean()),
                "hc50_coverage": int(top.hc50.notna().sum()),
                "hc50_median_um": float(top.hc50.median()) if top.hc50.notna().all() else None,
                "dev_pass_fraction": float(top.dev_pass.mean())
                if top.dev_pass.notna().all()
                else None,
                "fallback_step": selected.relaxation_step,
                "cluster_cap": selected.cluster_cap,
                "pairwise_threshold": selected.pairwise_threshold,
                "measured_activity": "not measured",
                "measured_hc50": "not available",
            }
        )
        rng = np.random.default_rng(config["random25_seed"])
        indices = np.asarray(
            [rng.choice(len(top), 25, replace=False) for _ in range(config["random25_draws"])]
        )
        np.save(output / f"{label}-random25-indices.npy", indices, allow_pickle=False)
        vote = top.B2.to_numpy()[indices].mean(axis=1)
        hc50 = np.median(top.hc50.to_numpy()[indices], axis=1)
        for i, (v, h) in enumerate(zip(vote, hc50, strict=True)):
            draws.append({"label": label, "draw": i, "apex_vote16": v, "predicted_hc50_median": h})
        random_summaries.append(
            {
                "label": label,
                "evidence": "APEX/HemoPI2 proxy, not wet-lab",
                "draws": len(indices),
                "sample_size": 25,
                **{
                    f"vote16_p{int(q * 100):02}": float(np.quantile(vote, q))
                    for q in [0.05, 0.5, 0.95]
                },
                "hc50_complete_draws": int(np.isfinite(hc50).sum()),
            }
        )
        print(
            f"{label}: Top100, changed={len(current - baseline_top)}, "
            f"fallback={selected.relaxation_step}",
            flush=True,
        )
    table = pd.DataFrame(results)
    table.to_csv(output / "selection_comparison.csv", index=False)
    table[(table.library == "L2") & (table.ranker == "B1")].to_csv(
        output / "filter_ablation.csv", index=False
    )
    pd.DataFrame(random_summaries).to_csv(output / "random25_proxy_summary.csv", index=False)
    pd.DataFrame(draws).to_csv(output / "random25_proxy_draws.csv.gz", index=False)
    write_json(
        output / "similarity_cache.json", {"challenge": challenge_cache, "known": known_cache}
    )


def stability(config: dict[str, Any], root: Path, output: Path, seed: int, size: int) -> None:
    if seed not in config["seeds"] or size not in config["subset_sizes"]:
        raise ValueError("Seed/size is outside the registered protocol")
    _, references, lookup = embedding_lookup(config)
    reference_unique = list(dict.fromkeys(references))
    reference_subset = keyed_subset(reference_unique, size, seed)
    variants = json.loads((root / "prepare/selection_manifest.json").read_text())["variants"]
    metrics, subsets = {}, []
    for variant in variants:
        sequences = read_fasta_sequences(root / "prepare" / f"{variant}.fasta")
        sample = keyed_subset(sequences, size, seed)
        metrics[variant] = _evaluate_variant(
            sequences, sample, references, reference_subset, lookup, seed=seed
        )
        subsets.extend(
            {"variant": variant, "position": i, "sequence": s} for i, s in enumerate(sample)
        )
        print(f"seed={seed} n={size} {variant}: FBD={metrics[variant]['fbd']:.5f}", flush=True)
    subsets.extend(
        {"variant": "reference", "position": i, "sequence": s}
        for i, s in enumerate(reference_subset)
    )
    pd.DataFrame(subsets).to_csv(output / "subset_ids.csv", index=False)
    decision = decide_library_adoption(metrics)
    baseline = metrics["L0"]
    rows = []
    for variant, values in metrics.items():
        rows.append(
            {
                "seed": seed,
                "subset_size": size,
                "variant": variant,
                **values,
                "fbd_gate": values["fbd"] <= baseline["fbd"] * 1.05,
                "conformity_gate": values["conformity"] >= baseline["conformity"] * 0.95,
                "diversity_gate": values["diversity"] >= baseline["diversity"] - 0.01,
                "eligible": decision["variants"][variant]["eligible"],
                "improvement_count": decision["variants"][variant]["improvement_count"],
            }
        )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    write_json(output / "gates.json", decision)


def summarize(config: dict[str, Any], root: Path, output: Path) -> None:
    tables = []
    for seed in config["seeds"]:
        small, large = [root / f"stability-s{seed}-n{n}" for n in config["subset_sizes"]]
        a, b = [pd.read_csv(p / "subset_ids.csv") for p in [small, large]]
        for variant in a.variant.unique():
            if (
                a[a.variant == variant].sequence.tolist()
                != b[b.variant == variant].sequence.tolist()[: len(a[a.variant == variant])]
            ):
                raise ValueError("Subset nesting failed")
        tables.extend(pd.read_csv(p / "metrics.csv") for p in [small, large])
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(output / "library_stability.csv", index=False)
    table.groupby(["variant", "subset_size"])[
        ["fbd_gate", "conformity_gate", "diversity_gate", "eligible"]
    ].mean().to_csv(output / "gate_frequencies.csv")
    # Only observed strain labels are compared; no missing broad panel is imputed.
    measured = Path(config["measured_run"])
    rows = pd.read_csv(measured / "prepare/rows.csv")
    archive = load_prediction_archive(measured / "apex/predictions.npz")
    index = {s: i for i, s in enumerate(archive.sequences)}
    scores = selection_scores(archive.mic_u_m)
    measured_results = []
    for ranker, values in scores.items():
        for pathogen, group in rows.groupby("apex_pathogen"):
            group = group.assign(score=[values[index[s]] for s in group.sequence])
            known = group[group.active16.notna()]
            top = known.sort_values(["score", "sequence"], ascending=[False, True]).head(
                (len(known) + 4) // 5
            )
            measured_results.append(
                {
                    "ranker": ranker,
                    "pathogen": pathogen,
                    "observations": len(known),
                    "groups": known.group.nunique(),
                    "top_count": len(top),
                    "top20_observed_strain_active": float(top.active16.mean()),
                    "primary": pathogen in config["primary_measured_pathogens"],
                    "evidence": "measured observed strain only; frozen APEX overlap unknown",
                    "broad_activity_comparison": "unavailable: incomplete strain panels",
                    "hc50_comparison": "unavailable: no verified measured HC50",
                }
            )
    pd.DataFrame(measured_results).to_csv(output / "measured_selection_comparison.csv", index=False)
    historical = load_measurements(Path("experimental/mic.csv"))
    activity = complete_panel_activity(historical)
    rng = np.random.default_rng(config["random25_seed"])
    indices = np.asarray(
        [rng.choice(len(activity), 25, replace=False) for _ in range(config["random25_draws"])]
    )
    values = activity.to_numpy()[indices].mean(axis=1)
    pd.DataFrame({"draw": range(len(values)), "mean_measured_11strain_activity": values}).to_csv(
        output / "random25_historical_measured_draws.csv", index=False
    )
    write_json(
        output / "random25_historical_measured_summary.json",
        {
            "cohort": "historical selected 46 peptides; not independent validation",
            "sampling": "25 of 46 without replacement; descriptive draws, no model selection",
            "seed": config["random25_seed"],
            "draws": len(values),
            "p05_p50_p95": np.quantile(values, [0.05, 0.5, 0.95]).tolist(),
            "hc50_measured": "unavailable; safety improvement not established",
        },
    )
    write_json(
        output / "inputs_after.json",
        verify_hashes(json.loads((root / "prepare/inputs.json").read_text())),
    )
    write_json(
        output / "decision.json",
        {
            "adopted_policy": "B1/L2/C0",
            "activity_improvement_established": False,
            "safety_improvement_established": False,
            "original_inputs_unchanged": True,
            "reason": "Proxy metrics; independent activity and HC50 evidence lacking",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_selection.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["prepare", "oracles", "tops", "top_pilot", "stability", "summary"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--summary-revision", type=int, default=1)
    args = parser.parse_args()
    if args.summary_revision < 1 or (args.summary_revision != 1 and args.stage != "summary"):
        raise ValueError("A positive summary revision is valid only for summary output")
    config = json.loads(args.config.read_text())
    root = args.output
    suffix = f"stability-s{args.seed}-n{args.size}" if args.stage == "stability" else args.stage
    if args.stage == "summary" and args.summary_revision > 1:
        suffix = f"summary-r{args.summary_revision}"
    output = root / suffix
    if args.stage != "prepare" and config != json.loads(
        (root / "prepare/protocol.json").read_text()
    ):
        raise ValueError("Configuration changed since preparation")
    fresh_output(
        output,
        [Path("data"), Path("reports"), Path(config["frozen_run"]), Path(config["oracle_dir"])],
    )
    start = time.monotonic()
    with threadpool_limits(limits=config["cpu_threads"]):
        if args.stage == "prepare":
            prepare(config, output, args.config)
        elif args.stage == "oracles":
            oracles(config, root, output)
        elif args.stage == "top_pilot":
            tops(config, root, output, Path(config["oracle_cache"]))
        elif args.stage == "tops":
            tops(config, root, output)
        elif args.stage == "stability":
            stability(config, root, output, args.seed, args.size)
        else:
            summarize(config, root, output)
    write_json(
        output / "run_manifest.json",
        {
            "stage": suffix,
            "runtime_seconds": time.monotonic() - start,
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "config_sha256": file_sha256(args.config),
            "script_sha256": file_sha256(Path(__file__)),
            "module_sha256": file_sha256(Path("src/robust_apex_qd/research/selection.py")),
            "uv_lock_sha256": file_sha256(Path("uv.lock")),
            "artifacts_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(f"Completed {suffix}: {time.monotonic() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
