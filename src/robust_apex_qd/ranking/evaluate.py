import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kendalltau, spearmanr

from robust_apex_qd.apex.ensemble import aggregate_predictions, load_prediction_archive
from robust_apex_qd.calibration.model import (
    STRAIN_TO_PATHOGEN,
    CalibrationArtifact,
    load_measurements,
)
from robust_apex_qd.features.embeddings import (
    ESM2_EMBEDDING_DIM,
    Esm2BatchEncoder,
    compute_embedding_diagnostics,
    write_embedding_memmap,
)
from robust_apex_qd.features.physchem import (
    PhyschemReference,
    compute_features,
    score_features,
)
from robust_apex_qd.ranking.objectives import (
    broad_objectives,
    conservative_mdr_proxy,
    percentile_score,
    weighted_quality,
)

RANKERS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")


def _calibrated_probabilities(
    tensor: np.ndarray,
    artifact: CalibrationArtifact,
) -> np.ndarray:
    log2_mic = np.log2(tensor)
    feature_values = {
        "median_log2_predicted_mic": np.median(log2_mic, axis=1),
        "model_mad_log2": np.median(
            np.abs(log2_mic - np.median(log2_mic, axis=1, keepdims=True)),
            axis=1,
        ),
    }
    matrix = np.stack([feature_values[name] for name in artifact.feature_order], axis=-1)
    coefficients = np.asarray(artifact.coefficients, dtype=np.float64)
    return expit(matrix @ coefficients + artifact.intercept)


def _measured_ground_truth(measurements: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for peptide_id in sorted(measurements["peptide_id"].unique()):
        group = measurements.loc[measurements["peptide_id"] == peptide_id]
        sequence = str(group["sequence"].iloc[0])
        active_by_pathogen = {
            STRAIN_TO_PATHOGEN[str(row["strain"])]: int(row["active"])
            for _, row in group.iterrows()
        }
        mdr_values = (
            active_by_pathogen["A. baumannii ATCC 19606"],
            active_by_pathogen["E. coli AIC222"],
            min(
                active_by_pathogen["E. coli ATCC 11775"],
                active_by_pathogen["E. coli AIC221"],
                active_by_pathogen["E. coli AIC222"],
            ),
            active_by_pathogen["K. pneumoniae ATCC 13883"],
            min(
                active_by_pathogen["P. aeruginosa PA01"],
                active_by_pathogen["P. aeruginosa PA14"],
            ),
            active_by_pathogen["S. aureus (ATCC BAA-1556) - MRSA"],
            active_by_pathogen["vancomycin-resistant E. faecalis ATCC 700802"],
            active_by_pathogen["vancomycin-resistant E. faecium ATCC 700221"],
        )
        rows.append(
            {
                "peptide_id": str(peptide_id),
                "sequence": str(sequence),
                "measured_success_rate_16": float(group["active"].mean()),
                "measured_mic50_u_m": float(group["mic"].quantile(0.50)),
                "measured_mic90_u_m": float(group["mic"].quantile(0.90)),
                "measured_mdr_success_rate": float(np.mean(mdr_values)),
            }
        )
    return pd.DataFrame(rows)


def _ranker_scores(
    tensor: np.ndarray,
    artifact: CalibrationArtifact,
    physchem_ood: np.ndarray,
    embedding_ood: np.ndarray,
) -> dict[str, np.ndarray]:
    aggregates = aggregate_predictions(tensor)
    c0_probabilities = (tensor <= 16.0).mean(axis=1)
    calibrated_probabilities = _calibrated_probabilities(tensor, artifact)
    c0_broad, _ = broad_objectives(c0_probabilities)
    broad_mean, broad_tail = broad_objectives(calibrated_probabilities)
    mdr_proxy = conservative_mdr_proxy(calibrated_probabilities)
    disagreement = aggregates.model_disagreement_mad_log2.astype(np.float64)
    b4 = 0.75 * percentile_score(broad_mean) + 0.25 * percentile_score(broad_tail)
    b5 = (
        0.60 * percentile_score(broad_mean)
        + 0.20 * percentile_score(broad_tail)
        + 0.20 * percentile_score(mdr_proxy)
        - 0.15 * percentile_score(disagreement)
    )
    return {
        "B0": percentile_score(
            aggregates.official_broad_mean_mic_u_m,
            higher_is_better=False,
        ),
        "B1": percentile_score(aggregates.median_log2_mic, higher_is_better=False),
        "B2": percentile_score(c0_broad),
        "B3": percentile_score(broad_mean),
        "B4": b4,
        "B5": b5,
        "B6": weighted_quality(
            broad_mean=broad_mean,
            broad_tail=broad_tail,
            mdr_proxy=mdr_proxy,
            disagreement=disagreement,
            physchem_ood=physchem_ood,
            embedding_ood=embedding_ood,
        ),
    }


def _safe_correlation(statistic: float) -> float:
    return float(statistic) if np.isfinite(statistic) else 0.0


def _ranker_metrics(rows: pd.DataFrame, score_name: str) -> dict[str, float | int]:
    ordered = rows.sort_values(
        [score_name, "peptide_id"],
        ascending=[False, True],
        kind="stable",
    )
    target = rows["measured_success_rate_16"]
    score = rows[score_name]
    metrics: dict[str, float | int] = {
        "spearman": _safe_correlation(spearmanr(score, target).statistic),
        "kendall_tau": _safe_correlation(kendalltau(score, target).statistic),
    }
    for top_k in (5, 10, 20):
        top = ordered.head(top_k)
        metrics[f"top_{top_k}_mean_success"] = float(top["measured_success_rate_16"].mean())
        metrics[f"top_{top_k}_active_on_half_count"] = int(
            (top["measured_success_rate_16"] >= 0.5).sum()
        )
    return metrics


def _bootstrap_ranker_comparison(
    rows: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    superiority = {ranker: {"top_10": 0, "top_20": 0} for ranker in RANKERS[1:]}
    intervals = {ranker: {"top_10": [], "top_20": []} for ranker in RANKERS}
    for _ in range(iterations):
        sampled_indices = rng.integers(0, len(rows), size=len(rows))
        sampled = rows.iloc[sampled_indices].copy()
        sampled["peptide_id"] = [f"sample_{index}" for index in range(len(sampled))]
        values = {ranker: _ranker_metrics(sampled, f"score_{ranker}") for ranker in RANKERS}
        for ranker in RANKERS:
            for top_k in (10, 20):
                intervals[ranker][f"top_{top_k}"].append(
                    values[ranker][f"top_{top_k}_mean_success"]
                )
        for ranker in RANKERS[1:]:
            for top_k in (10, 20):
                if (
                    values[ranker][f"top_{top_k}_mean_success"]
                    >= values["B0"][f"top_{top_k}_mean_success"]
                ):
                    superiority[ranker][f"top_{top_k}"] += 1
    return {
        "seed": seed,
        "iterations": iterations,
        "top_success_95": {
            ranker: {
                name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                for name, values in ranker_values.items()
            }
            for ranker, ranker_values in intervals.items()
        },
        "probability_not_worse_than_B0": {
            ranker: {name: count / iterations for name, count in ranker_values.items()}
            for ranker, ranker_values in superiority.items()
        },
    }


def run_ranker_evaluation(
    *,
    measurements_path: Path,
    apex_predictions_path: Path,
    calibration_path: Path,
    physchem_reference_path: Path,
    reference_embeddings_path: Path,
    peptide_embeddings_path: Path,
    report_path: Path,
    bootstrap_path: Path,
    device: str,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    measurements = load_measurements(measurements_path)
    archive = load_prediction_archive(apex_predictions_path)
    artifact = CalibrationArtifact.model_validate_json(calibration_path.read_text())
    reference = PhyschemReference.model_validate_json(physchem_reference_path.read_text())
    physchem_ood = np.asarray(
        [
            score_features(compute_features(sequence), reference).physchem_ood
            for sequence in archive.sequences
        ],
        dtype=np.float64,
    )
    encoder = Esm2BatchEncoder(device)
    write_embedding_memmap(
        archive.sequences,
        peptide_embeddings_path,
        512,
        encoder,
    )
    peptide_embeddings = np.load(peptide_embeddings_path, mmap_mode="r")
    reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
    if peptide_embeddings.shape != (len(archive.sequences), ESM2_EMBEDDING_DIM):
        raise ValueError("Experimental peptide embedding shape is invalid")
    diagnostics = compute_embedding_diagnostics(
        peptide_embeddings,
        reference_embeddings,
        seed=seed,
        pca_components=64,
        cluster_count=min(8, len(archive.sequences)),
        pca_reference_subset=10_000,
        thread_count=1,
    )
    scores = _ranker_scores(
        archive.mic_u_m,
        artifact,
        physchem_ood,
        diagnostics.embedding_ood,
    )
    ground_truth = _measured_ground_truth(measurements)
    sequence_to_index = {sequence: index for index, sequence in enumerate(archive.sequences)}
    if set(ground_truth["sequence"]) != set(sequence_to_index):
        raise ValueError("Measured and APEX peptide sequence sets differ")
    aligned_indices = np.asarray(
        [sequence_to_index[sequence] for sequence in ground_truth["sequence"]],
        dtype=np.int64,
    )
    for ranker, values in scores.items():
        ground_truth[f"score_{ranker}"] = values[aligned_indices]
    metrics = {ranker: _ranker_metrics(ground_truth, f"score_{ranker}") for ranker in RANKERS}
    bootstrap = _bootstrap_ranker_comparison(
        ground_truth,
        seed=seed,
        iterations=bootstrap_iterations,
    )
    b0 = metrics["B0"]
    component_count = {"B1": 1, "B2": 1, "B3": 1, "B4": 2, "B5": 4, "B6": 6}
    eligible = [
        ranker
        for ranker in RANKERS[1:]
        if metrics[ranker]["top_10_mean_success"] >= b0["top_10_mean_success"]
        and metrics[ranker]["top_20_mean_success"] >= b0["top_20_mean_success"]
        and metrics[ranker]["spearman"] >= b0["spearman"] - 0.02
        and bootstrap["probability_not_worse_than_B0"][ranker]["top_10"] >= 0.5
        and bootstrap["probability_not_worse_than_B0"][ranker]["top_20"] >= 0.5
    ]
    adopted = min(
        eligible,
        key=lambda ranker: (component_count[ranker], RANKERS.index(ranker)),
        default="B0",
    )
    report = pd.DataFrame(
        [{"ranker": ranker, **metrics[ranker], "adopted": ranker == adopted} for ranker in RANKERS]
    )
    bootstrap["adoption"] = {
        "adopted_ranker": adopted,
        "eligible_rankers": eligible,
        "rule": "top10/top20 >= B0; Spearman >= B0-0.02; bootstrap non-worse >= 0.5",
        "apex_proxy_is_not_wet_lab_prediction": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, float_format="%.10g", lineterminator="\n")
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n")
    return report, bootstrap
