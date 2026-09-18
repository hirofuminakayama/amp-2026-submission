import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, ApexPredictionArchive

ACTIVITY_THRESHOLD_UM = 16.0
FEATURE_ORDER = {
    "C0": ("predicted_success_fraction",),
    "C1": ("median_log2_predicted_mic",),
    "C2": ("median_log2_predicted_mic", "model_mad_log2"),
}
STRAIN_TO_PATHOGEN = {
    **{pathogen: pathogen for pathogen in APEX_PATHOGENS},
    "Acinetobacter baumannii ATCC 19606": APEX_PATHOGENS[0],
    "Escherichia coli ATCC 11775": APEX_PATHOGENS[1],
    "Escherichia coli AIC221": APEX_PATHOGENS[2],
    "Escherichia coli AIC222 (CRE)": APEX_PATHOGENS[3],
    "Klebsiella pneumoniae ATCC 13883": APEX_PATHOGENS[4],
    "Pseudomonas aeruginosa PAO1": APEX_PATHOGENS[5],
    "Pseudomonas aeruginosa PA14": APEX_PATHOGENS[6],
    "Staphylococcus aureus ATCC 12600": APEX_PATHOGENS[7],
    "Staphylococcus aureus ATCC BAA-1556 (MRSA)": APEX_PATHOGENS[8],
    "Enterococcus faecalis ATCC 700802 (VRE)": APEX_PATHOGENS[9],
    "Enterococcus faecium ATCC 700221 (VRE)": APEX_PATHOGENS[10],
}


class CalibrationArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    variant: str
    feature_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_sha256: str
    cv_metrics: dict[str, float]
    seed: int
    folds: int


@dataclass(frozen=True)
class CalibrationEvaluation:
    oof: pd.DataFrame
    summary: dict[str, Any]


def load_measurements(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "peptide_id",
        "sequence",
        "strain",
        "mic",
        "mic_unit",
        "mic_relation",
    }
    if not required <= set(data.columns):
        raise ValueError(f"MIC data is missing columns: {sorted(required - set(data.columns))}")
    if data[list(required)].isna().any().any():
        raise ValueError("MIC data contains missing required values")
    if set(data["mic_unit"].astype(str)) != {"uM"}:
        raise ValueError("MIC data must use uM units")
    if not set(data["mic_relation"].astype(str)) <= {"=", ">"}:
        raise ValueError("MIC relation must be '=' or '>'")
    if not set(data["strain"].astype(str)) <= set(STRAIN_TO_PATHOGEN):
        unknown = sorted(set(data["strain"].astype(str)) - set(STRAIN_TO_PATHOGEN))
        raise ValueError(f"MIC data contains unknown strains: {unknown}")
    normalized = data.copy()
    normalized["mic"] = pd.to_numeric(normalized["mic"], errors="raise")
    if not np.isfinite(normalized["mic"]).all() or (normalized["mic"] <= 0).any():
        raise ValueError("MIC values must be positive and finite")
    identities = normalized[["peptide_id", "sequence"]].drop_duplicates()
    if identities["peptide_id"].duplicated().any() or identities["sequence"].duplicated().any():
        raise ValueError("Peptide IDs and sequences must map one-to-one")
    pairs = normalized.assign(pathogen=normalized["strain"].map(STRAIN_TO_PATHOGEN))
    if pairs.duplicated(["peptide_id", "pathogen"]).any():
        raise ValueError("MIC data contains duplicate peptide/pathogen measurements")
    if ((normalized["mic_relation"] == ">") & (normalized["mic"] < ACTIVITY_THRESHOLD_UM)).any():
        raise ValueError("Censoring below the activity threshold leaves activity unknown")
    normalized["active"] = (
        (normalized["mic_relation"] == "=") & (normalized["mic"] <= ACTIVITY_THRESHOLD_UM)
    ).astype(np.int8)
    return normalized


def build_calibration_rows(
    measurements: pd.DataFrame,
    archive: ApexPredictionArchive,
) -> pd.DataFrame:
    if archive.pathogens != APEX_PATHOGENS:
        raise ValueError("APEX archive pathogen order differs from the calibration contract")
    sequence_to_index = {sequence: index for index, sequence in enumerate(archive.sequences)}
    if len(sequence_to_index) != len(archive.sequences):
        raise ValueError("APEX calibration sequences must be unique")
    rows: list[dict[str, object]] = []
    for _, measurement in measurements.iterrows():
        sequence = str(measurement["sequence"])
        if sequence not in sequence_to_index:
            raise ValueError(f"Measured sequence is absent from APEX predictions: {sequence}")
        pathogen = STRAIN_TO_PATHOGEN[str(measurement["strain"])]
        pathogen_index = APEX_PATHOGENS.index(pathogen)
        model_mic = archive.mic_u_m[sequence_to_index[sequence], :, pathogen_index]
        log2_mic = np.log2(model_mic)
        rows.append(
            {
                "peptide_id": str(measurement["peptide_id"]),
                "sequence": sequence,
                "strain": str(measurement["strain"]),
                "active": int(measurement["active"]),
                "predicted_success_fraction": float(np.mean(model_mic <= ACTIVITY_THRESHOLD_UM)),
                "median_log2_predicted_mic": float(np.median(log2_mic)),
                "model_mad_log2": float(np.median(np.abs(log2_mic - np.median(log2_mic)))),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(measurements):
        raise RuntimeError("Calibration row construction lost measurements")
    return result


def _metric_values(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    labels = rows["active"].to_numpy(dtype=np.int8)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    peptide = rows[["peptide_id", "active"]].copy()
    peptide["probability"] = probabilities
    per_peptide = peptide.groupby("peptide_id", sort=True).mean(numeric_only=True)
    if per_peptide["active"].nunique() > 1 and per_peptide["probability"].nunique() > 1:
        correlation = spearmanr(
            per_peptide["active"],
            per_peptide["probability"],
        ).statistic
    else:
        correlation = 0.0
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "peptide_success_spearman": float(correlation if np.isfinite(correlation) else 0.0),
    }


def _bootstrap_intervals(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, list[float]]:
    if iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    peptide_ids = np.asarray(sorted(rows["peptide_id"].unique()))
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        name: [] for name in ("brier", "log_loss", "auroc", "auprc", "peptide_success_spearman")
    }
    indexed = rows.reset_index(drop=True)
    for _ in range(iterations):
        selected = rng.choice(peptide_ids, size=len(peptide_ids), replace=True)
        indices = np.concatenate(
            [
                np.flatnonzero(indexed["peptide_id"].to_numpy() == peptide_id)
                for peptide_id in selected
            ]
        )
        sampled_rows = indexed.iloc[indices].copy()
        sampled_rows["peptide_id"] = [
            f"bootstrap_{draw_index}_{peptide_id}"
            for draw_index, peptide_id in enumerate(selected)
            for _ in range((indexed["peptide_id"] == peptide_id).sum())
        ]
        labels = sampled_rows["active"].to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        values = _metric_values(sampled_rows, probabilities[indices])
        for name, value in values.items():
            samples[name].append(value)
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in samples.items()
        if values
    }


def _training_sha256(rows: pd.DataFrame) -> str:
    columns = (
        "peptide_id",
        "sequence",
        "strain",
        "active",
        "predicted_success_fraction",
        "median_log2_predicted_mic",
        "model_mad_log2",
    )
    payload = rows.loc[:, columns].to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_calibration(
    rows: pd.DataFrame,
    *,
    seed: int,
    folds: int,
    bootstrap_iterations: int,
) -> CalibrationEvaluation:
    if rows["peptide_id"].nunique() < folds:
        raise ValueError("Calibration requires at least one peptide group per fold")
    labels = rows["active"].to_numpy(dtype=np.int8)
    groups = rows["peptide_id"].to_numpy()
    splitter = GroupKFold(n_splits=folds)
    fold_ids = np.full(len(rows), -1, dtype=np.int16)
    probabilities = {
        "C0": rows["predicted_success_fraction"].to_numpy(dtype=np.float64),
        "C1": np.empty(len(rows), dtype=np.float64),
        "C2": np.empty(len(rows), dtype=np.float64),
    }
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(rows, labels, groups=groups)
    ):
        if set(groups[train_indices]) & set(groups[validation_indices]):
            raise RuntimeError("Peptide leakage detected in GroupKFold")
        fold_ids[validation_indices] = fold
        for variant in ("C1", "C2"):
            features = list(FEATURE_ORDER[variant])
            model = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=1000)
            model.fit(rows.iloc[train_indices][features], labels[train_indices])
            probabilities[variant][validation_indices] = model.predict_proba(
                rows.iloc[validation_indices][features]
            )[:, 1]
    if np.any(fold_ids < 0):
        raise RuntimeError("GroupKFold did not assign every calibration row")
    oof = rows[["peptide_id", "sequence", "strain", "active"]].copy()
    oof.insert(3, "fold", fold_ids)
    for variant in ("C0", "C1", "C2"):
        oof[f"probability_{variant}"] = probabilities[variant]
    variants: dict[str, Any] = {}
    for index, variant in enumerate(("C0", "C1", "C2")):
        metrics = _metric_values(rows, probabilities[variant])
        variants[variant] = {
            "metrics": metrics,
            "bootstrap_95": _bootstrap_intervals(
                rows,
                probabilities[variant],
                seed=seed + index,
                iterations=bootstrap_iterations,
            ),
        }
    c0_metrics = variants["C0"]["metrics"]
    eligible = [
        variant
        for variant in ("C1", "C2")
        if variants[variant]["metrics"]["brier"] < c0_metrics["brier"]
        and variants[variant]["metrics"]["peptide_success_spearman"]
        >= c0_metrics["peptide_success_spearman"]
    ]
    adopted = (
        min(eligible, key=lambda name: variants[name]["metrics"]["brier"]) if eligible else None
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "folds": folds,
        "measurement_count": len(rows),
        "peptide_count": int(rows["peptide_id"].nunique()),
        "training_sha256": _training_sha256(rows),
        "feature_order": {name: list(values) for name, values in FEATURE_ORDER.items()},
        "variants": variants,
        "adopted_variant": adopted,
        "use_calibration": adopted is not None,
    }
    return CalibrationEvaluation(oof=oof, summary=summary)


def fit_calibration_artifact(
    rows: pd.DataFrame,
    evaluation: CalibrationEvaluation,
    *,
    variant: str,
) -> CalibrationArtifact:
    if variant not in ("C1", "C2"):
        raise ValueError("Only C1 or C2 can be serialized as a fitted calibrator")
    features = list(FEATURE_ORDER[variant])
    model = LogisticRegression(
        random_state=int(evaluation.summary["seed"]),
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(rows[features], rows["active"])
    metrics = evaluation.summary["variants"][variant]["metrics"]
    return CalibrationArtifact(
        schema_version=1,
        variant=variant,
        feature_order=tuple(features),
        coefficients=tuple(float(value) for value in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        training_sha256=str(evaluation.summary["training_sha256"]),
        cv_metrics={name: float(value) for name, value in metrics.items()},
        seed=int(evaluation.summary["seed"]),
        folds=int(evaluation.summary["folds"]),
    )
