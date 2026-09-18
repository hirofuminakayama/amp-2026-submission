"""Grouped exploratory MIC evaluation; exact observations alone train regressors."""

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def select_candidate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in comparisons
        if row["model"] != "APEX"
        and row["macro_top20_active"] is not None
        and np.isfinite(row["macro_top20_active"])
    ]
    if not candidates:
        raise ValueError("No evaluated exploratory candidate")
    return max(
        candidates,
        key=lambda row: (
            round(row["macro_top20_active"], 12),
            -row["mae_log2_exact"],
            row["model"],
        ),
    )


def fold_masks(rows: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    valid = rows.validation_fold.to_numpy() == fold
    other = ~valid
    if set(rows.loc[valid, "group"]) & set(rows.loc[other, "group"]):
        raise ValueError("A group crosses validation folds")
    train = other & rows.exact_mic.to_numpy(bool) & rows.mic_um.notna().to_numpy()
    return train, valid


def evaluate_predictions(
    rows: pd.DataFrame, prediction: str, pathogens: list[str]
) -> dict[str, Any]:
    top = []
    for pathogen in pathogens:
        subset = rows[(rows.apex_pathogen == pathogen) & rows.active16.notna()]
        if subset.empty or not np.isfinite(subset[prediction]).all():
            top.append(float("nan"))
            continue
        selected = subset.sort_values([prediction, "sequence"], kind="stable").head(
            max(1, math.ceil(len(subset) * 0.2))
        )
        top.append(float(selected.active16.mean()))
    exact = rows[rows.exact_mic & rows.mic_um.notna() & np.isfinite(rows[prediction])]
    target = np.log2(exact.mic_um)
    residual = exact[prediction] - target
    classified = rows[rows.active16.notna() & np.isfinite(rows[prediction])]
    return {
        "macro_top20_active": float(np.mean(top)) if top and np.isfinite(top).all() else None,
        "rows": len(rows),
        "groups": rows.group.nunique(),
        "predicted_rows": int(np.isfinite(rows[prediction]).sum()),
        "exact_rows": len(exact),
        "mae_log2_exact": float(residual.abs().mean()) if len(exact) else None,
        "spearman_exact": float(exact[prediction].corr(target, method="spearman"))
        if len(exact) > 1 and exact[prediction].nunique() > 1 and target.nunique() > 1
        else None,
        "accuracy16": float(((classified[prediction] <= 4) == classified.active16).mean())
        if len(classified)
        else None,
    }


def paired_interval(
    rows: pd.DataFrame, prediction: str, pathogens: list[str], seed: int, iterations: int
) -> dict[str, Any]:
    """Resample whole groups jointly for paired top-fraction differences."""
    groups = sorted(rows.group.unique())
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    if len(groups) < 2:
        return {
            "delta_ci_low": None,
            "delta_ci_high": None,
            "valid_replicates": 0,
            "undefined_replicates": iterations,
        }
    rng = np.random.default_rng(seed)
    indices = {group: np.flatnonzero(rows.group.to_numpy() == group) for group in groups}
    values = []
    for _ in range(iterations):
        sampled = rows.iloc[np.concatenate([indices[g] for g in rng.choice(groups, len(groups))])]
        left = evaluate_predictions(sampled, prediction, pathogens)["macro_top20_active"]
        right = evaluate_predictions(sampled, "apex_log2", pathogens)["macro_top20_active"]
        if left is not None and right is not None:
            values.append(left - right)
    ci = np.quantile(values, [0.025, 0.975]) if values else [None, None]
    return {
        "delta_ci_low": ci[0],
        "delta_ci_high": ci[1],
        "valid_replicates": len(values),
        "undefined_replicates": iterations - len(values),
    }


def conditional_features(features: np.ndarray, strains: np.ndarray) -> np.ndarray:
    """Shared sequence effect, fixed strain indicators, and strain-specific sequence effects."""
    onehot = np.eye(11, dtype=np.float32)[strains]
    return np.concatenate(
        [features, onehot, (onehot[:, :, None] * features[:, None, :]).reshape(len(features), -1)],
        axis=1,
    )


def ridge_fold(
    features: np.ndarray,
    strains: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, StandardScaler, Ridge]:
    scaler = StandardScaler().fit(features[train])
    transformed = conditional_features(scaler.transform(features), strains)
    model = Ridge(alpha=alpha).fit(transformed[train], targets[train])
    return model.predict(transformed[valid]), scaler, model
