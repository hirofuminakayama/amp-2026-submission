"""Homology-fold predictor primitives with explicit unsupported target handling."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SPECIES = (
    "Acinetobacter baumannii",
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Staphylococcus aureus",
    "Enterococcus faecalis",
    "Enterococcus faecium",
)
STRAIN_SPECIES = np.array([0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 6])


def bounded_loss(
    prediction: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    return torch.relu(lower - prediction).square() + torch.relu(prediction - upper).square()


def checked_masks(rows: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    if rows.groupby("homology_group").homology_fold.nunique().max() != 1:
        raise ValueError("A homology group crosses folds")
    valid = rows.homology_fold.to_numpy() == fold
    train = ~valid
    if fold >= 0 and (not valid.any() or not train.any()):
        raise ValueError("Empty training or validation fold")
    return train, valid


def fit_ridge_heads(
    features: np.ndarray, heads: np.ndarray, target: np.ndarray, head_count: int, alpha: float
) -> dict[str, Any]:
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)
    coef = np.zeros((head_count, features.shape[1]))
    intercept = np.zeros(head_count)
    support = np.zeros(head_count, dtype=bool)
    for h in range(head_count):
        mask = heads == h
        if mask.any():
            model = Ridge(alpha=alpha).fit(x[mask], target[mask])
            coef[h], intercept[h], support[h] = model.coef_, model.intercept_, True
    return dict(
        mean=scaler.mean_, scale=scaler.scale_, coef=coef, intercept=intercept, support=support
    )


def predict_ridge_heads(state: dict[str, Any], features: np.ndarray) -> np.ndarray:
    values = ((features - state["mean"]) / state["scale"]) @ state["coef"].T + state["intercept"]
    values[:, ~state["support"]] = np.nan
    return values


def supported_blend(
    new: np.ndarray, apex: np.ndarray, apex_weight: float
) -> tuple[np.ndarray, np.ndarray]:
    if new.shape != apex.shape or not 0 <= apex_weight <= 1:
        raise ValueError("Aligned heads and convex ensemble weight required")
    support = np.isfinite(new)
    return np.where(support, (1 - apex_weight) * new + apex_weight * apex, apex), support
