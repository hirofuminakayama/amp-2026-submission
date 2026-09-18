"""Portable HC50 regressors with fold-local preprocessing and feature identity checks."""

from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class HC50Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[2] = 2
    model: Literal["ridge", "median", "interval"]
    endpoint: Literal["consensus_hc50", "measured_hc50"] = "consensus_hc50"
    units: Literal["log2_uM"] = "log2_uM"
    feature_sha256: str
    mean: list[float]
    scale: list[float]
    coefficients: list[float]
    intercept: float
    training_groups: list[str]
    alpha: float


def fit_hc50(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    *,
    alpha: float,
    feature_sha256: str,
    median: bool = False,
) -> HC50Bundle:
    if set(groups[training]) & set(groups[validation]):
        raise ValueError("HC50 training and validation groups cross")
    if not len(training) or not np.isfinite(labels[training]).all():
        raise ValueError("Finite training HC50 labels required")
    if not np.isfinite(features).all():
        raise ValueError("Finite features required")
    scaler = StandardScaler().fit(features[training])
    if median:
        coefficients = np.zeros(features.shape[1])
        intercept = float(np.median(labels[training]))
    else:
        model = Ridge(alpha=alpha).fit(scaler.transform(features[training]), labels[training])
        coefficients, intercept = model.coef_, float(model.intercept_)
    return HC50Bundle(
        model="median" if median else "ridge",
        feature_sha256=feature_sha256,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coefficients=coefficients.tolist(),
        intercept=intercept,
        training_groups=sorted(set(groups[training])),
        alpha=alpha,
    )


def predict_hc50(bundle: HC50Bundle, features: np.ndarray, feature_sha256: str) -> np.ndarray:
    if bundle.feature_sha256 != feature_sha256 or features.shape[1] != len(bundle.coefficients):
        raise ValueError("HC50 feature schema or order mismatch")
    if len(bundle.mean) != features.shape[1] or len(bundle.scale) != features.shape[1]:
        raise ValueError("HC50 preprocessing feature shape mismatch")
    if not np.isfinite(features).all() or np.any(np.asarray(bundle.scale) <= 0):
        raise ValueError("Finite features and positive scale required")
    return ((features - bundle.mean) / bundle.scale) @ bundle.coefficients + bundle.intercept


def hc50_signal(*, mae: float, spearman: float | None, median_mae: float) -> dict[str, bool]:
    quantitative = mae < median_mae
    ranking = spearman is not None and spearman > 0
    return dict(
        quantitative_signal=quantitative,
        ranking_signal=ranking,
        scenario_candidate=quantitative or ranking,
    )


def fit_interval_hc50(
    features: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    *,
    alpha: float,
    feature_sha256: str,
    epochs: int = 200,
) -> HC50Bundle:
    if set(groups[training]) & set(groups[validation]):
        raise ValueError("HC50 endpoint groups cross folds")
    if not len(training) or np.any(lower[training] > upper[training]):
        raise ValueError("Nonempty valid HC50 training bounds required")
    if alpha <= 0 or epochs < 1 or not np.isfinite(features).all():
        raise ValueError("Positive training settings and finite features required")
    if np.isnan(lower[training]).any() or np.isnan(upper[training]).any():
        raise ValueError("Missing bounds must use directional infinities in memory")
    if np.any(~np.isfinite(lower[training]) & ~np.isfinite(upper[training])):
        raise ValueError("At least one finite HC50 bound required")
    scaler = StandardScaler().fit(features[training])
    x = torch.tensor(scaler.transform(features[training]), dtype=torch.float64)
    low, high = torch.tensor(lower[training]), torch.tensor(upper[training])
    initial = np.where(np.isfinite(lower[training]), lower[training], upper[training])
    coef = torch.zeros(features.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(float(np.median(initial)), dtype=torch.float64, requires_grad=True)
    _, inverse, counts = np.unique(groups[training], return_inverse=True, return_counts=True)
    weights = torch.tensor(1.0 / counts[inverse], dtype=torch.float64)
    weights /= weights.sum()
    optimizer = torch.optim.Adam([coef, bias], lr=0.02)
    for _ in range(epochs):
        values = x @ coef + bias
        loss = (
            weights * (torch.relu(low - values).square() + torch.relu(values - high).square())
        ).sum() + alpha * coef.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite HC50 interval loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return HC50Bundle(
        model="interval",
        endpoint="measured_hc50",
        feature_sha256=feature_sha256,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coefficients=coef.detach().tolist(),
        intercept=float(bias.detach()),
        training_groups=sorted(set(groups[training])),
        alpha=alpha,
    )
