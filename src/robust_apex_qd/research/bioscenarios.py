"""Conditional joint-hit scenarios and bounded deterministic portfolio selection."""

import math
from typing import Literal, TypedDict

import numpy as np
from scipy.special import ndtr


class SamplingSummary(TypedDict):
    sample_size: int
    draws: int
    analytic_expected_species_average_hits: float
    sampling_only: dict[str, float]
    predictive_scenario: dict[str, float]
    interpretation: str


def biological_rank_scores(
    mic_log2: np.ndarray, joint_probability: np.ndarray, apex_b1_score: np.ndarray
) -> dict[str, np.ndarray]:
    if (
        mic_log2.ndim != 2
        or joint_probability.shape != mic_log2.shape
        or apex_b1_score.shape != (len(mic_log2),)
        or not np.isfinite(mic_log2).all()
        or not np.isfinite(joint_probability).all()
        or not np.isfinite(apex_b1_score).all()
        or np.any((joint_probability < 0) | (joint_probability > 1))
    ):
        raise ValueError("Aligned finite MIC, probability and negative-log2 APEX scores required")
    mic_score = -np.median(mic_log2, axis=1)
    return {
        "bio-MIC": mic_score,
        "bio-joint": joint_probability.mean(axis=1),
        "bio-MIC-apex50": (mic_score + apex_b1_score) / 2,
    }


def empirical_errors(normal: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Finite out-of-fold residuals required")
    ordered = np.sort(residuals)
    return np.interp(ndtr(normal), np.linspace(0, 1, len(ordered)), ordered)


def marginal_joint_probability(
    mic: np.ndarray,
    hc50: np.ndarray,
    mic_residuals: list[np.ndarray],
    hc50_residuals: np.ndarray,
    *,
    ratio: float = 8,
    draws: int = 1000,
    seed: int = 42,
    batch_size: int = 256,
) -> np.ndarray:
    if (
        mic.ndim != 2
        or hc50.shape != (len(mic),)
        or not np.isfinite(mic).all()
        or not np.isfinite(hc50).all()
        or len(mic_residuals) != mic.shape[1]
    ):
        raise ValueError("Complete aligned endpoint prediction coverage required")
    if draws < 1 or batch_size < 1 or ratio <= 0:
        raise ValueError("Positive scenario settings required")
    rng = np.random.default_rng(seed)
    he = empirical_errors(rng.normal(size=draws), hc50_residuals)
    errors = [empirical_errors(rng.normal(size=draws), r) for r in mic_residuals]
    values = np.empty_like(mic, dtype=float)
    # Common integration draws make marginal scores independent of row order and batching.
    # These shared draws do not estimate dependence between candidate outcomes.
    for start in range(0, len(mic), batch_size):
        end = min(start + batch_size, len(mic))
        hc = hc50[None, start:end] + he[:, None]
        for species, error in enumerate(errors):
            activity = mic[None, start:end, species] + error[:, None]
            values[start:end, species] = (
                (activity <= 4) & (hc - activity >= math.log2(ratio))
            ).mean(axis=0)
    return values


def joint_trials(
    mic: np.ndarray,
    hc50: np.ndarray,
    mic_residuals: list[np.ndarray],
    hc50_residuals: np.ndarray,
    *,
    groups: list[str],
    draws: int = 1000,
    ratio: float = 8,
    mic_threshold: float = 16,
    correlation: float = 0,
    dependence: Literal["independent", "cluster", "species"] = "independent",
    seed: int = 42,
) -> np.ndarray:
    if mic.ndim != 2 or hc50.shape != (len(mic),) or len(groups) != len(mic):
        raise ValueError("Aligned candidate and species predictions required")
    if not np.isfinite(mic).all() or not np.isfinite(hc50).all():
        raise ValueError("Complete endpoint prediction coverage required")
    if len(mic_residuals) != mic.shape[1] or not -1 <= correlation <= 1:
        raise ValueError("One residual distribution per species and valid correlation required")
    if draws < 1 or ratio <= 0 or mic_threshold <= 0 or not len(mic) or mic.shape[1] < 1:
        raise ValueError("Positive draws and thresholds required")
    rng = np.random.default_rng(seed)
    if dependence == "cluster":
        _, inverse = np.unique(groups, return_inverse=True)
    elif dependence in {"independent", "species"}:
        inverse = np.arange(len(mic))
    else:
        raise ValueError("Unknown scenario dependence")
    hc_normal = rng.normal(size=(draws, int(inverse.max()) + 1))[:, inverse]
    hc = hc50 + empirical_errors(hc_normal, hc50_residuals)
    trials = np.empty((draws, *mic.shape), dtype=bool)
    for species, residuals in enumerate(mic_residuals):
        if dependence == "species":
            normal = rng.normal(size=(draws, 1))
        else:
            normal = rng.normal(size=(draws, int(inverse.max()) + 1))[:, inverse]
        normal = correlation * hc_normal + math.sqrt(1 - correlation**2) * normal
        activity = mic[:, species] + empirical_errors(normal, residuals)
        trials[:, :, species] = (activity <= math.log2(mic_threshold)) & (
            hc - activity >= math.log2(ratio)
        )
    return trials


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    return dict(
        mean=float(values.mean()),
        std=float(values.std()),
        p10=float(np.quantile(values, 0.1)),
        p50=float(np.quantile(values, 0.5)),
        p90=float(np.quantile(values, 0.9)),
    )


def random25_scenarios(
    trials: np.ndarray,
    *,
    sample_size: int = 25,
    draws: int = 10000,
    seed: int = 42,
) -> SamplingSummary:
    if trials.ndim != 3 or trials.dtype != bool or not len(trials):
        raise ValueError("Boolean scenario x candidate x species events required")
    if not 0 < sample_size <= trials.shape[1] or draws < 1 or trials.shape[2] < 1:
        raise ValueError("Valid sample size and draws required")
    rng = np.random.default_rng(seed)
    conditional = trials.mean(axis=0).mean(axis=1)
    sampling, predictive = np.empty(draws), np.empty(draws)
    for draw in range(draws):
        selected = rng.choice(trials.shape[1], sample_size, replace=False)
        scenario = rng.integers(len(trials))
        sampling[draw] = conditional[selected].sum()
        predictive[draw] = trials[scenario, selected].mean(axis=1).sum()
    return dict(
        sample_size=sample_size,
        draws=draws,
        analytic_expected_species_average_hits=float(sample_size * conditional.mean()),
        sampling_only=numeric_summary(sampling),
        predictive_scenario=numeric_summary(predictive),
        interpretation="conditional species-average joint hits; not a calibrated wet-lab interval",
    )


def select_portfolio(
    scores: np.ndarray,
    identities: list[str],
    *,
    count: int = 100,
    objective: Literal["mean", "cvar"] = "mean",
    groups: list[str] | None = None,
    cap: int | None = None,
) -> list[int]:
    if (
        scores.ndim != 2
        or not len(scores)
        or scores.shape[1] != len(identities)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Finite scenario x candidate scores required")
    if len(set(identities)) != len(identities) or not 0 < count <= len(identities):
        raise ValueError("Unique candidates and sufficient supply required")
    if cap is not None and (cap <= 0 or groups is None or len(groups) != len(identities)):
        raise ValueError("Valid group cap and aligned groups required")
    if objective not in {"mean", "cvar"}:
        raise ValueError("Unknown portfolio objective")
    order = sorted(range(len(identities)), key=lambda i: identities[i])
    selected = []
    totals = np.zeros(scores.shape[0])
    group_counts: dict[str, int] = {}
    tail = max(1, math.ceil(len(scores) * 0.1))
    for _ in range(count):
        eligible = [
            i
            for i in order
            if i not in selected
            and (cap is None or (groups is not None and group_counts.get(groups[i], 0) < cap))
        ]
        if not eligible:
            raise ValueError("Insufficient candidate supply under cluster cap")
        proposed = scores[:, eligible] + totals[:, None]
        if objective == "cvar":
            values = np.partition(proposed, tail - 1, axis=0)[:tail].mean(axis=0)
        else:
            values = proposed.mean(axis=0)
        winner = eligible[int(np.argmax(values))]
        selected.append(winner)
        totals += scores[:, winner]
        if groups is not None:
            group_counts[groups[winner]] = group_counts.get(groups[winner], 0) + 1
    return selected
