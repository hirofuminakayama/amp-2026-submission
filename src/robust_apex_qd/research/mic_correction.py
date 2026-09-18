"""Matched correction-only development comparisons with component-level uncertainty."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_apex_qd.evaluation.readiness import verify_hashes


def correction_masks(
    rows: pd.DataFrame, corrections: set[str], fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not corrections <= set(rows.observation_id):
        raise ValueError("Correction IDs must resolve to supplied observations")
    corrected = rows.observation_id.isin(corrections).to_numpy()
    held = rows.homology_fold.to_numpy() == fold
    train = ~held & rows.exact_regression.to_numpy(bool)
    return train, train & ~corrected, held & ~corrected


def paired_comparison(rows: pd.DataFrame) -> dict[str, Any]:
    mask = (
        rows.exact_regression.to_numpy(bool)
        & np.isfinite(rows.prediction_original.to_numpy(float))
        & np.isfinite(rows.prediction_corrected.to_numpy(float))
    )
    common = rows.loc[mask].copy()
    if len(common):
        y = np.log2(common.mic_um.to_numpy(float))
        common["delta"] = np.abs(common.prediction_corrected.to_numpy() - y) - np.abs(
            common.prediction_original.to_numpy() - y
        )
    else:
        common["delta"] = pd.Series(dtype=float)
    aggregate = common.groupby(["component_id", "species"]).delta.agg(["sum", "count"])
    groups = sorted(set(common.component_id))
    species = sorted(set(common.species))
    sums = np.zeros((len(groups), len(species)))
    counts = np.zeros_like(sums)
    positions = {g: i for i, g in enumerate(groups)}
    targets = {s: i for i, s in enumerate(species)}
    for (group, spec), r in aggregate.iterrows():
        sums[positions[group], targets[spec]] = r["sum"]
        counts[positions[group], targets[spec]] = r["count"]
    estimates = []
    if len(groups) >= 5:
        rng = np.random.default_rng(42)
        for _ in range(1000):
            selected = rng.integers(0, len(groups), len(groups))
            n = counts[selected].sum(axis=0)
            total = sums[selected].sum(axis=0)
            # A replicate lacking a target cannot silently change the macro denominator.
            if np.all(n > 0):
                estimates.append(float(np.mean(total / n)))
    ci = np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else None
    return dict(
        common_exact_rows=len(common),
        excluded_rows=len(rows) - len(common),
        components=len(groups),
        species=len(species),
        delta_macro_mae=float(common.groupby("species").delta.mean().mean())
        if len(common)
        else None,
        ci95=ci,
        bootstrap_valid_replicates=len(estimates),
        bootstrap_missing_target_replicates=1000 - len(estimates) if len(groups) >= 5 else 0,
        direction="corrected minus original; negative favors corrected",
        scope="development OOF; no external efficacy inference",
    )


def verify_completed_fit(
    directory: Path,
    training_ids: list[str],
    validation_ids: list[str],
    arm: dict[str, Any],
    seed: int,
    inputs: dict[str, str],
) -> None:
    """Reuse only complete fits matching registered data, settings, membership and artifacts."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["inputs_sha256"] != inputs:
        raise ValueError("Completed fit input versions differ")
    verify_hashes({str(directory / k): v for k, v in manifest["artifacts_sha256"].items()})
    metadata = json.loads((directory / "fit.json").read_text())
    if any(
        metadata.get(k) != v
        for k, v in dict(
            training_ids=training_ids,
            validation_ids=validation_ids,
            arm=arm,
            seed=seed,
            serialization_equal=True,
        ).items()
    ):
        raise ValueError("Completed fit membership or settings differ")
