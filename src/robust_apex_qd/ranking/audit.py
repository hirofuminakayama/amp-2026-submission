from typing import Any

import numpy as np
import pandas as pd

from robust_apex_qd.ranking.evaluate import _ranker_metrics


def audit_metrics(rows: pd.DataFrame, score: str) -> dict[str, float | int | None]:
    undefined = rows[score].nunique() < 2 or rows["measured_success_rate_16"].nunique() < 2
    # The legacy reporting helper maps undefined correlations to zero. Audit output must not.
    if undefined:
        ordered = rows.sort_values([score, "peptide_id"], ascending=[False, True], kind="stable")
        values: dict[str, float | int | None] = {"spearman": None, "kendall_tau": None}
        for k in (5, 10, 20):
            top = ordered.head(k)["measured_success_rate_16"]
            values[f"top_{k}_mean_success"] = float(top.mean())
            values[f"top_{k}_active_on_half_count"] = int((top >= 0.5).sum())
        return values
    return dict(_ranker_metrics(rows, score))


def paired_audit(rows: pd.DataFrame, *, seed: int, iterations: int) -> dict[str, Any]:
    if iterations <= 0 or rows.empty or rows["peptide_id"].duplicated().any():
        raise ValueError("Positive iterations and one row per unique peptide are required")
    if not np.isfinite(rows[["score_B0", "score_B1", "measured_success_rate_16"]]).all().all():
        raise ValueError("Audit scores and targets must be finite")
    point = {name: audit_metrics(rows, f"score_{name}") for name in ("B0", "B1")}
    samples: dict[str, list[float]] = {key: [] for key in point["B0"]}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled = rows.iloc[rng.integers(0, len(rows), size=len(rows))].copy()
        sampled["peptide_id"] = [f"sample_{index}" for index in range(len(sampled))]
        values = {name: audit_metrics(sampled, f"score_{name}") for name in ("B0", "B1")}
        for key in samples:
            left, right = values["B0"][key], values["B1"][key]
            if left is not None and right is not None:
                samples[key].append(float(right - left))
    return {
        "seed": seed,
        "iterations": iterations,
        "resampling_unit": "peptide",
        "point": point,
        "delta_B1_minus_B0": {
            key: {
                "valid_replicates": len(values),
                "undefined_replicates": iterations - len(values),
                "interval_95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
            }
            for key, values in samples.items()
        },
    }
