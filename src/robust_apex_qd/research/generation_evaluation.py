"""Read-only prediction using frozen strain-conditional ridge parameters."""

from collections.abc import Mapping

import numpy as np

from robust_apex_qd.research.models import conditional_features


def frozen_ridge_predict(
    features: np.ndarray, strain: int, state: Mapping[str, np.ndarray]
) -> np.ndarray:
    scaled = (features - state["mean"]) / state["scale"]
    design = conditional_features(scaled, np.full(len(features), strain))
    return design @ state["coef"] + state["intercept"]
