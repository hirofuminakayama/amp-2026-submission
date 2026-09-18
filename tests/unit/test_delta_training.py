import numpy as np
import pytest
import torch

from robust_apex_qd.research.mic_delta import DeltaObservation, DeltaPair
from robust_apex_qd.research.mic_models import fit_regressor, predict_regressor


def test_auxiliary_fit_preserves_zero_weight_and_checks_endpoints() -> None:
    common = dict(
        scaffold_id="s",
        target_id="t",
        chemical_profile="linear",
        study_id="p",
        comparable_assay_id="a",
        verification_evidence="fixture",
        objective="measured_mic",
        relation="=",
        assay_publication_verified=True,
    )
    pair = DeltaPair(
        left=DeltaObservation.model_validate(
            dict(observation_id="a", sequence="AA", mic_um=1, **common)
        ),
        right=DeltaObservation.model_validate(
            dict(observation_id="b", sequence="AB", mic_um=4, **common)
        ),
        partition="train",
    )
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 2.0, 1.0])
    settings = dict(
        width=4,
        heads=1,
        scale_floor=0.1,
        device="cpu",
        epochs=3,
        batch_size=2,
        learning_rate=0.01,
        loss="interval",
    )
    plain = fit_regressor(x, np.zeros(3, int), y, y, settings, 42)
    zero = fit_regressor(
        x,
        np.zeros(3, int),
        y,
        y,
        settings,
        42,
        pairs=[pair],
        observation_ids=["a", "b", "c"],
        delta_weight=0,
    )
    np.testing.assert_equal(predict_regressor(plain, x)[0], predict_regressor(zero, x)[0])
    fitted = fit_regressor(
        x,
        np.zeros(3, int),
        y,
        y,
        settings,
        42,
        pairs=[pair],
        observation_ids=["a", "b", "c"],
        delta_weight=0.3,
    )
    assert all(torch.isfinite(v).all() for v in fitted["state"].values())
    assert not np.array_equal(predict_regressor(plain, x)[0], predict_regressor(fitted, x)[0])
    with pytest.raises(ValueError):
        fit_regressor(
            x,
            np.zeros(3, int),
            y,
            y,
            settings,
            42,
            pairs=[pair],
            observation_ids=["a", "missing", "c"],
            delta_weight=0.3,
        )
