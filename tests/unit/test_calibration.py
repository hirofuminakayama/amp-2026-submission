import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, ApexPredictionArchive
from robust_apex_qd.calibration.model import (
    CalibrationArtifact,
    build_calibration_rows,
    evaluate_calibration,
    load_measurements,
)


def _measurement_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for peptide_index in range(10):
        for pathogen_index, strain in enumerate(APEX_PATHOGENS[:2]):
            active = (peptide_index + pathogen_index) % 2 == 0
            rows.append(
                {
                    "peptide_id": f"pep_{peptide_index}",
                    "sequence": "ACDEFGHI" + "K" * peptide_index,
                    "strain": strain,
                    "mic": 8.0 if active else 64.0,
                    "mic_unit": "uM",
                    "mic_relation": "=" if active else ">",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _archive() -> ApexPredictionArchive:
    sequences = tuple("ACDEFGHI" + "K" * index for index in range(10))
    tensor = np.full((10, 8, 11), 64.0, dtype=np.float32)
    for peptide_index in range(10):
        for pathogen_index in range(2):
            active = (peptide_index + pathogen_index) % 2 == 0
            tensor[peptide_index, : (6 if active else 2), pathogen_index] = 8.0
    return ApexPredictionArchive(
        sequences=sequences,
        model_names=tuple(f"model_{index}" for index in range(8)),
        pathogens=APEX_PATHOGENS,
        mic_u_m=tensor,
    )


def test_loader_normalizes_activity_and_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "mic.csv"
    pd.DataFrame(
        [
            {
                "peptide_id": "p1",
                "sequence": "ACDEFGHI",
                "strain": APEX_PATHOGENS[0],
                "mic": 16,
                "mic_unit": "uM",
                "mic_relation": "=",
            },
            {
                "peptide_id": "p2",
                "sequence": "KWKWKWKW",
                "strain": APEX_PATHOGENS[0],
                "mic": 64,
                "mic_unit": "uM",
                "mic_relation": ">",
            },
        ]
    ).to_csv(path, index=False)

    loaded = load_measurements(path)

    assert loaded["active"].tolist() == [1, 0]
    assert loaded["peptide_id"].nunique() == 2

    broken = pd.read_csv(path)
    broken.loc[0, "mic"] = np.nan
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_measurements(path)


def test_group_oof_is_deterministic_and_has_no_peptide_leakage(tmp_path: Path) -> None:
    path = tmp_path / "mic.csv"
    _measurement_csv(path)
    rows = build_calibration_rows(load_measurements(path), _archive())

    first = evaluate_calibration(rows, seed=42, folds=5, bootstrap_iterations=40)
    second = evaluate_calibration(rows, seed=42, folds=5, bootstrap_iterations=40)

    pd.testing.assert_frame_equal(first.oof, second.oof)
    assert first.summary == second.summary
    assert set(first.summary["variants"]) == {"C0", "C1", "C2"}
    assert first.summary["feature_order"] == {
        "C0": ["predicted_success_fraction"],
        "C1": ["median_log2_predicted_mic"],
        "C2": ["median_log2_predicted_mic", "model_mad_log2"],
    }
    for fold in range(5):
        validation = set(first.oof.loc[first.oof["fold"] == fold, "peptide_id"])
        training = set(first.oof.loc[first.oof["fold"] != fold, "peptide_id"])
        assert validation.isdisjoint(training)


def test_calibration_artifact_round_trip() -> None:
    artifact = CalibrationArtifact(
        schema_version=1,
        variant="C1",
        feature_order=("median_log2_predicted_mic",),
        coefficients=(-2.0,),
        intercept=-1.0,
        training_sha256="abc123",
        cv_metrics={"brier": 0.2},
        seed=42,
        folds=5,
    )

    loaded = CalibrationArtifact.model_validate_json(artifact.model_dump_json())

    assert loaded == artifact
    assert json.loads(artifact.model_dump_json())["feature_order"] == ["median_log2_predicted_mic"]
