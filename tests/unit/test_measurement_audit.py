from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.calibration.model import load_measurements


def test_measurement_loader_rejects_duplicate_alias_and_ambiguous_censor(tmp_path: Path) -> None:
    row = dict(
        peptide_id="p",
        sequence="ACDEFGHI",
        strain="E. coli ATCC 11775",
        mic=16,
        mic_unit="uM",
        mic_relation="=",
    )
    path = tmp_path / "mic.csv"
    for rows in (
        [row, {**row, "strain": "Escherichia coli ATCC 11775"}],
        [{**row, "mic": 8, "mic_relation": ">"}],
    ):
        pd.DataFrame(rows).to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_measurements(path)


def test_measurement_loader_rejects_inconsistent_identity(tmp_path: Path) -> None:
    path = tmp_path / "mic.csv"
    pd.DataFrame(
        [
            dict(
                peptide_id="p",
                sequence="ACDEFGHI",
                strain="E. coli ATCC 11775",
                mic=16,
                mic_unit="uM",
                mic_relation="=",
            ),
            dict(
                peptide_id="p",
                sequence="KLMNPQRS",
                strain="E. coli AIC221",
                mic=64,
                mic_unit="uM",
                mic_relation=">",
            ),
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one-to-one"):
        load_measurements(path)


def test_paired_audit_keeps_undefined_and_resamples_peptides() -> None:
    from robust_apex_qd.ranking.audit import paired_audit

    rows = pd.DataFrame(
        {
            "peptide_id": ["a", "b", "c"],
            "measured_success_rate_16": [0.0, 0.5, 1.0],
            "score_B0": [0.0, 0.5, 1.0],
            "score_B1": [0.0, 0.5, 1.0],
        }
    )
    first = paired_audit(rows, seed=42, iterations=30)
    assert first == paired_audit(rows, seed=42, iterations=30)
    assert first["delta_B1_minus_B0"]["top_10_mean_success"]["interval_95"] == [0.0, 0.0]
    rows["measured_success_rate_16"] = np.zeros(3)
    constant = paired_audit(rows, seed=42, iterations=30)
    assert constant["point"]["B0"]["spearman"] is None
    assert constant["delta_B1_minus_B0"]["spearman"]["valid_replicates"] == 0
