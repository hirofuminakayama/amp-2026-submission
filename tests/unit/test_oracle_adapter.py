import csv
from pathlib import Path

import pytest

from robust_apex_qd.evaluation.models import HemoPI2Prediction
from robust_apex_qd.evaluation.oracles import (
    load_hemopi2_predictions,
    write_hemopi2_predictions,
)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("SeqID", "Sequence", "HC50(μM)", "Prediction"))
        writer.writeheader()
        writer.writerows(rows)


def test_hemopi2_csv_is_aligned_and_normalized(tmp_path: Path) -> None:
    path = tmp_path / "final_output.csv"
    _write(
        path,
        [
            {
                "SeqID": "cand_1",
                "Sequence": "ACDEFGHI",
                "HC50(μM)": "99.5",
                "Prediction": "Hemolytic",
            },
            {
                "SeqID": "cand_2",
                "Sequence": "KLMNPQRS",
                "HC50(μM)": "120",
                "Prediction": "Non-Hemolytic",
            },
        ],
    )

    predictions = load_hemopi2_predictions(
        path,
        {"cand_1": "ACDEFGHI", "cand_2": "KLMNPQRS"},
    )

    assert predictions["cand_1"].hemolytic
    assert predictions["cand_2"].hc50_u_m == 120


def test_hemopi2_csv_rejects_missing_or_misaligned_rows(tmp_path: Path) -> None:
    path = tmp_path / "final_output.csv"
    _write(
        path,
        [{"SeqID": "cand_1", "Sequence": "STALESEQ", "HC50(μM)": "10", "Prediction": "Hemolytic"}],
    )

    with pytest.raises(ValueError, match="sequence differs"):
        load_hemopi2_predictions(path, {"cand_1": "ACDEFGHI"})


def test_hemopi2_cache_round_trip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "cache.csv"
    predictions = {
        "cand_2": HemoPI2Prediction(
            candidate_id="cand_2",
            sequence="KLMNPQRS",
            hc50_u_m=120,
            hemolytic=False,
        ),
        "cand_1": HemoPI2Prediction(
            candidate_id="cand_1",
            sequence="ACDEFGHI",
            hc50_u_m=80,
            hemolytic=True,
        ),
    }

    write_hemopi2_predictions(path, predictions)
    first = path.read_bytes()
    write_hemopi2_predictions(path, predictions)

    assert path.read_bytes() == first
    assert (
        load_hemopi2_predictions(
            path,
            {"cand_1": "ACDEFGHI", "cand_2": "KLMNPQRS"},
        )
        == predictions
    )
