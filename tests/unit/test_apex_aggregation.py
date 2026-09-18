import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.apex.ensemble import (
    APEX_PATHOGENS,
    ApexAggregates,
    aggregate_predictions,
    load_prediction_archive,
    write_json_manifest,
    write_prediction_archive,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mock_tensor_axis_and_aggregates() -> None:
    mic_u_m = np.full((1, 8, 11), 32.0, dtype=np.float32)
    mic_u_m[:, :4, :] = 8.0

    aggregates = aggregate_predictions(mic_u_m)

    assert isinstance(aggregates, ApexAggregates)
    np.testing.assert_array_equal(
        aggregates.official_pathogen_mean_mic_u_m,
        np.full((1, 11), 20.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        aggregates.pathogen_success16,
        np.full((1, 11), 0.5, dtype=np.float32),
    )
    assert aggregates.official_broad_mean_mic_u_m[0] == pytest.approx(20.0)
    assert aggregates.vote16[0] == pytest.approx(0.5)
    assert aggregates.median_log2_mic[0] == pytest.approx(np.log2(20.0))
    assert aggregates.q90_log2_mic[0] == pytest.approx(np.log2(20.0))
    assert aggregates.worst3_log2_mic[0] == pytest.approx(np.log2(20.0))
    assert aggregates.model_disagreement_mad_log2[0] == pytest.approx(1.0)


@pytest.mark.parametrize("shape", [(1, 7, 11), (1, 8, 10)])
def test_apex_tensor_requires_exact_model_and_pathogen_axes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"\[sequence, 8 models, 11 pathogens\]"):
        aggregate_predictions(np.ones(shape, dtype=np.float32))


def test_prediction_archive_is_byte_deterministic_and_preserves_names(tmp_path: Path) -> None:
    sequences = ["ACDEFGHIK", "KWKWKWKWK"]
    model_names = [f"model_{index}" for index in range(8)]
    tensor = np.arange(2 * 8 * 11, dtype=np.float32).reshape(2, 8, 11) + 1
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    write_prediction_archive(first, sequences, model_names, APEX_PATHOGENS, tensor)
    write_prediction_archive(second, sequences, model_names, APEX_PATHOGENS, tensor)
    loaded = load_prediction_archive(first)

    assert _sha256(first) == _sha256(second)
    assert loaded.sequences == tuple(sequences)
    assert loaded.model_names == tuple(model_names)
    assert loaded.pathogens == APEX_PATHOGENS
    np.testing.assert_array_equal(loaded.mic_u_m, tensor)


def test_canonical_pathogen_order_is_fixed() -> None:
    assert len(APEX_PATHOGENS) == 11
    assert APEX_PATHOGENS[0] == "A. baumannii ATCC 19606"
    assert APEX_PATHOGENS[-1] == "vancomycin-resistant E. faecium ATCC 700221"


def test_json_manifest_writer_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "apex_manifest.json"

    write_json_manifest(path, {"schema_version": 1})

    assert json.loads(path.read_text()) == {"schema_version": 1}
