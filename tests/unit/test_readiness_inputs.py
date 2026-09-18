from pathlib import Path

import pytest

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256


def test_output_cannot_overlap_input_or_previous_output(tmp_path: Path) -> None:
    source = tmp_path / "saved"
    source.mkdir()
    (source / "input.txt").write_text("original")
    for output in (source, source / "new", tmp_path):
        with pytest.raises(ValueError):
            fresh_output(output, [source])
    output = tmp_path / "new"
    fresh_output(output, [source])
    with pytest.raises(ValueError):
        fresh_output(output, [source])
    assert (source / "input.txt").read_text() == "original"


def test_hash_validation_refuses_changed_source(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_text("saved")
    expected = {str(path): file_sha256(path)}
    assert verify_hashes(expected) == expected
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        verify_hashes(expected)
