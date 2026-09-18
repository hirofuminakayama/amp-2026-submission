from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.research.mic_lineage import (
    MeasurementLineage,
    duplicate_links,
    official_training_mask,
    publication_groups,
    reference_articles,
    retained_training,
)


def test_references_are_one_based_and_ambiguous_tokens_fail_closed() -> None:
    articles = [{"pubmed": {"pubmedId": "123"}}, {"pubmed": {"pubmedId": "456"}}]
    assert reference_articles("2", articles) == ["pubmed:456"]
    assert reference_articles("1, 2", articles) == ["pubmed:123", "pubmed:456"]
    for token in [None, "", "0", "3", "1-2", "unknown"]:
        assert reference_articles(token, articles) == []


def test_lineage_requires_evidence_and_preserves_unknowns() -> None:
    row = MeasurementLineage(observation_id="a", source_record="export.csv:0")
    assert row.study_ids == [] and row.duplicate_of is None
    assert row.model_dump()["schema_version"] == 2
    with pytest.raises(ValueError):
        MeasurementLineage(observation_id="a", source_record="export.csv:0", study_ids=["p"])


def test_duplicate_ids_are_not_inferred_from_equal_values() -> None:
    rows = pd.DataFrame(dict(observation_id=["a", "b", "c"], export_key=["x"] * 3))
    links = duplicate_links(rows, {"a": [1], "b": [1], "c": [2]})
    assert links["b"] == ("same_source_assay", "a")
    assert links["c"] == ("distinct_assay_record", None)
    assert duplicate_links(rows, {})["b"] == ("unresolved_identical_export", None)


def test_publication_isolation_is_transitive_and_unknowns_stay_separate() -> None:
    groups = publication_groups(
        ["a", "b", "c", "d", "e"], {"a": ["p"], "b": ["p", "q"], "c": ["q"]}
    )
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["e"] and groups["d"] != groups["a"]


def test_auxiliary_filter_checks_every_heldout_sequence_and_boundary() -> None:
    matrix = np.array([[1, 0.6, 0.7], [0.6, 1, 0.2], [0.7, 0.2, 1]], dtype=np.float32)
    assert retained_training([0, 1, 2], [0], matrix, 0.6) == [1]
    with pytest.raises(ValueError):
        retained_training([1], [], matrix, 0.6)


def test_official_benchmark_excludes_boundary_and_all_auxiliary_duplicates() -> None:
    maxima = np.array([0.2, 0.6, 0.599, 0.8], dtype=np.float32)
    mask = official_training_mask(["test", "boundary", "safe", "close"], {"test"}, maxima, 0.6)
    assert mask.tolist() == [False, False, True, False]
    with pytest.raises(ValueError):
        official_training_mask(["a"], {"b"}, np.array([np.nan]), 0.6)


def test_diagnostic_fit_ignores_validation_labels_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    fit = importlib.import_module("prepare_mic_evaluation").fit_diagnostic
    rows = pd.DataFrame(
        dict(
            observation_id=["a", "b", "c"],
            sequence=["A", "B", "C"],
            species=["s"] * 3,
            sequence_index=[0, 1, 2],
            mic_um=[1.0, 4.0, 2.0],
        )
    )
    x = np.array([[0.0], [1.0], [100.0]])
    train, valid = np.array([True, True, False]), np.array([False, False, True])
    first = fit(rows, x, train, valid, "mic_um", tmp_path / "first")
    rows.loc[2, "mic_um"] = 1e6
    second = fit(rows, x, train, valid, "mic_um", tmp_path / "second")
    np.testing.assert_equal(first.prediction.to_numpy(), second.prediction.to_numpy())
    np.testing.assert_equal(first.median_prediction.to_numpy(), second.median_prediction.to_numpy())
    weights = np.load(tmp_path / "first/species-0.npz")
    assert weights["mean"].item() == 0.5
    assert (tmp_path / "first/manifest.json").is_file()
    with pytest.raises(ValueError, match="overlap"):
        fit(rows, x, train, train, "mic_um", tmp_path / "bad")


@pytest.mark.parametrize(
    "raw,relation,lower,upper",
    [("8", "=", 8.0, 8.0), (">64", ">", 64.0, None), ("≤1", "<=", None, 1.0)],
)
def test_lineaged_observation_retains_units_bounds_and_missing_metadata(
    raw: str, relation: str, lower: float | None, upper: float | None
) -> None:
    from robust_apex_qd.research.mic_lineage import LineagedMICObservation

    payload = dict(
        observation_id="a",
        source="battleamp",
        source_id=1,
        sequence="AAAA",
        target="s",
        species="s",
        target_level="species",
        chemical_form="unknown",
        chemical_evidence="missing",
        raw_value=raw,
        raw_unit="µM",
        relation=relation,
        mic_um=lower or upper,
        exact_mic=relation == "=",
        exclusion_reasons=[],
        missing_fields=["study"],
        objective="measured_mic",
        lower_um=lower,
        upper_um=upper,
        conversion_evidence="source_uM",
        legacy_included=False,
        screen_included=False,
        lineage=dict(observation_id="a", source_record="raw:1"),
    )
    row = LineagedMICObservation.model_validate(payload)
    assert row.model_validate_json(row.model_dump_json()) == row
    assert row.raw_value == raw and row.raw_unit == "µM" and row.study is None
    with pytest.raises(ValueError):
        LineagedMICObservation.model_validate({**payload, "lower_um": float("inf")})
    with pytest.raises(ValueError):
        LineagedMICObservation.model_validate({**payload, "objective": "qmap_consensus"})
    with pytest.raises(ValueError):
        LineagedMICObservation.model_validate({**payload, "lower_um": 0.1, "upper_um": 1000.0})
