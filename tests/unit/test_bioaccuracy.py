import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from robust_apex_qd.research.bioaccuracy import (
    EndpointObservation,
    chemistry_key,
    dbaasp_hemolysis,
    joint_hit_bounds,
    molecular_labels,
    peptide_metrics,
    qmap_hc50,
)
from robust_apex_qd.research.biofeatures import feature_contract, research_features


def observation(identifier: str, value: float, **extra: object) -> EndpointObservation:
    return EndpointObservation.model_validate(
        dict(
            observation_id=identifier,
            source="fixture",
            source_id="1",
            sequence="ACDEFGHIK",
            endpoint="measured_mic",
            target="species-a",
            species="species-a",
            value_um=value,
            relation="=",
            **extra,
        )
    )


def test_repeated_assay_does_not_change_molecular_precision() -> None:
    a = observation("a", 8)
    b = observation("b", 64).model_copy(update={"sequence": "KLMNPQRST"})
    original = molecular_labels([a, b])
    repeated = molecular_labels([a, b, a.model_copy(update={"observation_id": "a-again"})])
    pd.testing.assert_frame_equal(original, repeated)
    scores = pd.DataFrame(
        {"molecule_id": original.molecule_id, "species": original.species, "prediction": [0, 1]}
    )
    assert peptide_metrics(original, scores, ks=(1, 10)) == peptide_metrics(
        repeated, scores, ks=(1, 10)
    )
    result = peptide_metrics(original, scores, ks=(1, 10))
    assert result[0]["precision_lower"] == 1
    assert result[1]["precision_lower"] is None
    assert result[1]["available"] == 2


def test_conflicting_assays_keep_uncertainty_and_unknown_is_not_inactive() -> None:
    rows = molecular_labels([observation("a", 8), observation("b", 64, medium="other")])
    assert rows.iloc[0].hit_lower == 0
    assert rows.iloc[0].hit_upper == 1
    assert rows.iloc[0].assays == 2
    missing = EndpointObservation.model_validate(
        dict(observation("a", 8).model_dump(), value_um=None, relation=None)
    )
    assert molecular_labels([missing]).empty


def test_unknown_chemistry_does_not_join_known_forms() -> None:
    a = observation("a", 8)
    free = a.model_copy(update={"chemistry_support": {"nterminal": "reported_free"}})
    amidated = a.model_copy(update={"cterminal": "AMIDATION"})
    assert len({chemistry_key(x) for x in [a, free, amidated]}) == 3


def test_joint_hit_respects_censoring_and_requires_molecular_match() -> None:
    mic = observation("mic", 16)
    hc = mic.model_copy(
        update={
            "observation_id": "hc",
            "endpoint": "measured_hc50",
            "value_um": 128,
            "relation": ">",
            "target": "erythrocytes",
        }
    )
    assert joint_hit_bounds(mic, hc, ratio=8) == (1, 1)
    assert joint_hit_bounds(mic, hc, ratio=16) == (0, 1)
    assert joint_hit_bounds(mic, hc.model_copy(update={"cterminal": "amide"}), ratio=8) is None
    percent = hc.model_copy(update={"endpoint": "hemolysis_percent"})
    with pytest.raises(ValueError, match="HC50"):
        joint_hit_bounds(mic, percent)


def test_qmap_hc50_is_consensus_not_measured_or_a_censor_interval() -> None:
    raw = dict(
        id=1,
        sequence="ACDEFGHIK",
        nterminal=None,
        cterminal=None,
        bonds=[],
        hemolytic_hc50=[16, 128, 72],
    )
    result = qmap_hc50(raw)
    assert result is not None
    assert result.endpoint == "consensus_hc50"
    assert result.value_um == 72
    assert result.relation == "="
    assert result.rbc_species is None
    assert result.chemistry_support["stereochemistry"] == "unknown"
    with pytest.raises(ValidationError):
        observation("a", -1)


def test_versioned_boman_and_local_features_preserve_legacy() -> None:
    legacy = research_features("ACDEFGHIK", boman="legacy")
    standard = research_features("ACDEFGHIK", boman="standard")
    assert legacy["legacy_solubility_mean"] == pytest.approx(-0.3711111111)
    assert standard["boman_standard"] == pytest.approx(1.5344444444)
    assert "boman_index" not in standard
    extended = research_features("ACDEFGHIK", boman="both", local=True, interactions=True)
    assert all(np.isfinite(list(extended.values())))
    assert extended["moment_100_window12_max"] == pytest.approx(extended["moment_100_full"])
    assert feature_contract("standard") != feature_contract("legacy")


def test_missing_predictions_do_not_shrink_available_cohort_silently() -> None:
    labels = molecular_labels(
        [observation("a", 8), observation("b", 64).model_copy(update={"sequence": "KLMNPQRST"})]
    )
    scores = labels[["molecule_id", "species"]].assign(prediction=[np.nan, 1])
    result = peptide_metrics(labels, scores, ks=(1,))[0]
    assert result["available"] == 2
    assert result["predicted"] == 1
    assert result["coverage"] == 0.5


def test_strict_mic_boundary_and_missing_joint_endpoint() -> None:
    mic = observation("mic", 16).model_copy(update={"relation": ">="})
    rows = molecular_labels([mic])
    assert (rows.iloc[0].hit_lower, rows.iloc[0].hit_upper) == (0, 1)
    absent = mic.model_copy(update={"endpoint": "measured_hc50", "value_um": None})
    assert joint_hit_bounds(mic, absent) is None


def test_raw_hemolysis_percent_is_preserved_without_becoming_hc50() -> None:
    peptide = dict(id=1, sequence="ACDEFGHIK")
    assay = dict(
        id=2,
        targetCell=dict(name="Human erythrocytes"),
        activityMeasureForLysisValue="10% Hemolysis",
        concentration="50",
        unit=dict(name="µM"),
    )
    row, reason = dbaasp_hemolysis(peptide, assay, peptide)
    assert reason == "included" and row is not None
    assert row.endpoint == "hemolysis_percent"
    assert row.hemolysis_percent == 10 and row.test_concentration_um == 50
    assert row.value_um is None
    assay["activityMeasureForLysisValue"] = "50% Hemolysis"
    row, reason = dbaasp_hemolysis(peptide, assay, peptide)
    assert row is not None and row.endpoint == "measured_hc50" and row.value_um == 50
    assay["activityMeasureForLysisValue"] = "50-60% Hemolysis"
    assert dbaasp_hemolysis(peptide, assay, peptide)[0] is None
