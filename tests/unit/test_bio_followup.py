from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.research.bio_followup import (
    ChemistryEvidence,
    apply_chemistry_evidence,
    chemistry_status,
    corrected_observations,
    joint_development_split,
    primary_joint_coverage,
    refit_mic_rows,
)
from robust_apex_qd.research.bioaccuracy import EndpointObservation


def observation(identifier: str = "mic:1") -> EndpointObservation:
    return EndpointObservation(
        observation_id=identifier,
        source="test",
        source_id="1",
        sequence="AAAAAAAA",
        endpoint="measured_mic",
        target="E",
        species="E",
        relation=">",
        value_um=16,
        chemistry_support={
            "nterminal": "reported_free",
            "cterminal": "reported_free",
            "bonds": "reported_free",
            "stereochemistry": "unknown",
        },
    )


def test_free_termini_and_uppercase_sequence_do_not_establish_l_stereochemistry() -> None:
    row = observation()
    assert chemistry_status(row) == "unknown"
    known = row.model_copy(
        update={
            "stereochemistry": "L",
            "chemistry_support": {**row.chemistry_support, "stereochemistry": "reported_L"},
        }
    )
    assert chemistry_status(known) == "known_linear_free_L"
    modified = row.model_copy(update={"cterminal": "NH2"})
    assert chemistry_status(modified) == "known_modified"
    assert chemistry_status(row.model_copy(update={"stereochemistry": "unknown"})) == "unknown"
    assert chemistry_status(row.model_copy(update={"stereochemistry": "D"})) == "known_modified"


def test_corrections_are_id_scoped_and_preserve_other_endpoints_and_censoring() -> None:
    mic = observation()
    hc = mic.model_copy(update={"observation_id": "hc:1", "endpoint": "measured_hc50"})
    retained, audit = corrected_observations([mic, hc], ["mic:1", "absent"])
    assert retained == [hc]
    assert retained[0].relation == ">"
    assert {r["observation_id"]: r["matches"] for r in audit} == {"absent": 0, "mic:1": 1}
    assert mic.chemistry_support["cterminal"] == "reported_free"
    with pytest.raises(ValueError, match="Duplicate"):
        corrected_observations([mic], ["mic:1", "mic:1"])


def test_paper_and_homology_bridges_share_outer_and_inner_folds() -> None:
    sequences = list("abcdefghijklmno")
    identity = np.eye(len(sequences))
    identity[0, 1] = identity[1, 0] = 0.7
    papers = {"b": ["paper:1"], "c": ["paper:1"]}
    split = joint_development_split(sequences, identity, papers)
    assert split["groups"]["a"] == split["groups"]["c"]
    assert set(split["outer"].values()) == set(range(5))
    for fold, inner in split["inner"].items():
        assert all(split["outer"][s] != int(fold) for s in inner)
        if "a" in inner:
            assert inner["a"] == inner["b"] == inner["c"]
    changed = identity.copy()
    changed[0, 1] = np.nan
    with pytest.raises(ValueError, match="identity"):
        joint_development_split(sequences, changed, papers)


def test_insufficient_components_are_not_presented_as_five_folds() -> None:
    split = joint_development_split(list("abc"), np.eye(3), {})
    assert not split["folds_ready"]


def test_chemistry_evidence_is_observation_scoped_and_hash_checked(tmp_path: Path) -> None:
    import hashlib

    from robust_apex_qd.features.embeddings import file_sha256

    source = tmp_path / "paper.xml"
    source.write_text("reviewed source")
    mic = observation()
    hc = mic.model_copy(update={"observation_id": "hc:1", "endpoint": "measured_hc50"})
    record = ChemistryEvidence(
        observation_id=mic.observation_id,
        observation_sha256=hashlib.sha256(mic.model_dump_json().encode()).hexdigest(),
        sources_sha256={str(source): file_sha256(source)},
        paper_id="pubmed:1",
        locator="Table 1",
        rationale="Table distinguishes the L sequence from the separately labelled D analogue",
        stereochemistry="L",
        chemistry_support={**mic.chemistry_support, "stereochemistry": "reported_L"},
    )
    updated = apply_chemistry_evidence([mic, hc], [record])
    assert chemistry_status(updated[0]) == "known_linear_free_L"
    assert chemistry_status(updated[1]) == "unknown"
    assert updated[0].relation == ">"
    assert mic.stereochemistry is None
    with pytest.raises(ValueError, match="observation hash"):
        apply_chemistry_evidence([mic.model_copy(update={"value_um": 32})], [record])
    source.write_text("changed source")
    with pytest.raises(ValueError, match="source hash"):
        apply_chemistry_evidence([mic], [record])


def test_primary_pairs_in_one_component_do_not_support_nested_evaluation() -> None:
    mic = observation().model_copy(
        update={
            "stereochemistry": "L",
            "chemistry_support": {
                **observation().chemistry_support,
                "stereochemistry": "reported_L",
            },
        }
    )
    hc = mic.model_copy(
        update={
            "observation_id": "hc:1",
            "endpoint": "measured_hc50",
            "rbc_species": "human",
            "value_um": 256,
        }
    )
    split = {"groups": {mic.sequence: "g"}, "outer": {mic.sequence: 0}, "inner": {}}
    report, pairs = primary_joint_coverage([mic, hc], split)
    assert len(pairs) == 1
    assert report["paired_components"] == 1
    assert not report["nested_supported"]
    assert primary_joint_coverage([mic], split)[0]["primary_joint_pairs"] == 0


def test_refit_rows_exclude_corrected_and_modified_ids_without_changing_feature_index() -> None:
    import pandas as pd

    mic = observation()
    modified = mic.model_copy(update={"observation_id": "mic:2", "cterminal": "NH2"})
    raw = pd.DataFrame(
        [
            dict(
                observation_id=name,
                sequence=mic.sequence,
                sequence_index=19,
                objective="measured_mic",
            )
            for name in ["mic:1", "mic:2", "excluded:3"]
        ]
    )
    split = {"groups": {mic.sequence: "new-group"}, "outer": {mic.sequence: 3}}
    rows = refit_mic_rows(raw, [mic, modified], split)
    assert rows.observation_id.tolist() == ["mic:1"]
    assert rows.sequence_index.tolist() == [19]
    assert rows.homology_group.tolist() == ["new-group"]
    assert rows.homology_fold.tolist() == [3]
    raw.loc[0, "sequence"] = "CCCCCCCC"
    with pytest.raises(ValueError, match="sequence"):
        refit_mic_rows(raw, [mic, modified], split)
