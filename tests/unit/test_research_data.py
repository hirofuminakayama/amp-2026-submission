import pytest

from robust_apex_qd.research.data import (
    activity_label,
    conservative_identity,
    grouped_split,
    normalize_battle,
    normalize_qmap,
    parse_mic,
)
from robust_apex_qd.research.metadata import publication_keys


@pytest.mark.parametrize(
    ("text", "relation", "value"),
    [(" > 16 ", ">", 16), ("≤8", "<=", 8), ("2.5e1", "=", 25)],
)
def test_parse_preserves_censoring(text: str, relation: str, value: float) -> None:
    assert parse_mic(text) == (relation, value)


@pytest.mark.parametrize("text", ["", "nan", "inf", "0", "-1", "2-8", "~16"])
def test_ambiguous_or_invalid_mic_is_not_an_exact_measurement(text: str) -> None:
    assert parse_mic(text) == (None, None)


@pytest.mark.parametrize(
    ("relation", "value", "expected"),
    [
        ("<", 16, 1),
        ("<=", 16, 1),
        (">", 16, 0),
        (">=", 16, None),
        (">", 8, None),
        ("<", 32, None),
        ("=", 16, 1),
        ("=", 32, 0),
    ],
)
def test_activity_label_requires_logically_determined_class(
    relation: str, value: float, expected: int | None
) -> None:
    assert activity_label(relation, value) == expected


def test_groups_keep_transitive_homology_and_history_out_of_holdout() -> None:
    sequences = ["AAAAACCCCC", "CCCCCGGGGG", "GGGGGTTTTT", "WWWWWWWWWW"]
    groups, splits = grouped_split(sequences, {sequences[0]}, threshold=0.5, seed=42)
    assert len({groups[s] for s in sequences[:3]}) == 1
    assert {splits[s] for s in sequences[:3]} == {"historical"}
    assert grouped_split(list(reversed(sequences)), {sequences[0]}, 0.5, 42) == (groups, splits)


def test_conservative_identity_penalizes_short_fragments() -> None:
    assert conservative_identity("AAAA", "AAAACCCC") == 0.5
    assert conservative_identity("AAAA", "TTTT") == 0


def test_consensus_never_becomes_an_exact_or_binary_measurement() -> None:
    row = {
        "id": 1,
        "sequence": "ACDEFGHIK",
        "nterminal": None,
        "cterminal": None,
        "bonds": [],
        "targets": {"Escherichia coli": [8, 8, 8]},
    }
    observation = normalize_qmap(row)[0]
    assert observation.consensus_um == 8
    assert observation.active16 is None
    assert not observation.exact_mic


def test_mass_conversion_requires_confirmed_chemical_form() -> None:
    raw = {
        "id": "1",
        "concentration": ">16",
        "unit": "µg/ml",
        "targetSpecies": "Escherichia coli ATCC 25922",
        "activity": "999",
    }
    sequence = {"sequence": "ACDEFGHIK", "nTerminus": "", "cTerminus": ""}
    unknown = normalize_battle(raw, sequence, None, 0)
    assert unknown.mic_um is None
    assert "unverified_chemical_form_for_mass_conversion" in unknown.exclusion_reasons
    chemistry = {
        "id": 1,
        "sequence": "ACDEFGHIK",
        "nterminal": None,
        "cterminal": None,
        "bonds": [],
    }
    known = normalize_battle(raw, sequence, chemistry, 0)
    assert known.relation == ">"
    assert known.mic_um is not None and known.mic_um != 999
    assert known.apex_pathogen is None
    assert "no_exact_apex_strain_match" in known.exclusion_reasons
    assert not known.exact_mic


def test_unknown_chemistry_um_is_exploratory_and_modified_is_separate() -> None:
    raw = {
        "id": "1",
        "concentration": "8",
        "unit": "µM",
        "targetSpecies": "Acinetobacter baumannii ATCC 19606",
    }
    sequence = {"sequence": "ACDEFGHIK", "nTerminus": "", "cTerminus": ""}
    unknown = normalize_battle(raw, sequence, None, 0)
    assert unknown.active16 == 1
    assert not unknown.primary_eligible
    assert unknown.apex_pathogen == "A. baumannii ATCC 19606"
    chemistry = {
        "id": 1,
        "sequence": "ACDEFGHIK",
        "nterminal": None,
        "cterminal": "AMD",
        "bonds": [],
    }
    modified = normalize_battle(raw, sequence, chemistry, 0)
    assert not modified.primary_eligible


def test_study_identity_uses_publication_identifiers_not_database_row_ids() -> None:
    first = {"id": 1, "title": "First spelling", "pubmed": {"pubmedId": "123"}}
    second = {"id": 99, "title": "Other spelling", "pubmed": {"pubmedId": "123"}}
    assert publication_keys([first]) == publication_keys([second]) == {"pubmed:123"}
    assert publication_keys([{"id": 1}]) == set()


def test_study_links_merge_dissimilar_sequences_before_partitioning() -> None:
    sequences = ["AAAAAAAAAA", "WWWWWWWWWW", "CCCCCCCCCC"]
    groups, splits = grouped_split(sequences, {sequences[0]}, linked_sequences=[sequences[:2]])
    assert groups[sequences[0]] == groups[sequences[1]]
    assert splits[sequences[1]] == "historical"
