import csv
import gzip
import json
import math
from pathlib import Path

import pytest
import yaml

from robust_apex_qd.features.physchem import (
    FEATURE_NAMES,
    PhyschemReference,
    compute_features,
    fit_reference,
    score_features,
    write_candidate_features,
    write_reference,
)
from robust_apex_qd.io.fasta import FastaRecord
from robust_apex_qd.validation.compliance import challenge_valid_records


def test_known_peptide_features_match_fixed_values() -> None:
    features = compute_features("ACDEFGHIK")

    assert tuple(features) == FEATURE_NAMES
    assert features["length"] == 9.0
    assert features["charge_ph_7_4"] == pytest.approx(-1.3810129760)
    assert features["charge_density"] == pytest.approx(-0.1534458862)
    assert features["gravy"] == pytest.approx(-0.3222222222)
    assert features["hydrophobic_moment"] == pytest.approx(0.3235743389)
    assert features["aromaticity"] == pytest.approx(1 / 9)
    assert features["boman_index"] == pytest.approx(-0.3711111111)
    assert features["isoelectric_point"] == pytest.approx(5.3219713211)
    assert features["molecular_weight"] == pytest.approx(1019.1318)
    assert features["shannon_entropy"] == pytest.approx(math.log2(9))
    assert features["longest_homopolymer"] == 1.0
    assert features["max_aa_fraction"] == pytest.approx(1 / 9)
    assert features["cysteine_count"] == 1.0


@pytest.mark.parametrize("sequence", ["A", "AAAAAAAA", "CCCCCCCC", "KWKWKWKW"])
def test_short_and_constant_sequences_produce_only_finite_values(sequence: str) -> None:
    features = compute_features(sequence)

    assert all(math.isfinite(value) for value in features.values())


def test_reference_round_trip_and_scoring_are_finite(tmp_path: Path) -> None:
    sequences = [
        "ACDEFGHIK",
        "KWKWKWKWK",
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "FLPLLAGLAANFLPKIF",
        "GLFDIVKKVVGAFGSL",
    ]
    reference = fit_reference(sequences, reference_sha256="abc123")
    path = tmp_path / "physchem_reference.json"
    write_reference(reference, path)
    loaded = PhyschemReference.model_validate_json(path.read_text())
    payload = json.loads(path.read_text())

    assert loaded == reference
    assert payload["schema_version"] == 2
    assert payload["reference_count"] == len(sequences)
    assert payload["reference_filter"] == "canonical_20_and_length_8_to_50"
    assert payload["feature_order"] == list(FEATURE_NAMES)
    assert payload["reference_sha256"] == "abc123"
    assert all(
        set(summary) == {"median", "iqr", "q0_005", "q0_01", "q0_99", "q0_995"}
        for summary in payload["statistics"].values()
    )

    scored = score_features(compute_features("AAAAAAAA"), reference)
    assert math.isfinite(scored.physchem_ood)
    assert math.isfinite(scored.soft_penalty)
    assert scored.hard_reject


def test_reference_sequences_score_below_an_extreme_candidate() -> None:
    sequences = [
        "ACDEFGHIK",
        "KWKWKWKWK",
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "FLPLLAGLAANFLPKIF",
        "GLFDIVKKVVGAFGSL",
    ]
    reference = fit_reference(sequences, reference_sha256="abc123")
    reference_scores = [
        score_features(compute_features(sequence), reference).physchem_ood for sequence in sequences
    ]
    extreme_score = score_features(compute_features("AAAAAAAA"), reference).physchem_ood

    assert float(sorted(reference_scores)[len(reference_scores) // 2]) < extreme_score


def test_candidate_feature_writer_preserves_id_and_row_order(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv.gz"
    with gzip.open(candidates, "wt", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("candidate_id", "sequence"))
        writer.writeheader()
        writer.writerow({"candidate_id": "cand_2", "sequence": "KWKWKWKWK"})
        writer.writerow({"candidate_id": "cand_1", "sequence": "ACDEFGHIK"})
    reference = fit_reference(
        ["ACDEFGHIK", "KWKWKWKWK", "GLFDIVKKVVGAFGSL"],
        reference_sha256="abc123",
    )
    output = tmp_path / "candidate_physchem.csv.gz"

    row_count = write_candidate_features(candidates, output, reference)

    with gzip.open(output, "rt", newline="") as file:
        rows = list(csv.DictReader(file))
    assert row_count == 2
    assert [row["candidate_id"] for row in rows] == ["cand_2", "cand_1"]
    assert all(row["sequence"] for row in rows)
    numeric_names = (*FEATURE_NAMES, "physchem_ood", "soft_penalty")
    assert all(math.isfinite(float(row[name])) for row in rows for name in numeric_names)


def test_candidate_config_fixes_feature_order_and_dtype() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/candidate.yaml").read_text())
    physchem = config["features"]["physicochemical"]

    assert physchem["dtype"] == "float64"
    assert tuple(physchem["feature_order"]) == FEATURE_NAMES


def test_reference_subset_contains_only_challenge_valid_sequences() -> None:
    records = [
        FastaRecord("short", "ACDEFGH"),
        FastaRecord("minimum", "ACDEFGHI"),
        FastaRecord("noncanonical", "ACDEFGHX"),
        FastaRecord("maximum", "A" * 50),
        FastaRecord("long", "A" * 51),
    ]

    valid = challenge_valid_records(records)

    assert [record.header for record in valid] == ["minimum", "maximum"]
