import json
import subprocess
import sys
from pathlib import Path

import Levenshtein
import pytest

from robust_apex_qd.io.fasta import FastaFormatError, FastaRecord, read_fasta, write_fasta
from robust_apex_qd.validation.compliance import (
    OFFICIAL_CHALLENGE_SIMILARITY_MAX,
    RejectionReason,
    validate_records,
    validate_submission,
)
from robust_apex_qd.validation.similarity import local_similarity

ROOT = Path(__file__).resolve().parents[2]
INVALID_FIXTURES = ROOT / "tests/fixtures/invalid_submission"


def test_fasta_round_trip_preserves_header_sequence_pairs(tmp_path: Path) -> None:
    records = [FastaRecord("first item", "ACDEFGHI"), FastaRecord("second", "KLMNPQRST")]
    path = tmp_path / "records.fasta"
    write_fasta(records, path)
    assert read_fasta(path) == records


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("ACDEFGHI\n", "before the first header"),
        (">header\n", "empty sequence"),
        (">\nACDEFGHI\n", "empty header"),
    ],
)
def test_fasta_rejects_malformed_input(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.fasta"
    path.write_text(text)
    with pytest.raises(FastaFormatError, match=message):
        read_fasta(path)


@pytest.mark.parametrize(
    ("sequence", "reason"),
    [
        ("ACDEFGH", RejectionReason.LENGTH_OUT_OF_RANGE),
        ("ACDEFGHI", None),
        ("A" * 50, None),
        ("A" * 51, RejectionReason.LENGTH_OUT_OF_RANGE),
        ("ACDEFGHX", RejectionReason.INVALID_ALPHABET),
    ],
)
def test_canonical_alphabet_and_length_boundaries(
    sequence: str, reason: RejectionReason | None
) -> None:
    report = validate_records([FastaRecord("candidate", sequence)])
    assert (reason in report.reason_counts) if reason is not None else report.is_valid


def test_duplicate_and_exact_reference_overlap_have_distinct_reasons() -> None:
    sequence = "ACDEFGHI"
    report = validate_records(
        [FastaRecord("one", sequence), FastaRecord("two", sequence)],
        references={sequence},
    )
    assert report.reason_counts[RejectionReason.DUPLICATE_SEQUENCE] == 1
    assert report.reason_counts[RejectionReason.EXACT_REFERENCE_OVERLAP] == 2


def test_levenshtein_official_and_selection_thresholds_are_separate() -> None:
    reference = "A" * 10
    exact_boundary = "A" * 8 + "CC"
    above_boundary = "A" * 9 + "C"
    selection_boundary = "A" * 39 + "C" * 11
    assert Levenshtein.ratio(exact_boundary, reference) == pytest.approx(0.80)
    assert Levenshtein.ratio(above_boundary, reference) > OFFICIAL_CHALLENGE_SIMILARITY_MAX
    assert Levenshtein.ratio(selection_boundary, "A" * 50) == pytest.approx(0.78)

    at_boundary = validate_records(
        [FastaRecord("candidate", exact_boundary)], references={reference}, check_similarity=True
    )
    above = validate_records(
        [FastaRecord("candidate", above_boundary)], references={reference}, check_similarity=True
    )
    assert at_boundary.is_valid
    assert above.reason_counts[RejectionReason.CHALLENGE_SIMILARITY_EXCEEDED] == 1


def test_smith_waterman_matches_starter_kit_operationalization() -> None:
    assert local_similarity("ACDEFGHI", "ACDEFGHI") == pytest.approx(1.0)
    assert local_similarity("ACDEFGHI", "XXXXACDE") == pytest.approx(0.5)
    assert local_similarity("AAAAAAAA", "CCCCCCCC") == pytest.approx(0.0)


def test_submission_contract_and_order(tmp_path: Path) -> None:
    output = tmp_path / "submission"
    output.mkdir()
    library = [FastaRecord("seq1", "ACDEFGHI"), FastaRecord("seq2", "KLMNPQRS")]
    write_fasta(library, output / "library.fasta")
    write_fasta([library[1]], output / "top.fasta")
    (output / "ranking.tsv").write_text(
        "rank\tcandidate_id\tsequence\tfinal_score\n1\tcand_000002\tKLMNPQRS\t1.000000\n"
    )
    (output / "manifest.json").write_text(
        '{"schema_version": 1, "library_count": 2, "top_count": 1, "manual_intervention": false}\n'
    )
    report = validate_submission(output, references=set(), library_size=2, top_k=1)
    assert report.is_valid


@pytest.mark.parametrize(
    ("name", "references", "check_similarity", "reason"),
    [
        ("alphabet.fasta", set(), False, RejectionReason.INVALID_ALPHABET),
        ("length.fasta", set(), False, RejectionReason.LENGTH_OUT_OF_RANGE),
        ("duplicate.fasta", set(), False, RejectionReason.DUPLICATE_SEQUENCE),
        (
            "overlap.fasta",
            {"VNWKKILGKIIKVVK"},
            False,
            RejectionReason.EXACT_REFERENCE_OVERLAP,
        ),
        (
            "similarity_above_080.fasta",
            {"VNWKKILGKIIKVVK"},
            True,
            RejectionReason.CHALLENGE_SIMILARITY_EXCEEDED,
        ),
    ],
)
def test_invalid_fixtures_have_specific_rejection_reasons(
    name: str,
    references: set[str],
    check_similarity: bool,
    reason: RejectionReason,
) -> None:
    report = validate_records(
        read_fasta(INVALID_FIXTURES / name),
        references=references,
        check_similarity=check_similarity,
    )
    assert report.reason_counts[reason] >= 1


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("alphabet.fasta", RejectionReason.INVALID_ALPHABET),
        ("length.fasta", RejectionReason.LENGTH_OUT_OF_RANGE),
        ("duplicate.fasta", RejectionReason.DUPLICATE_SEQUENCE),
        ("overlap.fasta", RejectionReason.EXACT_REFERENCE_OVERLAP),
        ("similarity_above_080.fasta", RejectionReason.CHALLENGE_SIMILARITY_EXCEEDED),
    ],
)
def test_invalid_fixture_cli_exits_nonzero_with_specific_reason(
    tmp_path: Path, name: str, reason: RejectionReason
) -> None:
    output = tmp_path / name.removesuffix(".fasta")
    output.mkdir()
    records = read_fasta(INVALID_FIXTURES / name)
    write_fasta(records, output / "library.fasta")
    write_fasta(records, output / "top.fasta")
    ranking_lines = ["rank\tcandidate_id\tsequence\tfinal_score"]
    ranking_lines.extend(
        f"{index}\tcand_{index:06d}\t{record.sequence}\t1.000000"
        for index, record in enumerate(records, start=1)
    )
    (output / "ranking.tsv").write_text("\n".join(ranking_lines) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "library_count": len(records),
                "top_count": len(records),
                "manual_intervention": False,
            }
        )
        + "\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robust_apex_qd.cli",
            "verify-local",
            "--output-dir",
            str(output),
            "--library-size",
            str(len(records)),
            "--top-k",
            str(len(records)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert reason.value in result.stderr
