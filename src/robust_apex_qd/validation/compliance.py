import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import Levenshtein

from robust_apex_qd.io.fasta import FastaFormatError, FastaRecord, read_fasta

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
OFFICIAL_CHALLENGE_SIMILARITY_MAX = 0.80
SELECTION_CHALLENGE_SIMILARITY_MAX = 0.78
REQUIRED_ARTIFACTS = ("library.fasta", "top.fasta", "ranking.tsv", "manifest.json")


class RejectionReason(str, Enum):
    MISSING_ARTIFACT = "missing_artifact"
    MALFORMED_FASTA = "malformed_fasta"
    INVALID_ALPHABET = "invalid_alphabet"
    LENGTH_OUT_OF_RANGE = "length_out_of_range"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    EXACT_REFERENCE_OVERLAP = "exact_reference_overlap"
    CHALLENGE_SIMILARITY_EXCEEDED = "challenge_similarity_exceeded"
    INCORRECT_COUNT = "incorrect_count"
    TOP_NOT_SUBSET = "top_not_subset"
    INVALID_RANKING = "invalid_ranking"
    INVALID_MANIFEST = "invalid_manifest"


@dataclass(frozen=True)
class ValidationIssue:
    reason: RejectionReason
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def reason_counts(self) -> Counter[RejectionReason]:
        return Counter(issue.reason for issue in self.issues)

    def add(self, reason: RejectionReason, message: str) -> None:
        self.issues.append(ValidationIssue(reason, message))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)


class SubmissionValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "\n".join(f"- {issue.reason.value}: {issue.message}" for issue in report.issues)
        super().__init__(f"Submission validation failed:\n{details}")


def validate_records(
    records: list[FastaRecord],
    references: set[str] | None = None,
    *,
    check_similarity: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    reference_sequences = references or set()
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        sequence = record.sequence
        invalid = sorted(set(sequence) - CANONICAL_AMINO_ACIDS)
        if invalid:
            report.add(
                RejectionReason.INVALID_ALPHABET,
                f"Record {index} ('{record.header}') has invalid characters {invalid}",
            )
        if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
            report.add(
                RejectionReason.LENGTH_OUT_OF_RANGE,
                f"Record {index} ('{record.header}') has length {len(sequence)}",
            )
        if sequence in seen:
            report.add(
                RejectionReason.DUPLICATE_SEQUENCE,
                f"Record {index} ('{record.header}') duplicates an earlier sequence",
            )
        seen.add(sequence)
        if sequence in reference_sequences:
            report.add(
                RejectionReason.EXACT_REFERENCE_OVERLAP,
                f"Record {index} ('{record.header}') exactly overlaps the challenge reference",
            )
        if check_similarity and any(
            Levenshtein.ratio(sequence, reference) > OFFICIAL_CHALLENGE_SIMILARITY_MAX
            for reference in reference_sequences
        ):
            report.add(
                RejectionReason.CHALLENGE_SIMILARITY_EXCEEDED,
                f"Record {index} ('{record.header}') exceeds the fixed 0.80 official maximum",
            )
    return report


def challenge_valid_records(records: list[FastaRecord]) -> list[FastaRecord]:
    return [
        record
        for record in records
        if MIN_LENGTH <= len(record.sequence) <= MAX_LENGTH
        and set(record.sequence) <= CANONICAL_AMINO_ACIDS
    ]


def passes_selection_similarity(
    sequence: str,
    references: set[str],
    maximum: float = SELECTION_CHALLENGE_SIMILARITY_MAX,
) -> bool:
    if maximum > OFFICIAL_CHALLENGE_SIMILARITY_MAX:
        raise ValueError("Selection similarity cannot exceed the fixed official maximum 0.80")
    return all(Levenshtein.ratio(sequence, reference) <= maximum for reference in references)


def _read_or_report(path: Path, report: ValidationReport) -> list[FastaRecord]:
    try:
        return read_fasta(path)
    except (FastaFormatError, OSError) as error:
        report.add(RejectionReason.MALFORMED_FASTA, f"{path.name}: {error}")
        return []


def _validate_ranking(path: Path, top: list[FastaRecord], report: ValidationReport) -> None:
    try:
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
    except (OSError, csv.Error) as error:
        report.add(RejectionReason.INVALID_RANKING, str(error))
        return
    required = {"rank", "candidate_id", "sequence", "final_score"}
    if not rows or not required.issubset(rows[0]):
        report.add(RejectionReason.INVALID_RANKING, f"Required columns are {sorted(required)}")
        return
    if [row["sequence"] for row in rows] != [record.sequence for record in top]:
        report.add(RejectionReason.INVALID_RANKING, "ranking.tsv order differs from top.fasta")
    if [row["rank"] for row in rows] != [str(index) for index in range(1, len(rows) + 1)]:
        report.add(RejectionReason.INVALID_RANKING, "rank must be contiguous and one-based")


def _validate_manifest(path: Path, library_size: int, top_k: int, report: ValidationReport) -> None:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        report.add(RejectionReason.INVALID_MANIFEST, str(error))
        return
    expected = {
        "schema_version": 1,
        "library_count": library_size,
        "top_count": top_k,
        "manual_intervention": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            report.add(
                RejectionReason.INVALID_MANIFEST,
                f"manifest field {key!r} must be {value!r}",
            )


def validate_submission(
    output_dir: Path,
    references: set[str],
    *,
    library_size: int = 50_000,
    top_k: int = 100,
) -> ValidationReport:
    report = ValidationReport()
    for name in REQUIRED_ARTIFACTS:
        if not (output_dir / name).is_file():
            report.add(RejectionReason.MISSING_ARTIFACT, name)
    if report.reason_counts[RejectionReason.MISSING_ARTIFACT]:
        return report

    library = _read_or_report(output_dir / "library.fasta", report)
    top = _read_or_report(output_dir / "top.fasta", report)
    if library:
        report.extend(validate_records(library, references))
    if top:
        report.extend(validate_records(top, references, check_similarity=True))
    if len(library) != library_size:
        report.add(
            RejectionReason.INCORRECT_COUNT,
            f"library.fasta has {len(library)} records; expected {library_size}",
        )
    if len(top) != top_k:
        report.add(
            RejectionReason.INCORRECT_COUNT,
            f"top.fasta has {len(top)} records; expected {top_k}",
        )
    library_sequences = {record.sequence for record in library}
    for index, record in enumerate(top, start=1):
        if record.sequence not in library_sequences:
            report.add(
                RejectionReason.TOP_NOT_SUBSET,
                f"Top record {index} is absent from library.fasta",
            )
    _validate_ranking(output_dir / "ranking.tsv", top, report)
    _validate_manifest(output_dir / "manifest.json", library_size, top_k, report)
    return report


def require_valid_submission(
    output_dir: Path,
    references: set[str],
    *,
    library_size: int = 50_000,
    top_k: int = 100,
) -> ValidationReport:
    report = validate_submission(output_dir, references, library_size=library_size, top_k=top_k)
    if not report.is_valid:
        raise SubmissionValidationError(report)
    return report
