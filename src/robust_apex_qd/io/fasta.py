from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


class FastaFormatError(ValueError):
    """Raised when a FASTA file cannot be represented as header/sequence records."""


def read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                if not sequence_parts:
                    raise FastaFormatError(f"Record '{header}' has an empty sequence")
                records.append(FastaRecord(header, "".join(sequence_parts).upper()))
            header = line[1:].strip()
            if not header:
                raise FastaFormatError(f"Line {line_number} has an empty header")
            sequence_parts = []
            continue
        if header is None:
            raise FastaFormatError(
                f"Line {line_number} contains sequence data before the first header"
            )
        sequence_parts.append(line)

    if header is not None:
        if not sequence_parts:
            raise FastaFormatError(f"Record '{header}' has an empty sequence")
        records.append(FastaRecord(header, "".join(sequence_parts).upper()))
    if not records:
        raise FastaFormatError(f"FASTA file is empty: {path}")
    return records


def read_fasta_sequences(path: Path) -> list[str]:
    return [record.sequence for record in read_fasta(path)]


def write_fasta(records: Iterable[FastaRecord], path: Path) -> None:
    materialized = list(records)
    if not materialized:
        raise FastaFormatError("Cannot write an empty FASTA file")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in materialized:
        if not record.header.strip():
            raise FastaFormatError("Cannot write an empty FASTA header")
        if not record.sequence:
            raise FastaFormatError(f"Cannot write an empty sequence for '{record.header}'")
        lines.extend((f">{record.header}", record.sequence.upper()))
    path.write_text("\n".join(lines) + "\n")
