"""Original-paper observations with explicit cell provenance and conservative reconciliation."""

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_apex_qd.research.data import activity_label, parse_mic
from robust_apex_qd.validation.compliance import MAX_LENGTH, MIN_LENGTH

COMPETITION_SPECIES = frozenset(
    {
        "Escherichia coli",
        "Staphylococcus aureus",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
        "Acinetobacter baumannii",
        "Enterococcus faecium",
        "Enterococcus faecalis",
    }
)


class PaperObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    observation_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    sequence: str = Field(pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    peptide_name: str
    target: str = Field(min_length=1)
    species: str = Field(min_length=1)
    raw_value: str
    raw_unit: str
    value_representation: Literal["literal", "paired_mass_um"] = "literal"
    chemical_form: Literal["linear_free_L", "modified", "unknown"]
    molecular_weight: float | None = Field(default=None, gt=0)
    molecular_weight_evidence: str | None = None
    nterminal: str | None = None
    cterminal: str | None = None
    bonds: str | None = None
    stereochemistry: str | None = None
    source_file: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    table_id: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    cell_text: str
    sequence_evidence: str = Field(min_length=1)
    chemistry_evidence: str = Field(min_length=1)
    assay_evidence: str = Field(min_length=1)
    review_evidence: str = Field(min_length=1)
    footnotes: str
    medium: str | None = None
    cfu: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_original_value(self) -> "PaperObservation":
        if self.chemical_form == "linear_free_L" and (
            self.nterminal not in {None, "free"}
            or self.cterminal not in {None, "free"}
            or self.bonds not in {None, "none"}
            or self.stereochemistry not in {None, "all L"}
        ):
            raise ValueError("Primary chemical form contradicts reported modification")
        expected = self.cell_text.strip()
        if self.value_representation == "paired_mass_um":
            parts = expected.split("/")
            if len(parts) != 2 or self.raw_unit not in {"uM", "µM", "μM"}:
                raise ValueError("Expected paired mass/molar MIC cell")
            relation, _ = parse_mic(parts[0])
            if relation is None:
                raise ValueError("Unresolved paired censoring")
            expected = (relation if relation != "=" else "") + parts[1]
        if self.raw_value != expected:
            raise ValueError("Registered MIC does not preserve the original cell")
        return self

    @property
    def bounds(self) -> tuple[str, float | None, float | None]:
        if self.molecular_weight is not None and not self.molecular_weight_evidence:
            raise ValueError("Mass conversion requires molecular weight evidence")
        return mic_bounds(self.raw_value, self.raw_unit, self.molecular_weight)

    @property
    def primary_eligible(self) -> bool:
        try:
            _ = self.bounds
        except ValueError:
            return False
        return (
            self.chemical_form == "linear_free_L"
            and MIN_LENGTH <= len(self.sequence) <= MAX_LENGTH
            and self.species in COMPETITION_SPECIES
        )

    def trainer_record(self) -> dict[str, Any]:
        """Return the existing scalar trainer row shape without fabricating a DB identity."""
        if not self.primary_eligible:
            raise ValueError("Only primary-eligible original observations enter training")
        relation, lower, upper = self.bounds
        value = lower if lower is not None else upper
        return dict(
            observation_id=self.observation_id,
            source="paper",
            sequence=self.sequence,
            target=self.target,
            species=self.species,
            objective="measured_mic",
            relation=relation,
            lower_um=lower,
            upper_um=upper,
            mic_um=value if relation != "interval" else None,
            exact_mic=relation == "=",
            exact_regression=relation == "=",
            active16=activity_label(relation, value),
            chemical_form="reported_linear_free",
            medium=self.medium,
            cfu=self.cfu,
            study=self.paper_id,
            assay_publication_verified=True,
            primary_eligible=True,
        )


def mic_bounds(raw: str, unit: str, mass: float | None) -> tuple[str, float | None, float | None]:
    normalized = unit.replace("μ", "u").replace("µ", "u").lower().replace(" ", "")
    if normalized == "um":
        scale = 1.0
    elif normalized in {"ug/ml", "mg/l"} and mass and math.isfinite(mass) and mass > 0:
        scale = 1000 / mass
    else:
        raise ValueError("Unresolved unit or molecular weight")
    interval = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[\u2013-]\s*(\d+(?:\.\d+)?)\s*", raw)
    if interval:
        low, high = float(interval[1]) * scale, float(interval[2]) * scale
        if not 0 < low <= high or not math.isfinite(high):
            raise ValueError("Invalid interval")
        return "interval", low, high
    relation, value = parse_mic(raw)
    if relation is None or value is None:
        raise ValueError("Ambiguous or nonpositive measured MIC")
    value *= scale
    if not math.isfinite(value):
        raise ValueError("Converted concentration is not finite")
    return (
        relation,
        None if relation in {"<", "<="} else value,
        (None if relation in {">", ">="} else value),
    )


def node_text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def table_grid(table: ET.Element) -> list[list[str]]:
    """Expand JATS spans; retain footnote markers so stripping needs a reviewed rule."""
    occupied: dict[tuple[int, int], str] = {}
    rows = table.findall(".//tr")
    for i, row in enumerate(rows):
        col = 0
        for cell in row:
            if cell.tag not in {"td", "th"}:
                continue
            while (i, col) in occupied:
                col += 1
            for di in range(int(cell.get("rowspan", "1"))):
                for dj in range(int(cell.get("colspan", "1"))):
                    if (i + di, col + dj) in occupied:
                        raise ValueError("Overlapping table spans")
                    occupied[i + di, col + dj] = node_text(cell)
            col += int(cell.get("colspan", "1"))
    width = max((j + 1 for _, j in occupied), default=0)
    return [[occupied.get((i, j), "") for j in range(width)] for i in range(len(rows))]


def compare_existing(paper: PaperObservation, old: dict[str, Any]) -> dict[str, Any]:
    """Same values suggest a duplicate; they do not prove assay identity or replication."""
    studies = old.get("lineage", {}).get("study_ids", [])
    status = "different_measurement_or_unlinked"
    reason = "same sequence/target alone does not establish a shared assay"
    if paper.paper_id in studies:
        status = "unresolved"
        reason = "paper linkage exists; chemistry, assay and value identity need review"
        try:
            relation, low, high = paper.bounds
            value = low if low is not None else high
            if (
                relation == old.get("relation")
                and old.get("mic_um") is not None
                and value is not None
                and math.isclose(value, old["mic_um"], rel_tol=0.01, abs_tol=0.005)
            ):
                status = "duplicate_candidate"
                reason = "paper/sequence/target/value agree; independent repetition not inferred"
        except ValueError:
            pass
    return dict(
        paper_observation_id=paper.observation_id,
        old_observation_id=old["observation_id"],
        status=status,
        reason=reason,
    )


def read_source_table(path: Path, table_id: str) -> list[list[str]]:
    """Read one explicitly registered original table; no execution of workbook formulas."""
    if path.suffix == ".xml":
        root = ET.parse(path).getroot()
        table = root.find(f'.//table-wrap[@id="{table_id}"]')
        if table is None:
            raise ValueError("Registered table missing")
        return table_grid(table)
    if path.suffix == ".docx":
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        table = root.findall(".//w:tbl", ns)[int(table_id)]
        # Word vertical merges retain the cell and may be intentionally empty.
        # Keep physical cell coordinates; registered rules must resolve merged labels.
        return [
            [
                "".join(t.text or "" for t in cell.findall(".//w:t", ns))
                for cell in row.findall("w:tc", ns)
            ]
            for row in table.findall("w:tr", ns)
        ]
    if path.suffix == ".xlsx":
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with ZipFile(path) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = [
                    node_text(t)
                    for t in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)
                ]
            if not re.fullmatch(r"sheet[1-9][0-9]*", table_id):
                raise ValueError("Invalid worksheet identifier")
            root = ET.fromstring(archive.read(f"xl/worksheets/{table_id}.xml"))
        cells = {}
        for row in root.findall(".//s:row", ns):
            for cell in row:
                value = cell.findtext("s:v", None, ns)
                if value is None:
                    continue
                if cell.find("s:f", ns) is not None:
                    raise ValueError("Formula-derived values are not original measurement cells")
                address = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell.get("r", ""))
                if address is None:
                    raise ValueError("Invalid spreadsheet address")
                col = 0
                for ch in address[1]:
                    col = col * 26 + ord(ch) - ord("A") + 1
                cells[int(address[2]) - 1, col - 1] = (
                    strings[int(value)] if cell.get("t") == "s" else value
                )
        height = max((i + 1 for i, _ in cells), default=0)
        width = max((j + 1 for _, j in cells), default=0)
        return [[cells.get((i, j), "") for j in range(width)] for i in range(height)]
    raise ValueError("Unsupported table format; use a separately reviewed transcription")
