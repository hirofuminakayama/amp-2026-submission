import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from robust_apex_qd.research.paper_mic import (
    PaperObservation,
    compare_existing,
    mic_bounds,
    table_grid,
)


def test_bounds_preserve_inequalities_and_mass_conversion() -> None:
    assert mic_bounds(">64", "uM", None) == (">", 64.0, None)
    assert mic_bounds("≤1", "µM", None) == ("<=", None, 1.0)
    assert mic_bounds("2\u20138", "uM", None) == ("interval", 2.0, 8.0)
    assert mic_bounds("8", "µg/mL", 2000) == ("=", 4.0, 4.0)
    for raw, unit, mass in [
        ("8", "ug/mL", None),
        ("8±2", "uM", None),
        ("NaN", "uM", None),
        ("0", "uM", None),
        ("8", "%", None),
    ]:
        with pytest.raises(ValueError):
            mic_bounds(raw, unit, mass)


def test_table_grid_expands_spans_without_dropping_footnotes() -> None:
    root = ET.fromstring(
        '<table><tr><th rowspan="2">strain</th><th colspan="2">MIC</th>'
        "</tr><tr><th>A</th><th>B</th></tr>"
        "<tr><td>EC</td><td>8<xref> a</xref></td><td>&gt;64</td></tr></table>"
    )
    assert table_grid(root) == [
        ["strain", "MIC", "MIC"],
        ["strain", "A", "B"],
        ["EC", "8 a", ">64"],
    ]


def observation(**updates: object) -> PaperObservation:
    return PaperObservation.model_validate(
        dict(
            observation_id="paper:example:T1:r2:c2",
            paper_id="example",
            sequence="AKKLLKKLL",
            peptide_name="P1",
            target="Escherichia coli ATCC 25922",
            species="Escherichia coli",
            raw_value="8",
            raw_unit="uM",
            chemical_form="unknown",
            source_sha256="a" * 64,
            source_file="paper.xml",
            table_id="T1",
            row=2,
            column=2,
            cell_text="8",
            sequence_evidence="Table S1 row P1",
            chemistry_evidence="not reported",
            assay_evidence="Methods 2",
            review_evidence="checked original table and footnotes",
            footnotes="a: broth microdilution",
        )
        | updates
    )


def test_paper_without_database_id_is_quarantined_if_chemistry_unknown() -> None:
    row = observation()
    assert row.bounds == ("=", 8.0, 8.0)
    assert not row.primary_eligible
    assert "source_id" not in row.model_dump()
    with pytest.raises(ValueError):
        row.trainer_record()


def test_only_verified_linear_form_enters_trainer_and_keeps_origin() -> None:
    row = observation(chemical_form="linear_free_L")
    converted = row.trainer_record()
    assert converted["source"] == "paper"
    assert converted["lower_um"] == 8
    assert converted["observation_id"] == row.observation_id
    assert converted["objective"] == "measured_mic"


def test_reject_unsupported_forms_and_nonfinite_masses() -> None:
    for kwargs in [
        dict(molecular_weight=float("inf")),
        dict(row=-1),
        dict(source_sha256="bad"),
        dict(sequence="AcK"),
    ]:
        with pytest.raises(ValueError):
            observation(**kwargs)


def test_match_is_not_duplicate_or_correction_without_assay_linkage() -> None:
    row = observation()
    old = dict(
        observation_id="db:1",
        sequence=row.sequence,
        target=row.target,
        mic_um=8.0,
        relation="=",
        lineage={"study_ids": ["example"]},
    )
    assert compare_existing(row, old)["status"] == "duplicate_candidate"
    old["mic_um"] = 16.0
    result = compare_existing(row, old)
    assert result["status"] == "unresolved"
    assert result["old_observation_id"] == "db:1"
    old["lineage"] = {"study_ids": ["other-paper"]}
    assert compare_existing(row, old)["status"] == "different_measurement_or_unlinked"


def test_docx_and_xlsx_keep_cell_addresses_and_reject_formulas(tmp_path: Path) -> None:
    from zipfile import ZipFile

    from robust_apex_qd.research.paper_mic import read_source_table

    doc = tmp_path / "source.docx"
    with ZipFile(doc, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>&gt;8</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>',
        )
    assert read_source_table(doc, "0") == [[">8"]]
    book = tmp_path / "source.xlsx"
    for formula in ["", "<f>A1*2</f>"]:
        with ZipFile(book, "w") as archive:
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetData><row r="2"><c r="C2">'
                + formula
                + "<v>8</v></c></row></sheetData></worksheet>",
            )
        if formula:
            with pytest.raises(ValueError, match="Formula"):
                read_source_table(book, "sheet1")
        else:
            assert read_source_table(book, "sheet1") == [["", "", ""], ["", "", "8"]]


def test_curator_replay_corrects_only_ledger_and_prevents_double_addition(tmp_path: Path) -> None:
    import hashlib
    import json
    import subprocess
    import sys

    source = tmp_path / "source.xml"
    source.write_text(
        '<article><table-wrap id="T1"><table><tr><td>8</td></tr></table></table-wrap></article>'
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    row = observation(
        source_file=str(source), source_sha256=digest, row=0, column=0, chemical_form="modified"
    ).model_dump()
    free = row | dict(
        observation_id="paper:free", sequence="AKKLLKKLA", chemical_form="linear_free_L"
    )
    old = dict(
        observation_id="db:1",
        sequence=row["sequence"],
        target=row["target"],
        chemical_form="reported_linear_free",
        objective="measured_mic",
        mic_um=8.0,
        relation="=",
        lineage={"study_ids": ["example"]},
    )
    prior = tmp_path / "old.jsonl"
    prior.write_text(json.dumps(old) + "\n")
    old_bytes = prior.read_bytes()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"papers": [{"paper_id": "example", "rights_status": "full_ready"}]})
    )
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            dict(
                sources_sha256={str(source): digest},
                observations=[row, free],
                corrections=[
                    dict(
                        old_observation_id="db:1",
                        field="chemical_form",
                        old_value="reported_linear_free",
                        new_value="modified",
                        paper_observation_id=row["observation_id"],
                        source_sha256=digest,
                        reason="Explicit terminal amide in source",
                    )
                ],
            )
        )
    )
    output = tmp_path / "output"
    cmd = [
        sys.executable,
        "scripts/curate_mic_papers.py",
        "--rules",
        str(rules),
        "--registry",
        str(registry),
        "--existing",
        str(prior),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True, cwd=Path(__file__).resolve().parents[2])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["corrections"] == 1 and summary["trainer_additions"] == 1
    assert summary["dispositions"]["legacy_overlap_quarantined"] == 1
    assert prior.read_bytes() == old_bytes
    source.write_text(source.read_text().replace("<td>8", "<td>16"))
    assert subprocess.run(cmd, capture_output=True).returncode != 0


def test_paired_units_cannot_drop_censoring_or_substitute_a_prediction() -> None:
    row = observation(
        cell_text=">128/64.53", raw_value=">64.53", value_representation="paired_mass_um"
    )
    assert row.bounds == (">", 64.53, None)
    with pytest.raises(ValueError, match="preserve"):
        observation(
            cell_text=">128/64.53", raw_value="64.53", value_representation="paired_mass_um"
        )
    with pytest.raises(ValueError, match="preserve"):
        observation(raw_value="4")


def test_primary_form_cannot_contradict_reported_terminal_modification() -> None:
    with pytest.raises(ValueError, match="chemical"):
        observation(chemical_form="linear_free_L", cterminal="amide")
