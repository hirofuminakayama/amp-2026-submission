"""Audit remaining research-data gaps without changing the frozen partition or labels."""

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.SeqUtils import molecular_weight

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.provenance import match_activity, modal_mic, training_folds

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def published_panel(workbook: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(workbook) as archive:
        strings = [
            "".join(item.itertext()) for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {}
        for cell in sheet.findall(".//s:sheetData/s:row/s:c", NS):
            value = cell.find("s:v", NS)
            if value is not None and value.text is not None:
                cells[cell.attrib["r"]] = (
                    strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                )
    if cells.get("B1") != "A. baumannii ATCC 19606 (-)":
        raise ValueError("Unexpected published MIC panel")
    rows = []
    for index in range(1, 81):
        if cells.get(f"A{index + 1}") != f"Archaeasin-{index}":
            raise ValueError("Published peptide identifiers are not aligned")
        for column in "BCDEFGHIJKL":
            coordinate = f"{column}{index + 1}"
            raw = cells.get(coordinate, "")
            mic, active = modal_mic(raw)
            rows.append(
                {
                    "peptide_id": f"Archaeasin-{index}",
                    "target": cells[f"{column}1"],
                    "cell": coordinate,
                    "raw_value": raw,
                    "mic_um": mic,
                    "active16": active,
                    "aggregation": "mode_of_replicates",
                    "sequence": None,
                    "study_doi": "10.1038/s41564-025-02061-0",
                    "value_status": "reported" if mic is not None else "unreported_cell",
                    "sequence_link_status": "not_verified",
                    "adoption_eligible": False,
                }
            )
    return rows


def certificates(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text()
        sequence = re.search(r"Peptide Sequence:\s*([^\n]+)", text)
        weight = re.search(r"Molecular Weight \(Theoretical\):\s*([\d.]+)", text)
        name = re.search(r"Peptide Name:\s*([^\n]+)", text)
        seq = sequence[1].strip() if sequence else None
        canonical = seq is not None and set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
        mw = float(weight[1]) if weight else None
        expected = molecular_weight(seq, seq_type="protein") if canonical else None
        rows.append(
            {
                "certificate": path.name,
                "certificate_text_sha256": file_sha256(path),
                "sample_name": name[1].strip() if name else None,
                "sequence": seq,
                "reported_molecular_weight": mw,
                "linear_free_mass": expected,
                "mass_difference": mw - expected
                if mw is not None and expected is not None
                else None,
                "terminal_state": "not_explicitly_certified",
                "numbered_measurement_link": "not_verified",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.dataset, args.metadata, args.sources])
    metadata_manifest = json.loads((args.metadata / "metadata_manifest.json").read_text())
    metadata = {}
    for row in metadata_manifest["records"]:
        if row["status"] == "fetched":
            path = args.metadata / f"{row['id']}.json"
            if file_sha256(path) != row["sha256"]:
                raise ValueError("Metadata checksum mismatch")
            metadata[row["id"]] = json.loads(path.read_text())
    dataset_manifest = json.loads((args.dataset / "dataset_manifest.json").read_text())
    assignments = pd.read_csv(args.dataset / "sequence_splits.csv")
    if (
        file_sha256(args.dataset / "sequence_splits.csv")
        != (dataset_manifest["artifacts_sha256"]["sequence_splits.csv"])
    ):
        raise ValueError("Frozen split checksum mismatch")
    folds = training_folds(assignments.to_dict("records"))
    inner = assignments[assignments.split == "train"][["sequence", "group"]].copy()
    inner["validation_fold"] = inner.sequence.map(folds)
    if inner.groupby("group").validation_fold.nunique().max() != 1:
        raise ValueError("A group crosses inner folds")
    inner.to_csv(args.output / "development_folds.csv", index=False)
    fold_counts: Counter[int] = Counter()
    train_observations = pd.read_json(args.dataset / "observations_train.jsonl", lines=True)
    for row in train_observations[train_observations.primary_eligible].itertuples():
        fold_counts[folds[row.sequence]] += 1
    write_json(
        args.output / "development_protocol.json",
        {
            "outer_partition": "train only; development, holdout, historical are unchanged",
            "folds": 5,
            "seed": 42,
            "allocation": "largest sequence groups first, seeded hash tie-break, least-loaded fold",
            "preprocessing": "fit on the other four training folds only",
            "evaluation": "pooled OOF exploration; no independent APEX superiority claim",
            "frozen_assignment_sha256": file_sha256(args.dataset / "sequence_splits.csv"),
            "development_folds_sha256": file_sha256(args.output / "development_folds.csv"),
            "technical_validation_rows_by_fold": dict(sorted(fold_counts.items())),
        },
    )
    links = []
    for path in sorted(args.dataset.glob("observations_*.jsonl")):
        if file_sha256(path) != dataset_manifest["artifacts_sha256"][path.name]:
            raise ValueError("Dataset checksum mismatch")
        for line in path.open():
            row = json.loads(line)
            if row["source"] == "battleamp" and row["source_id"] in metadata:
                links.append(
                    {
                        "observation_id": row["observation_id"],
                        "split": row["split"],
                        "source_id": row["source_id"],
                        **match_activity(row, metadata[row["source_id"]]),
                    }
                )
    with (args.output / "assay_links.jsonl").open("w") as handle:
        for row in links:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    panel = published_panel(args.sources / "41564_2025_2061_MOESM8_ESM.xlsx")
    pd.DataFrame(panel).to_csv(args.output / "archaeasin_published_panel.csv", index=False)
    certificate_directory = args.output / "certificates"
    certificate_directory.mkdir()
    with zipfile.ZipFile(args.sources / "COA_archaeasins.zip") as archive:
        for name in archive.namelist():
            if name.startswith("__MACOSX/") or not name.lower().endswith(".pdf"):
                continue
            path = certificate_directory / Path(name).name
            with path.open("xb") as handle:
                handle.write(archive.read(name))
            subprocess.run(
                ["pdftotext", "-layout", str(path), str(path.with_suffix(".txt"))], check=True
            )
    coa = certificates(certificate_directory)
    pd.DataFrame(coa).to_csv(args.output / "archaeasin_certificates.csv", index=False)
    summary = {
        "schema_version": 1,
        "dataset_manifest_sha256": file_sha256(args.dataset / "dataset_manifest.json"),
        "assay_links_by_status": dict(Counter(row["status"] for row in links)),
        "matched_assays_with_one_publication_candidate": sum(
            row["status"] == "matched_current_assay" and len(row["publication_candidates"]) == 1
            for row in links
        ),
        "published_panel_slots": len(panel),
        "published_numeric_measurements": sum(row["mic_um"] is not None for row in panel),
        "certificates": len(coa),
        "new_adoption_eligible_rows": 0,
        "inner_oof_technical_rows": sum(fold_counts.values()),
        "inner_validation_rows_by_fold": dict(sorted(fold_counts.items())),
        "unresolved": [
            "Numbered archaeasin IDs are not linked to certificate sample names/sequences",
            "Blank published cells are not inferred to be inactive or censored",
            "Mass consistency alone does not certify termini or stereochemistry",
            "Current DBAASP assay reference tokens are not verified paper identifiers",
            "Full checkpoint-specific APEX training list remains unavailable",
            "Outer development is unchanged; inner OOF reuses existing training observations only",
        ],
        "source_sha256": {
            str(path.relative_to(args.sources)): file_sha256(path)
            for path in sorted(args.sources.rglob("*"))
            if path.is_file()
        },
        "code_sha256": {
            str(path): file_sha256(path)
            for path in [Path(__file__), Path("src/robust_apex_qd/research/provenance.py")]
        },
    }
    write_json(args.output / "feasibility_review.json", summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"source_sha256", "code_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
