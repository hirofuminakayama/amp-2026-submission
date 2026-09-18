"""Verify a registered published MIC table and export strictly isolated pair candidates."""

import argparse
import itertools
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from run_mic_research import checked_manifest, finish_stage, write_json

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.activity_pairs import (
    cliff_similarity,
    matches_reviewed_assay,
    paper_mic,
)
from robust_apex_qd.research.mic_delta import DeltaObservation, build_delta_pairs
from robust_apex_qd.research.mic_lineage import capture_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = checked_manifest(args.prepared / "manifest.json")
    registry_path = Path("configs/mic_pair_sources.json")
    registry = json.loads(registry_path.read_text())
    inputs[str(registry_path)] = file_sha256(registry_path)
    digest = file_sha256(args.paper)
    if digest != registry["paper"]["sha256"]:
        raise ValueError("Published table differs from the reviewed source snapshot")
    inputs[str(args.paper)] = digest
    root = ET.parse(args.paper).getroot()
    if root.findtext('.//article-id[@pub-id-type="pmid"]') != "33401476":
        raise ValueError("Curation rules apply only to the reviewed article")
    table = root.find('.//table-wrap[@id="antibiotics-10-00036-t001"]')
    if table is None:
        raise ValueError("Reviewed MIC table missing")
    targets = [
        "Enterococcus faecium E007",
        "Staphylococcus aureus ATCC 29231",
        "Staphylococcus aureus MW2",
        "Klebsiella pneumoniae WGLW2",
        "Acinetobacter baumannii ATCC 17978",
        "Pseudomonas aeruginosa PA14",
        "Enterobacter aerogenes ATCC 13048",
        "Candida albicans ATCC 10231",
    ]
    lookup = {}
    for row in table.findall(".//tr"):
        cells = ["".join(c.itertext()).strip() for c in row]
        if len(cells) != 10 or not cells[0].startswith("C"):
            continue
        for target, value in zip(targets, cells[2:], strict=True):
            lookup[cells[1], target] = (cells[0], *paper_mic(value))
    fresh_output(args.output, [args.prepared, args.paper])
    capture_execution(args.output)
    (args.output / "paper.xml").write_bytes(args.paper.read_bytes())
    rows = pd.read_json(args.prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    lineaged = {
        r["observation_id"]: r
        for r in map(json.loads, (args.prepared / "observations.jsonl").read_text().splitlines())
    }
    curated, audit = [], []
    for row in rows.itertuples():
        match = lookup.get((row.sequence, row.target))
        if match is None:
            continue
        name, relation, mic = match
        original = lineaged[row.observation_id]
        metadata = original["lineage"]["metadata_path"]
        attributed = False
        if metadata:
            record = json.loads(Path(metadata).read_text())
            attributed = (
                any(
                    a.get("pubmed", {}).get("pubmedId") == "33401476"
                    for a in record.get("articles", [])
                )
                and "pubmed:33401476" in original["lineage"]["study_ids"]
            )
        eligible = (
            attributed
            and len(original["lineage"]["raw_assays"]) == 1
            and matches_reviewed_assay(original["lineage"]["raw_assays"][0], registry["assay"])
            and row.exact_regression
            and relation == "="
            and np.isclose(
                row.mic_um, mic, rtol=registry["value_rtol"], atol=registry["value_atol_um"]
            )
            and original["chemical_form"] == "reported_linear_free"
        )
        audit.append(
            dict(
                observation_id=row.observation_id,
                table_name=name,
                published_relation=relation,
                published_mic_um=mic,
                exported_mic_um=row.mic_um,
                eligible=bool(eligible),
                chemistry_evidence=original["chemical_evidence"],
                reason="table/sequence/target/DB-chemistry matched"
                if eligible
                else "censoring, attribution, chemistry or value mismatch",
            )
        )
        if eligible:
            curated.append(
                DeltaObservation(
                    observation_id=row.observation_id,
                    sequence=row.sequence,
                    scaffold_id=str(row.homology_group),
                    target_id=row.target,
                    chemical_profile=original["chemical_form"],
                    study_id="pubmed:33401476",
                    comparable_assay_id="33401476:table1:methods4.2",
                    verification_evidence=(
                        f"{args.paper} SHA256={digest}; Table1 sequence/strain/MIC; Methods4.2; "
                        f"chemical form from {original['chemical_evidence']} "
                        "(not independently mass-spectrometry verified)"
                    ),
                    mic_um=float(row.mic_um),
                    objective="measured_mic",
                    relation="=",
                    assay_publication_verified=True,
                )
            )
    partition = dict(zip(rows.sequence, rows.homology_fold.astype(str), strict=True))
    endpoints, diagnostics = [], []
    for a, b in itertools.combinations(curated, 2):
        if a.sequence == b.sequence or a.target_id != b.target_id:
            continue
        similarity = cliff_similarity(a.sequence, b.sequence)
        same = partition[a.sequence] == partition[b.sequence] and a.scaffold_id == b.scaffold_id
        delta = np.log2(b.mic_um / a.mic_um)
        diagnostics.append(
            dict(
                left_observation_id=a.observation_id,
                right_observation_id=b.observation_id,
                similarity=similarity,
                delta_log2_um=delta,
                one_dilution=bool(abs(delta) >= 1),
                two_dilution=bool(abs(delta) >= 2),
                same_partition=same,
                eligible=same and similarity >= registry["similarity_threshold"],
            )
        )
        if same and similarity >= registry["similarity_threshold"]:
            endpoints.append((a.observation_id, b.observation_id))
    pairs = []
    by_id = {r.observation_id: r for r in curated}
    for fold in sorted(set(partition.values())):
        pairs.extend(
            build_delta_pairs(
                curated,
                [(a, b) for a, b in endpoints if partition[by_id[a].sequence] == fold],
                partition,
                fold,
            )
        )
    (args.output / "observations.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in curated)
    )
    (args.output / "pairs.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in pairs))
    pd.DataFrame(audit).to_csv(args.output / "curation_audit.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "pair_manifest.csv", index=False)
    write_json(
        args.output / "curation.json",
        dict(
            paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7824259/",
            license="CC BY 4.0",
            table_matches=len(audit),
            verified_observations=len(curated),
            strict_pairs=len(pairs),
            pair_folds=sorted({p.partition for p in pairs}),
            value_tolerance="1% relative + 0.005 uM rounding",
            scope="single-paper pilot; other DB-only or unknown-assay rows remain ineligible",
            chemistry=(
                "matching reported_linear_free from existing source; "
                "no independent chemical remeasurement"
            ),
        ),
    )
    finish_stage(args.output, inputs, started)


if __name__ == "__main__":
    main()
