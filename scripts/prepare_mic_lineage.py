"""Reconstruct measurement lineage without rewriting pinned measurements or old folds."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from run_mic_research import checked_manifest, finish_stage, write_json

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.data import parse_mic
from robust_apex_qd.research.metadata import publication_keys
from robust_apex_qd.research.mic_data import normalized_observations
from robust_apex_qd.research.mic_lineage import (
    LineagedMICObservation,
    MeasurementLineage,
    capture_execution,
    duplicate_links,
    reference_articles,
)


def text_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null", "nan", "n/a", "unknown", "-"} else value


def assay_key(row: dict[str, Any], current: bool) -> tuple[str, ...]:
    relation, value = parse_mic(text_value(row.get("concentration" if current else "raw_value")))
    return (
        text_value(row.get("targetSpecies" if current else "target")),
        str(relation),
        str(value),
        text_value(row.get("unit" if current else "raw_unit"))
        .replace("μ", "µ")
        .replace("uM", "µM"),
        *(text_value(row.get(k)) for k in ["medium", "cfu", "note"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("work/competition_exploration/20260912-a/phase1-r4"),
    )
    parser.add_argument(
        "--prepared", type=Path, default=Path("work/mic_prediction/20260913-a/prepare")
    )
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--reference-js", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.development, args.prepared, *args.metadata])
    capture_execution(args.output)
    started = time.monotonic()
    inputs = checked_manifest(args.development / "split_manifest.json")
    inputs.update(checked_manifest(args.prepared / "manifest.json"))
    # The public display uses the raw reference token and ++index over the article array.
    registration_path = Path("configs/mic_phase01_sources.json")
    registration = json.loads(registration_path.read_text())
    source = next(r for r in registration["sources"] if r["path"] == args.reference_js.name)
    verify_hashes({str(args.reference_js): source["sha256"]})
    inputs[str(registration_path)] = file_sha256(registration_path)
    js = args.reference_js.read_text()
    if not all(s in js for s in ["text(item.reference)", "$.each(data.articles", "html(++index)"]):
        raise ValueError("Reference display evidence has changed; review before linking studies")
    inputs[str(args.reference_js)] = file_sha256(args.reference_js)
    metadata = {}
    for root in args.metadata:
        manifest = json.loads((root / "metadata_manifest.json").read_text())
        inputs[str(root / "metadata_manifest.json")] = file_sha256(root / "metadata_manifest.json")
        for record in manifest["records"]:
            if record["status"] != "fetched":
                continue
            path = root / f"{record['id']}.json"
            verify_hashes({str(path): record["sha256"]})
            inputs[str(path)] = record["sha256"]
            raw = json.loads(path.read_text())
            if int(raw["id"]) != int(record["id"]):
                raise ValueError("Metadata ID mismatch")
            if raw["id"] in metadata and metadata[raw["id"]][2] != record["sha256"]:
                raise ValueError("Conflicting metadata snapshots require explicit resolution")
            metadata[raw["id"]] = (raw, str(path), record["sha256"])
    source = pd.read_json(args.development / "observations.jsonl", lines=True)
    included_ids = set(pd.read_json(args.prepared / "rows.jsonl", lines=True).observation_id)
    # Retain excluded and duplicate exports, including chemically unknown and invalid values.
    records = normalized_observations(source)
    lineage, export_keys = [], []
    assay_index = {}
    for identifier, (raw, _, _) in metadata.items():
        index: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for assay in raw.get("targetActivities", []):
            if text_value(assay.get("activityMeasureGroup")) == "MIC":
                index.setdefault(assay_key(assay, True), []).append(assay)
        assay_index[identifier] = index
    for old in source.to_dict("records"):
        identifier = old["observation_id"]
        row = MeasurementLineage(
            observation_id=identifier,
            source_record=(
                f"battleamp/data/dbaasp/dbaasp_activity.csv:zero_based_row={identifier.split(':')[1]}"
                if old["source"] == "battleamp"
                else f"qmap_hf/dbaasp.json:id={old['source_id']}:target={old['target']}"
            ),
        )
        if old["source_id"] in metadata:
            raw, path, digest = metadata[old["source_id"]]
            if raw["sequence"] == old["sequence"]:
                matches = (
                    assay_index[old["source_id"]].get(assay_key(old, False), [])
                    if old["source"] == "battleamp"
                    else []
                )
                chemistry_matches = all(
                    text_value(old[k]) == text_value(raw[v])
                    for k, v in [("nterminal", "nTerminus"), ("cterminal", "cTerminus")]
                )
                studies = (
                    reference_articles(matches[0].get("reference"), raw.get("articles", []))
                    if len(matches) == 1 and chemistry_matches
                    else []
                )
                row = MeasurementLineage(
                    **row.model_dump(
                        exclude={
                            "metadata_path",
                            "metadata_sha256",
                            "assay_ids",
                            "study_ids",
                            "publication_candidates",
                            "linkage_evidence",
                            "raw_assays",
                        }
                    ),
                    metadata_path=path,
                    metadata_sha256=digest,
                    assay_ids=sorted(int(a["id"]) for a in matches),
                    study_ids=studies,
                    publication_candidates=sorted(publication_keys(raw.get("articles", []))),
                    linkage_evidence=(
                        f"unique sequence/termini/target/value/unit/medium/cfu/note match; "
                        f"one-based article display SHA256={inputs[str(args.reference_js)]}; "
                        "database attribution, not original-paper assay validation"
                        if studies
                        else None
                    ),
                    raw_assays=matches,
                )
        lineage.append(row)
        key = [
            old[k]
            for k in [
                "source",
                "source_id",
                "sequence",
                "target",
                "raw_value",
                "raw_unit",
                "chemical_form",
                "medium",
                "cfu",
                "note",
            ]
        ]
        export_keys.append(
            hashlib.sha256(json.dumps([text_value(v) for v in key]).encode()).hexdigest()
        )
    duplicate_frame = source[
        ["observation_id", "source", "source_id", "sequence", "duplicate_export", "included"]
    ].copy()
    duplicate_frame["export_key"] = export_keys
    links = duplicate_links(duplicate_frame, {r.observation_id: r.assay_ids for r in lineage})
    audit = []
    with (args.output / "observations.jsonl").open("w") as handle:
        for observation, record, (_, old) in zip(records, lineage, source.iterrows(), strict=True):
            status, duplicate_of = links[record.observation_id]
            updated = MeasurementLineage.model_validate(
                {**record.model_dump(), "duplicate_status": status, "duplicate_of": duplicate_of}
            )
            # Database attribution does not establish comparable experimental conditions.
            observation.update(
                schema_version=2,
                lineage=updated.model_dump(),
                duplicate_of=duplicate_of,
                assay_publication_verified=False,
                conversion_evidence="source_uM"
                if text_value(observation["raw_unit"]) in {"µM", "μM", "uM"}
                else observation["conversion_evidence"],
            )
            observation["legacy_included"] = bool(old.included)
            observation["screen_included"] = record.observation_id in included_ids
            handle.write(
                LineagedMICObservation.model_validate(observation).model_dump_json() + "\n"
            )
            audit.append(
                dict(
                    observation_id=record.observation_id,
                    duplicate_status=status,
                    duplicate_of=duplicate_of,
                    assay_matches=len(record.assay_ids),
                    study_ids="|".join(record.study_ids),
                    metadata_status="sequence_matched"
                    if record.metadata_path
                    else "missing_or_sequence_mismatch",
                )
            )
    audit_frame = duplicate_frame.merge(
        pd.DataFrame(audit), on="observation_id", validate="one_to_one"
    )
    audit_frame.to_csv(args.output / "duplicate_audit.csv", index=False)
    screen = [r for r in lineage if r.observation_id in included_ids]
    write_json(
        args.output / "observation_manifest.json",
        dict(
            schema_version=2,
            observations=len(records),
            screen_rows=len(screen),
            db_attributed_screen_rows=sum(bool(r.study_ids) for r in screen),
            db_attributed_screen_fraction=sum(bool(r.study_ids) for r in screen) / len(screen),
            publication_candidate_screen_rows=sum(bool(r.publication_candidates) for r in screen),
            duplicate_status_counts=audit_frame.duplicate_status.value_counts().to_dict(),
            original_paper_assay_verified=0,
            delta_eligible=False,
            duplicate_policy=(
                "Current database assay linkage; identical exports without unique "
                "matching assay IDs remain unresolved. Legacy exclusions retained,"
                " not asserted to be proven duplicates."
            ),
            metadata_policy=(
                "Current metadata cannot rewrite historical measurements, "
                "chemistry, or frozen training membership."
            ),
        ),
    )
    inputs[str(Path(__file__))] = file_sha256(Path(__file__))
    finish_stage(args.output, inputs, started)


if __name__ == "__main__":
    main()
