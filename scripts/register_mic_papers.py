"""Build original-paper rights and observed-use registries from pinned local evidence."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.paper_mic import node_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    inputs = {str(args.config): file_sha256(args.config)}
    freeze = Path(cfg["candidate_freeze"])
    inputs[str(freeze)] = file_sha256(freeze)
    candidates = json.loads(freeze.read_text())["papers"]
    reviews = json.loads(Path(cfg["reviews"]).read_text())
    inputs[cfg["reviews"]] = file_sha256(Path(cfg["reviews"]))
    if not 1 <= len(candidates) <= 30 or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("Expected unique bounded paper candidates")
    papers, rights = [], []
    for path in Path(cfg["sources"]).iterdir():
        if path.is_file():
            inputs[str(path)] = file_sha256(path)
    generator_sequences = set()
    for token in cfg["generator"]["known_reference_files"]:
        path = Path(token)
        inputs[token] = file_sha256(path)
        generator_sequences.update(r.sequence for r in read_fasta(path))
    for candidate in candidates:
        pmc = candidate.get("pmcid")
        path = Path(cfg["sources"]) / f"{pmc}.xml"
        review = reviews[candidate["id"]]
        permission = ""
        if path.exists():
            inputs[str(path)] = file_sha256(path)
            root = ET.parse(path).getroot()
            if root.findtext('.//article-id[@pub-id-type="pmid"]') != candidate["id"]:
                raise ValueError("Paper identifier mismatch")
            permission = node_text(root.find(".//permissions"))
        status = review["rights_status"]
        if status == "full_ready" and (
            "by-nc" in permission.lower() or "non-commercial" in permission.lower()
        ):
            raise ValueError("Restricted license cannot be auto-cleared")
        record = dict(
            paper_id="pubmed:" + candidate["id"],
            pmcid=pmc,
            doi=candidate.get("doi"),
            title=candidate["title"],
            publication_date=candidate.get("firstPublicationDate"),
            url="https://doi.org/" + candidate["doi"],
            xml_url=f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML"
            if pmc
            else None,
            snapshot=str(path) if path.exists() else None,
            sha256=inputs.get(str(path)),
            retrieved_date=cfg["retrieved_date"],
            discovery=candidate["discovery"],
            supplementary_archive=str(Path(cfg["sources"]) / f"{pmc}-supp.zip")
            if (Path(cfg["sources"]) / f"{pmc}-supp.zip").exists()
            else None,
            supplementary_url=f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/supplementaryFiles"
            if pmc
            else None,
            **review,
        )
        papers.append(record)
        rights.append(
            dict(
                paper_id=record["paper_id"],
                status=status,
                permission_text=permission,
                license_evidence=f"{path}: article-meta/permissions"
                if path.exists()
                else review["reason"],
                reuse_scope=(
                    "Original authored measurement cells only; attribution and changes disclosed"
                ),
                third_party_exceptions=(
                    "Database comparison/training sets and third-party annotations "
                    "excluded; article license is not database clearance"
                ),
                published=False,
            )
        )
    usage: dict[str, set[str]] = defaultdict(set)
    sequences: dict[str, str] = {}
    evidence = []
    for pattern in cfg["fit_manifest_globs"]:
        paths = sorted(Path(".").glob(pattern))
        if not paths:
            raise ValueError(f"No fit evidence: {pattern}")
        for path in paths:
            d = json.loads(path.read_text())
            roles = {
                role: d.get(key, [])
                for role, key in [("training", "training_ids"), ("validation", "validation_ids")]
            }
            if not any(roles.values()):
                continue
            inputs[str(path)] = file_sha256(path)
            for role, ids in roles.items():
                for identifier in ids:
                    if not isinstance(identifier, str):
                        raise ValueError("Fit IDs must be explicit observation strings")
                    usage[identifier].add(role)
            evidence.append(
                dict(path=str(path), kind="fit_ids", counts={k: len(v) for k, v in roles.items()})
            )
    for token in cfg["oof_files"]:
        path = Path(token)
        d = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
        inputs[token] = file_sha256(path)
        for r in d[["observation_id", "sequence"]].drop_duplicates().itertuples(index=False):
            usage[r.observation_id].add("validation_or_model_selection")
            sequences[r.observation_id] = r.sequence
        evidence.append(dict(path=token, kind="saved_oof", rows=len(d)))
    studies, candidates_by_id = defaultdict(set), defaultdict(set)
    inventory_ids = set()
    for token in cfg["observation_files"]:
        path = Path(token)
        inputs[token] = file_sha256(path)
        for line in path.open():
            row = json.loads(line)
            identifier = row["observation_id"]
            inventory_ids.add(identifier)
            sequences[identifier] = row["sequence"]
            studies[identifier].update(row.get("lineage", {}).get("study_ids", []))
            candidates_by_id[identifier].update(
                row.get("lineage", {}).get("publication_candidates", [])
            )
    unknown_sequences = sorted(set(usage) - set(sequences))
    records = [
        dict(
            observation_id=k,
            sequence=sequences.get(k),
            roles=sorted(v),
            verified_study_ids=sorted(studies[k]),
            publication_candidates=sorted(candidates_by_id[k]),
        )
        for k, v in sorted(usage.items())
    ]
    fresh_output(args.output, [args.config, freeze, Path(cfg["sources"])])
    capture_execution(args.output)
    (args.output / "generator_reference_sequences.json").write_text(
        json.dumps(sorted(generator_sequences)) + "\n"
    )
    (args.output / "used_observations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (args.output / "downloaded_without_use_evidence.json").write_text(
        json.dumps(sorted(inventory_ids - set(usage)), indent=2) + "\n"
    )
    for name, value in [
        ("paper_registry.json", dict(papers=papers)),
        ("rights_manifest.json", dict(papers=rights)),
        (
            "exposure_manifest.json",
            dict(
                evidence=evidence,
                used_observations=len(records),
                used_sequences=len({r["sequence"] for r in records if r["sequence"]}),
                unresolved_sequence_ids=unknown_sequences,
                verified_used_studies=sorted(set().union(*(studies[k] for k in usage))),
                candidate_used_studies=sorted(set().union(*(candidates_by_id[k] for k in usage))),
                downloaded_without_use_evidence=len(inventory_ids - set(usage)),
                unknowns=cfg["unknowns"],
                generator=cfg["generator"],
                limitation=(
                    "Observed use is a lower bound. Unlisted IDs/papers are not certified unused; "
                    "downloaded-only means no evidence in registered runs."
                ),
            ),
        ),
    ]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    manifest = dict(
        inputs_sha256=inputs,
        artifacts_sha256={
            str(p.relative_to(args.output)): file_sha256(p)
            for p in args.output.rglob("*")
            if p.is_file()
        },
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                papers=len(papers),
                rights={
                    s: sum(p["rights_status"] == s for p in papers)
                    for s in sorted({p["rights_status"] for p in papers})
                },
                used_observations=len(records),
                unknown_sequence_ids=len(unknown_sequences),
            )
        )
    )


if __name__ == "__main__":
    main()
