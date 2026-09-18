"""Fetch only the public DBAASP records corresponding to registered HC50 molecules."""

import argparse
import concurrent.futures
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fetch_research_metadata import fetch
from run_competition_bioaccuracy import BioaccuracyConfig, read_observations, write_json

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_bioaccuracy.json"))
    args = parser.parse_args()
    config = BioaccuracyConfig.model_validate_json(args.config.read_text())
    output = args.root / "hc50-metadata"
    fresh_output(output, [config.metadata, args.root / "prepare"])
    observations = read_observations(args.root / "prepare/hc50_observations.jsonl")
    identifiers = sorted({int(r.source_id) for r in observations if r.endpoint == "consensus_hc50"})
    original = json.loads((config.metadata / "metadata_manifest.json").read_text())
    cache = {int(r["id"]): r for r in original["records"] if r["status"] == "fetched"}
    records = []
    for identifier in identifiers:
        if identifier in cache:
            record = cache[identifier]
            path = config.metadata / f"{identifier}.json"
            verify_hashes({str(path): record["sha256"]})
            shutil.copyfile(path, output / path.name)
            records.append(dict(**record, reused=True))
    missing = [identifier for identifier in identifiers if identifier not in cache]
    if missing:
        first = fetch(missing[0], output)
        records.append(dict(**first, reused=False))
        if first["status"] == "fetched":
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                for record in executor.map(
                    lambda identifier: fetch(identifier, output), missing[1:]
                ):
                    records.append(dict(**record, reused=False))
                    if len(records) % 50 == 0:
                        print(
                            json.dumps(dict(recorded=len(records), requested=len(identifiers))),
                            flush=True,
                        )
        else:
            for identifier in missing[1:]:
                records.append(
                    dict(id=identifier, status="not_attempted", reason="initial_fetch_failed")
                )
    write_json(
        output / "metadata_manifest.json",
        dict(
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            requested_ids=identifiers,
            original_cache=str(config.metadata),
            original_retrieved_at=original.get("retrieved_at"),
            source_observations_sha256=file_sha256(args.root / "prepare/hc50_observations.jsonl"),
            scope="registered QMAP HC50 IDs only; current source observations kept separate",
            code_sha256={
                str(p): file_sha256(p)
                for p in [Path(__file__), Path("scripts/fetch_research_metadata.py")]
            },
            records=sorted(records, key=lambda r: r["id"]),
        ),
    )
    print(
        json.dumps(
            dict(
                requested=len(identifiers),
                fetched=sum(r["status"] == "fetched" for r in records),
                reused=len(set(identifiers) & set(cache)),
            )
        )
    )


if __name__ == "__main__":
    main()
