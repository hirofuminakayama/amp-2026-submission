"""Audit current DBAASP provenance for peptides measured on exact APEX strains."""

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from robust_apex_qd.calibration.model import STRAIN_TO_PATHOGEN
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256


def fetch(identifier: int, output: Path) -> dict[str, Any]:
    url = f"https://dbaasp.org/peptides/{identifier}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        record = json.loads(payload)
        if record["id"] != identifier:
            raise ValueError("DBAASP response ID mismatch")
        path = output / f"{identifier}.json"
        with path.open("xb") as handle:
            handle.write(payload)
        return {
            "id": identifier,
            "url": url,
            "sha256": file_sha256(path),
            "status": "fetched",
            "sequence": record["sequence"],
            "article_ids": [article["id"] for article in record.get("articles", [])],
            "pubmed_ids": [
                article.get("pubmed", {}).get("pubmedId")
                for article in record.get("articles", [])
                if article.get("pubmed")
            ],
            "unusual_amino_acids": record.get("unusualAminoAcids"),
            "intrachain_bonds": record.get("intrachainBonds"),
            "interchain_bonds": record.get("interchainBonds"),
            "nterminal": record.get("nTerminus"),
            "cterminal": record.get("cTerminus"),
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
        KeyError,
    ) as error:
        return {"id": identifier, "url": url, "status": "failed", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.activity])
    frame = pd.read_csv(args.activity)
    identifiers = sorted(
        int(value)
        for value in frame.loc[frame.targetSpecies.isin(STRAIN_TO_PATHOGEN), "id"].unique()
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(lambda identifier: fetch(identifier, args.output), identifiers))
    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_activity_sha256": file_sha256(args.activity),
        "scope": "exact APEX strain matches; current metadata cannot rewrite pinned measurements",
        "study_mapping": "article candidates per peptide; assay-to-article links unverified",
        "records": rows,
    }
    (args.output / "metadata_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"requested": len(rows), "fetched": sum(row["status"] == "fetched" for row in rows)}
        )
    )


if __name__ == "__main__":
    main()
