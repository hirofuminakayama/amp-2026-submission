"""Fetch pinned public research inputs, verifying existing files without overwriting them."""

import argparse
import json
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from robust_apex_qd.features.embeddings import file_sha256


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    url: str
    revision: str
    sha256: str


class Sources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    retrieved_on: str
    sources: list[Source]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/research_sources.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = Sources.model_validate_json(args.manifest.read_text())
    for source in manifest.sources:
        path = args.output / source.path
        if not path.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Source path must remain inside the output directory")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(source.url, timeout=60) as response:
                payload = response.read()
            import hashlib

            if hashlib.sha256(payload).hexdigest() != source.sha256:
                raise ValueError(f"Downloaded source hash mismatch: {source.path}")
            with path.open("xb") as handle:
                handle.write(payload)
        if file_sha256(path) != source.sha256:
            raise ValueError(f"Existing source hash mismatch: {source.path}")
    print(json.dumps({"verified_sources": len(manifest.sources)}))


if __name__ == "__main__":
    main()
