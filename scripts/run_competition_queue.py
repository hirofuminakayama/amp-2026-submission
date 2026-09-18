"""Execute a frozen local research queue and verify every completed stage."""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.features.embeddings import file_sha256


def verify_completion(
    path: Path, expected: dict[str, Any], source: tuple[str, str] | None = None
) -> None:
    manifest = json.loads(path.read_text())
    if source is not None:
        source_path = Path(source[0]).resolve()
        source_hashes = [
            digest
            for name, digest in manifest.get("input_sha256", {}).items()
            if Path(name).resolve() == source_path
        ]
        if "source_sha256" in manifest:
            source_hashes.append(manifest["source_sha256"])
        if not source_hashes or any(digest != source[1] for digest in source_hashes):
            raise ValueError("Completion was produced by a different executed source")
    for field, value in expected.items():
        actual = manifest
        for key in field.split("."):
            actual = actual[key]
        if actual != value:
            raise ValueError(f"Completion differs from expected {field}")
    verify_hashes(
        {str(path.parent / name): digest for name, digest in manifest["artifacts_sha256"].items()}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {p: file_sha256(Path(p)) for p in config["sources"]}
    (args.output / "protocol.json").write_text(
        json.dumps(dict(config=config, sources_sha256=sources), indent=2) + "\n"
    )
    archive = args.output / "sources"
    archive.mkdir()
    for name, digest in sources.items():
        path = Path(name)
        (archive / (digest + "-" + path.name)).write_bytes(path.read_bytes())
    records = []
    state: dict[str, Any] = dict(status="running", records=records)

    def save() -> None:
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    try:
        for job in config["jobs"]:
            state["current"] = job["id"]
            save()
            started = time.monotonic()
            for name in job.get("wait_for", []):
                path = Path(name)
                while not path.exists():
                    if time.monotonic() - started > config["wait_timeout_seconds"]:
                        raise TimeoutError(f"Timed out waiting for {path}")
                    time.sleep(30)
            verify_hashes(sources)
            completion = Path(job["completion"])
            reused = completion.exists()
            if not reused:
                command = job.get("command")
                if not command:
                    raise ValueError("Missing completed stage and no command registered")
                with (args.output / (job["id"] + ".log")).open("w") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode:
                    raise RuntimeError(f"Stage {job['id']} exited {result.returncode}")
            command = job.get("command", [])
            scripts = [
                part for part in command if part.startswith("scripts/") and part.endswith(".py")
            ]
            source = (scripts[0], sources[scripts[0]]) if scripts else None
            verify_completion(completion, job.get("expected", {}), source)
            records.append(
                dict(
                    id=job["id"],
                    reused=reused,
                    completion=str(completion),
                    completion_sha256=file_sha256(completion),
                    seconds=time.monotonic() - started,
                )
            )
            print(f"Verified {job['id']}", flush=True)
            save()
        state["status"] = "complete"
        save()
    except Exception as error:
        state.update(status="failed", error=str(error))
        save()
        logging.exception("Failed")
        raise


if __name__ == "__main__":
    main()
