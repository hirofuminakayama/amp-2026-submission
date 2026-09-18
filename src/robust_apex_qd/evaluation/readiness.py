from pathlib import Path

from robust_apex_qd.features.embeddings import file_sha256


def fresh_output(path: Path, inputs: list[Path]) -> None:
    resolved = path.resolve()
    for source in inputs:
        source = source.resolve()
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError("Output must be separate from saved inputs")
    if resolved.exists():
        raise ValueError("Output must be a new directory")
    resolved.mkdir(parents=True)


def verify_hashes(expected: dict[str, str]) -> dict[str, str]:
    actual = {path: file_sha256(Path(path)) for path in expected}
    mismatches = [path for path in expected if actual[path] != expected[path]]
    if mismatches:
        raise ValueError(f"Input hash mismatch: {mismatches}")
    return actual
