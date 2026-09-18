import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import robust_apex_qd.pipeline as pipeline
from robust_apex_qd.generation.sampler import SamplerOutOfMemoryError
from robust_apex_qd.pipeline import PipelineOptions, run_pipeline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oom_keeps_existing_valid_output_and_batch_size(tmp_path: Path) -> None:
    output = tmp_path / "generate"
    output.mkdir()
    old_files = {
        name: f"known-good-{name}\n"
        for name in ("library.fasta", "top.fasta", "ranking.tsv", "manifest.json")
    }
    for name, content in old_files.items():
        (output / name).write_text(content)
    before = {name: _sha256(output / name) for name in old_files}
    calls: list[int] = []

    def failing_backend(design_length: int, batch_size: int, round_seed: int) -> list[str]:
        del design_length, round_seed
        calls.append(batch_size)
        raise SamplerOutOfMemoryError("simulated OOM")

    options = PipelineOptions.smoke(
        output_dir=output,
        raw_pool_size=32,
        library_size=24,
        top_k=8,
        batch_size=7,
    )
    options = replace(options, minimum_length=10, maximum_length=10)
    with pytest.raises(SamplerOutOfMemoryError, match="simulated OOM"):
        run_pipeline(options, backend=failing_backend)

    assert calls == [7]
    assert {name: _sha256(output / name) for name in old_files} == before


def test_shared_metadata_failure_happens_before_output_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "generate"
    output.mkdir()
    old_files = {
        name: f"known-good-{name}\n"
        for name in ("library.fasta", "top.fasta", "ranking.tsv", "manifest.json")
    }
    for name, content in old_files.items():
        (output / name).write_text(content)
    before = {name: _sha256(output / name) for name in old_files}

    def failing_copy(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated shared metadata failure")

    monkeypatch.setattr(pipeline.shutil, "copy2", failing_copy)
    options = PipelineOptions.smoke(
        output_dir=output,
        raw_pool_size=40,
        library_size=32,
        top_k=8,
        batch_size=4,
    )

    with pytest.raises(OSError, match="simulated shared metadata failure"):
        run_pipeline(options)

    assert {name: _sha256(output / name) for name in old_files} == before
