import importlib
import json
from pathlib import Path

import pytest


def test_queue_completion_requires_expected_counts_and_intact_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    verify = importlib.import_module("run_competition_queue").verify_completion
    artifact = tmp_path / "raw.txt"
    artifact.write_text("raw outputs")
    from robust_apex_qd.features.embeddings import file_sha256

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(job=dict(count=120000), artifacts_sha256={artifact.name: file_sha256(artifact)})
        )
    )
    verify(manifest, {"job.count": 120000})
    with pytest.raises(ValueError, match="executed source"):
        verify(manifest, {"job.count": 120000}, ("script.py", "different"))
    with pytest.raises(ValueError, match="expected"):
        verify(manifest, {"job.count": 60000})
    artifact.write_text("truncated")
    with pytest.raises(ValueError):
        verify(manifest, {"job.count": 120000})


@pytest.mark.parametrize("absolute", [False, True])
def test_queue_completion_resolves_source_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    verify = importlib.import_module("run_competition_queue").verify_completion
    monkeypatch.chdir(tmp_path)
    source = "scripts/generate.py"
    key = str(Path(source).resolve()) if absolute else source
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(input_sha256={key: "expected"}, artifacts_sha256={})))
    verify(manifest, {}, (source, "expected"))
    with pytest.raises(ValueError, match="executed source"):
        verify(manifest, {}, (source, "changed"))
    with pytest.raises(ValueError, match="executed source"):
        verify(manifest, {}, ("other/generate.py", "expected"))


def test_queue_completion_rejects_conflicting_source_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    verify = importlib.import_module("run_competition_queue").verify_completion
    source = "scripts/generate.py"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                input_sha256={source: "expected", str(Path(source).resolve()): "changed"},
                artifacts_sha256={},
            )
        )
    )
    with pytest.raises(ValueError, match="executed source"):
        verify(manifest, {}, (source, "expected"))
