import csv
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import robust_apex_qd.pipeline as pipeline
from robust_apex_qd.io.fasta import FastaRecord, write_fasta
from robust_apex_qd.pipeline import (
    PipelineOptions,
    _publish_output,
    _run_advanced_selection,
    run_pipeline,
)
from robust_apex_qd.validation.compliance import SubmissionValidationError, ValidationReport


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_output(path: Path) -> dict[str, str]:
    path.mkdir()
    names = ("library.fasta", "top.fasta", "ranking.tsv", "manifest.json")
    for name in names:
        (path / name).write_text(f"known-good-{name}\n")
    return {name: _sha256(path / name) for name in names}


def test_full_size_config_routes_through_advanced_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "library_selection:\n  enabled: true\n  adopted_variant: L2\n  minimum_full_pool_size: 1\n"
    )
    output = tmp_path / "advanced"
    options = replace(
        PipelineOptions.smoke(
            output_dir=output,
            raw_pool_size=40,
            library_size=32,
            top_k=8,
            batch_size=4,
        ),
        config_path=config,
    )

    def fake_advanced(active_options: PipelineOptions, temporary: Path) -> None:
        records = [
            FastaRecord(f"cand_{index:06d}", "ACDEFGH" + "A" * index)
            for index in range(1, active_options.library_size + 1)
        ]
        write_fasta(records, temporary / "library.fasta")
        write_fasta(records[: active_options.top_k], temporary / "top.fasta")
        with (temporary / "ranking.tsv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("rank", "candidate_id", "sequence", "final_score"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for rank, record in enumerate(records[: active_options.top_k], start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "candidate_id": record.header,
                        "sequence": record.sequence,
                        "final_score": 1.0,
                    }
                )

    monkeypatch.setattr(pipeline, "_run_advanced_selection", fake_advanced)
    monkeypatch.setattr(pipeline, "require_valid_submission", lambda *args, **kwargs: None)

    run_pipeline(options)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["advanced_selection"] is True
    assert manifest["manual_intervention"] is False
    for name, expected in manifest["output_sha256"].items():
        assert _sha256(output / name) == expected


def test_validation_and_swap_failures_preserve_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    before = _old_output(output)
    options = PipelineOptions.smoke(
        output_dir=output,
        raw_pool_size=40,
        library_size=32,
        top_k=8,
        batch_size=4,
    )

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise SubmissionValidationError(ValidationReport())

    monkeypatch.setattr(pipeline, "require_valid_submission", fail_validation)
    with pytest.raises(SubmissionValidationError):
        run_pipeline(options)
    assert {name: _sha256(output / name) for name in before} == before

    monkeypatch.undo()
    original_rename = Path.rename
    failed = False

    def fail_publication_once(source: Path, target: Path) -> Path:
        nonlocal failed
        if source.name.startswith(".output.tmp-") and target == output and not failed:
            failed = True
            raise OSError("simulated publication failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_publication_once)
    with pytest.raises(OSError, match="simulated publication failure"):
        run_pipeline(options)
    assert {name: _sha256(output / name) for name in before} == before


def test_advanced_selection_propagates_configured_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / "temporary"
    work = temporary / "work"
    work.mkdir(parents=True)
    with gzip.open(work / "candidates.csv.gz", "wt", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("candidate_id", "sequence", "valid"),
        )
        writer.writeheader()
        writer.writerow({"candidate_id": "cand_1", "sequence": "ACDEFGHI", "valid": "True"})
    known = tmp_path / "custom-known.fasta"
    challenge = tmp_path / "custom-challenge.fasta"
    known.write_text(">known\nACDEFGHI\n")
    challenge.write_text(">challenge\nKLMNPQRS\n")
    options = replace(
        PipelineOptions.smoke(output_dir=tmp_path / "output"),
        training_fasta=known,
        challenge_fasta=challenge,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_run_checked", commands.append)

    _run_advanced_selection(options, temporary)

    by_script = {Path(command[1]).name: command for command in commands}
    assert by_script["compute_physchem.py"][
        by_script["compute_physchem.py"].index("--reference-fasta") + 1
    ] == str(known)
    assert by_script["compute_embeddings.py"][
        by_script["compute_embeddings.py"].index("--reference-fasta") + 1
    ] == str(known)
    assert by_script["select_library.py"][
        by_script["select_library.py"].index("--challenge-fasta") + 1
    ] == str(challenge)
    assert by_script["select_top.py"][by_script["select_top.py"].index("--known-fasta") + 1] == str(
        known
    )


def test_backup_cleanup_failure_is_nonfatal_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "value.txt").write_text("old\n")
    temporary = tmp_path / ".output.tmp-run"
    temporary.mkdir()
    (temporary / "value.txt").write_text("new\n")
    monkeypatch.setattr(
        pipeline.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(PermissionError("busy backup")),
    )

    _publish_output(temporary, output)

    assert (output / "value.txt").read_text() == "new\n"
    assert "retained backup" in caplog.text
    backups = list(tmp_path.glob("output.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "value.txt").read_text() == "old\n"
