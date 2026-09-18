import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from robust_apex_qd import cli
from robust_apex_qd.generation.sampler import Candidate
from robust_apex_qd.pipeline import PipelineOptions, _git_source_state, _select_official_top

ROOT = Path(__file__).resolve().parents[2]


def test_official_broad_spectrum_entry_point_uses_category_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_text = (ROOT / "pyproject.toml").read_text()
    assert (
        'generate_broad_spectrum = "robust_apex_qd.cli:generate_broad_spectrum_main"'
        in project_text
    )
    monkeypatch.setattr(sys, "argv", ["generate_broad_spectrum", "--show-defaults"])

    with pytest.raises(SystemExit) as exit_info:
        cli.generate_broad_spectrum_main()

    assert exit_info.value.code == 0
    defaults = json.loads(capsys.readouterr().out)
    assert Path(defaults["output_dir"]).name == "generate_broad_spectrum"


def test_vendored_challenge_verifier_retains_bsd_notice() -> None:
    notice = ROOT / "licenses/amp-challenge-2027-BSD-3-Clause.txt"

    assert notice.is_file()
    text = notice.read_text()
    assert "BSD 3-Clause License" in text
    assert "Copyright (c) 2026, Ewa Szczurek lab" in text
    assert "Redistribution and use in source and binary forms" in text
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" in text


def test_clean_clone_uses_official_antibacterial_reference() -> None:
    script = (ROOT / "scripts/clean_clone_verify.sh").read_text()

    assert '--antibacterial-fasta "${clone_dir}/data/antibacterial.fasta"' in script


def test_clean_clone_supports_feature_branch_and_writes_summary_log() -> None:
    script = (ROOT / "scripts/clean_clone_verify.sh").read_text()

    assert 'submission_branch="${SUBMISSION_BRANCH:-}"' in script
    assert '--branch "${submission_branch}" --single-branch' in script
    assert 'git -C "${clone_dir}" lfs install --local' in script
    assert 'summary_log="${work_dir}/clean_clone.log"' in script


def test_git_source_state_hashes_tracked_and_untracked_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex@example.invalid",
            "commit",
            "-qm",
            "Initial",
        ],
        cwd=repository,
        check=True,
    )

    clean = _git_source_state(repository)
    (repository / "tracked.txt").write_text("dirty\n")
    (repository / "untracked.txt").write_text("new\n")
    dirty = _git_source_state(repository)
    (repository / "untracked.txt").write_text("changed\n")
    changed = _git_source_state(repository)

    assert not clean.dirty
    assert clean.diff_sha256 is None
    assert dirty.commit == clean.commit
    assert dirty.dirty
    assert dirty.diff_sha256 is not None and len(dirty.diff_sha256) == 64
    assert changed.diff_sha256 != dirty.diff_sha256


def test_official_top_passes_full_library_to_lazy_ranked_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        Candidate("cand_1", "AAAAAAAAAA", 0, 0, 42, 10, 10, True, ""),
        Candidate("cand_2", "CCCCCCCCCC", 1, 0, 42, 10, 10, True, ""),
    ]
    options = replace(
        PipelineOptions.smoke(output_dir=tmp_path / "output", library_size=2, top_k=1),
        selection_similarity_max=0.78,
    )
    observed: dict[str, object] = {}

    def fake_select_top(
        library: list[str],
        top_k: int,
        references: set[str],
        known_amps: list[str],
        work_dir: Path,
        *,
        challenge_similarity_threshold: float,
    ) -> list[str]:
        del references, known_amps, work_dir
        observed["library"] = library
        observed["threshold"] = challenge_similarity_threshold
        return library[:top_k]

    import ampdiffusion_starter_kit.generate as upstream_generate

    monkeypatch.setattr(upstream_generate, "select_top", fake_select_top)

    selected = _select_official_top(
        candidates,
        options,
        {"AAAAAAAAAA"},
        tmp_path / "work",
    )

    assert observed["library"] == ["AAAAAAAAAA", "CCCCCCCCCC"]
    assert observed["threshold"] == 0.78
    assert selected == [candidates[0]]
