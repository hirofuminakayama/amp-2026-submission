import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.features.embeddings import file_sha256


def test_stage_manifest_covers_nested_manifests_and_detects_modified_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("run_competition_bioaccuracy")
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    output = tmp_path / "output"
    (output / "fold").mkdir(parents=True)
    (output / "fold/manifest.json").write_text("{}\n")
    expected = {str(source): file_sha256(source)}
    runner.finish_stage(output, expected, time.monotonic())
    saved = json.loads((output / "manifest.json").read_text())
    assert set(saved["artifacts_sha256"]) == {"fold/manifest.json"}
    runner.checked_manifest(output / "manifest.json")
    source.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="hash"):
        runner.finish_stage(output, expected, time.monotonic())
    (output / "fold/manifest.json").write_text("changed\n")
    with pytest.raises(ValueError, match="hash"):
        runner.checked_manifest(output / "manifest.json")


def test_nested_score_requires_complete_coverage_and_reports_small_cohorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    score = importlib.import_module("evaluate_bio_selection").score_labels
    labels = pd.DataFrame(
        dict(
            sequence=["a", "b"],
            species=["E", "E"],
            molecule_id=["a", "b"],
            hit_lower=[1, 0],
            hit_upper=[1, 0],
        )
    )
    with pytest.raises(ValueError, match="coverage"):
        score(labels, {"a": 1.0})
    result = score(labels, {"a": 1.0, "b": 0.0})
    assert result[0]["lower"] == 1 and result[0]["requested"] == 1
    assert all(not r["supported"] and r["lower"] is None for r in result[1:])


def test_censored_hc50_metrics_do_not_treat_threshold_as_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    metric = importlib.import_module("train_bio_measured_hc50").bound_metrics
    result = metric(np.array([4.0, 7.0]), np.array([4.0, np.inf]), np.array([5.0, 9.0]))
    assert result["exact_rows"] == 1
    assert result["exact"]["mae"] == 1
    assert result["interval_mae"] == 0.5


def test_family_choice_ignores_outer_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    choose = importlib.import_module("combine_bio_nested").choose_family
    audits = {
        "ridge": dict(
            selected_arm="h", candidates=[dict(arm="h", inner_score=0.5)], outer_score=0.9
        ),
        "mlp": dict(selected_arm="h", candidates=[dict(arm="h", inner_score=0.6)], outer_score=0.1),
    }
    assert choose(audits) == "mlp"
    audits["ridge"]["outer_score"], audits["mlp"]["outer_score"] = -100, 100
    assert choose(audits) == "mlp"
