import importlib
from pathlib import Path

import pytest


def test_generation_costs_count_shared_sources_once_and_exclude_recorded_pauses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    account = importlib.import_module("audit_competition_scale").generation_totals
    source = dict(
        source_path="ddim.csv",
        newly_generated=True,
        input_rows=120000,
        generation_wall_seconds=100.0,
        paused_seconds=20.0,
    )
    other = dict(
        source_path="old.csv",
        newly_generated=False,
        input_rows=60000,
        generation_wall_seconds=0.0,
        paused_seconds=0.0,
    )
    result = account([source, source, other])
    assert result == dict(
        new_raw_count=120000,
        generation_wall_seconds=100.0,
        paused_seconds=20.0,
        active_generation_wall_seconds=80.0,
    )
    with pytest.raises(ValueError, match="pause"):
        account([{**source, "paused_seconds": 101.0}])
    with pytest.raises(ValueError, match="identity"):
        account([source, {**source, "input_rows": 120001}])
