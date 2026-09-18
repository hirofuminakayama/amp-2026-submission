from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.research.data import normalize_battle
from robust_apex_qd.research.exploration import (
    ExplorationConstraints,
    actual_homology,
    assign_folds,
    development_row,
    oracle_union,
    select_exploration_top,
)


def test_unknown_chemistry_molar_is_usable_but_mass_is_not() -> None:
    raw = {"id": "1", "targetSpecies": "Escherichia coli", "concentration": ">16", "unit": "uM"}
    peptide = {"sequence": "ACDEFGHIK"}
    row = development_row(normalize_battle(raw, peptide, None, 0))
    assert row["usable"] and row["active16"] == 0
    assert row["lower_um"] == 16 and row["upper_um"] is None
    assert not row["exact_regression"] and not row["strict_chemistry"]
    raw["unit"] = "ug/ml"
    assert not development_row(normalize_battle(raw, peptide, None, 1))["usable"]
    chemistry = {"sequence": peptide["sequence"], "bonds": [], "nterminal": "", "cterminal": ""}
    converted = development_row(normalize_battle(raw, peptide, chemistry, 2))
    assert converted["usable"] and converted["mic_um"] != 16
    raw["concentration"] = ""
    assert not development_row(normalize_battle(raw, peptide, None, 3))["usable"]


def test_actual_alignment_and_group_isolation() -> None:
    assert actual_homology("ACDEFGHIKL", "ACDEFGHIKM")
    assert not actual_homology("ACDEFGHIKL", "ACDEYYYYFGHIKL")
    sequences = ["ACDEFGHIKL", "ACDEFGHIKM", "WWWWWWWWWW", "LLLLLLLLLL", "KKKKKKKKKK", "RRRRRRRRRR"]
    folds = assign_folds(sequences, [(sequences[0], sequences[1])], 42)
    assert folds.loc[sequences[0], "homology_fold"] == folds.loc[sequences[1], "homology_fold"]
    assert folds.equals(assign_folds(list(reversed(sequences)), [(sequences[0], sequences[1])], 42))


def test_constraints_official_limit_and_oracle_union() -> None:
    with pytest.raises(ValueError):
        ExplorationConstraints(challenge_max=0.81)
    pool = pd.DataFrame(
        {
            "sequence": ["AAAAAAAA", "CCCCCCCC", "DDDDDDDD"],
            "raw_order": [2, 1, 0],
            "a": [3, 2, 1],
            "b": [1, 2, 3],
        }
    )
    assert oracle_union(pool, ["a", "b"], 1) == {"AAAAAAAA", "DDDDDDDD"}
    pool["candidate_id"] = ["a", "b", "c"]
    pool["embedding_cluster"] = 0
    pool["hard_reject"] = [True, False, False]
    constraints = ExplorationConstraints(
        challenge_max=0.8, known_max=None, pairwise_max=None, cluster_cap=None, physchem="none"
    )
    similarity = {"AAAAAAAA": 0.81, "CCCCCCCC": 0.8, "DDDDDDDD": 0.2}
    selected = select_exploration_top(
        pool, "a", constraints, 2, similarity.__getitem__, lambda s: 1.0
    )
    assert selected.sequence.tolist() == ["CCCCCCCC", "DDDDDDDD"]
    with pytest.raises(ValueError, match="infeasible"):
        select_exploration_top(
            pool.iloc[:2], "a", constraints, 2, similarity.__getitem__, lambda s: 0.0
        )


def test_score_ties_and_hard_union_boundary() -> None:
    pool = pd.DataFrame(
        {
            "sequence": ["AAAAAAAA", "CCCCCCCC", "DDDDDDDD"],
            "raw_order": [0, 1, 2],
            "candidate_id": ["a", "b", "c"],
            "embedding_cluster": [0, 1, 2],
            "hard_reject": False,
            "score": [1.0, 1.0, 1.0],
        }
    )
    c = ExplorationConstraints(known_max=None, pairwise_max=None, cluster_cap=None, physchem="none")
    chosen = select_exploration_top(
        pool.sample(frac=1, random_state=42), "score", c, 2, lambda s: 0.0, lambda s: 0.0
    )
    assert chosen.sequence.tolist() == pool.sequence.tolist()[:2]
    pool["score"] = [np.nan, np.nan, 1.0]
    with pytest.raises(ValueError, match="infeasible"):
        select_exploration_top(pool, "score", c, 2, lambda s: 0.0, lambda s: 0.0)


def test_consensus_remains_separate_and_boundaries_keep_unknown() -> None:
    from robust_apex_qd.research.data import normalize_qmap

    raw = {
        "id": 7,
        "sequence": "ACDEFGHIK",
        "bonds": [],
        "nterminal": "",
        "cterminal": "",
        "targets": {"Escherichia coli": [1, 2, 8]},
    }
    row = development_row(normalize_qmap(raw)[0])
    assert row["usable"] and row["objective"] == "qmap_consensus"
    assert row["mic_um"] is None and not row["exact_regression"]
    for text, active, lower, upper in [
        ("<8", 1, None, 8),
        (">=16", None, 16, None),
        ("=8", 1, 8, 8),
    ]:
        observation = normalize_battle(
            {"id": "1", "targetSpecies": "Escherichia coli", "concentration": text, "unit": "uM"},
            {"sequence": "ACDEFGHIK"},
            None,
            0,
        )
        result = development_row(observation)
        assert result["active16"] == active
        assert (result["lower_um"], result["upper_um"]) == (lower, upper)


def test_group_one_is_explicitly_not_oof() -> None:
    frame = assign_folds(["ACDEFGHIK"], [], 42)
    assert frame.exact_fold.tolist() == [-1]
    assert frame.homology_fold.tolist() == [-1]


def test_scenario_directions_and_missing_components(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("report_competition_selection")
    frame = pd.DataFrame(
        {
            "activity": [1.0, 0.0],
            "weak_species": [1.0, 0.0],
            "safety": [1.0, 0.0],
            "hc50_complete_median": [np.nan, np.nan],
            "fbd": [0.1, 10.0],
            "diversity": [10, 1],
            "library_diversity": [0.9, 0.1],
        }
    )
    scores = module.scenario_scores(frame, {"equal": [0.2] * 5})
    assert scores.scenario_equal.iloc[0] > scores.scenario_equal.iloc[1]
    assert np.isfinite(scores.scenario_equal).all()
    assert scores.family_safety.equals(frame.safety.rank(pct=True))


def test_portfolio_keeps_selected_quota_and_final_constraints() -> None:
    from robust_apex_qd.research.exploration import select_portfolio

    pool = pd.DataFrame(
        {
            "sequence": ["AAAAAAAA", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE"],
            "raw_order": [0, 1, 2, 3],
            "candidate_id": ["a", "c", "d", "e"],
            "embedding_cluster": [0, 1, 2, 3],
            "hard_reject": False,
            "consensus": [4.0, 3.0, 2.0, 1.0],
        }
    )
    for i in range(7):
        pool[f"species_{i}"] = [1.0, 2.0, 3.0, 4.0]
    config = ExplorationConstraints(
        known_max=None, pairwise_max=None, cluster_cap=None, physchem="none"
    )
    selected = select_portfolio(pool, 0.5, config, 2, lambda s: 0.0, lambda s: 0.0)
    assert selected.sequence.tolist() == ["EEEEEEEE", "AAAAAAAA"]
    assert selected.portfolio_role.tolist() == ["specialist", "consensus"]


def test_unmapped_activity_id_rejected_and_conflicting_sequence_is_unknown() -> None:
    from robust_apex_qd.research.exploration import normalize_activity

    raw = {"id": "2", "targetSpecies": "Escherichia coli", "concentration": "8", "unit": "uM"}
    with pytest.raises(ValueError, match="Unmapped measurement ID"):
        normalize_activity(raw, {1: {"sequence": "ACDEFGHIK"}}, {}, 0)
    row = normalize_activity(
        raw,
        {2: {"sequence": "ACDEFGHIK"}},
        {2: {"sequence": "ACDEFGHIL", "bonds": [], "nterminal": "", "cterminal": ""}},
        0,
    )
    assert row["chemical_form"] == "unknown" and row["usable"]


def test_runner_rejects_unregistered_nonempty_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("run_competition_selection")
    destination = tmp_path / "previous"
    destination.mkdir()
    (destination / "keep.txt").write_text("previous experiment")
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(module, "prepare", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--config", str(config), "--output", str(destination), "--stage", "prepare"],
    )
    with pytest.raises(ValueError, match="nonempty"):
        module.main()
    assert [p.name for p in destination.iterdir()] == ["keep.txt"]
