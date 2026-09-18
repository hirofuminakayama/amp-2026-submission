from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.ranking.adopted import consensus_rank, select_adopted


def test_official_small_run_keeps_adopted_ranker_while_synthetic_smoke_stays_separate(
    tmp_path: Path,
) -> None:
    from robust_apex_qd.pipeline import PipelineOptions, _adopted_selection_enabled

    config = tmp_path / "final.yaml"
    config.write_text("inference:\n  policy: ddim-lref-rankmean\n")
    options = replace(PipelineOptions.smoke(output_dir=tmp_path / "run"), config_path=config)
    assert not _adopted_selection_enabled(options)
    assert _adopted_selection_enabled(replace(options, sampler="official"))


def test_consensus_requires_complete_aligned_species_predictions() -> None:
    pool = pd.DataFrame({"species": [0.0, 1.0, 0.5]})
    predictions = {str(i): np.tile([1.0, 3.0, 2.0], (7, 1)).T for i in range(5)}
    scores = consensus_rank(pool, predictions)
    assert scores[0] > scores[2] > scores[1]
    predictions["0"][1, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        consensus_rank(pool, predictions)


def test_adopted_selection_uses_membership_and_similarity_without_saved_allowlist(
    tmp_path: Path,
) -> None:
    pool = pd.DataFrame(
        dict(
            candidate_id=["a", "b", "c"],
            sequence=["AAAAAAAA", "CCCCCCCC", "DDDDDDDD"],
            raw_order=[0, 1, 2],
            rankmean=[0.9, 0.8, 0.7],
            embedding_cluster=[0, 1, 2],
            hard_reject=[False] * 3,
        )
    )
    result = select_adopted(pool, ["CCCCCCCC", "DDDDDDDD"], 1, set(), set())
    assert result.sequence.tolist() == ["CCCCCCCC"]
    with pytest.raises(ValueError, match="infeasible"):
        select_adopted(pool, ["CCCCCCCC"], 1, {"CCCCCCCC"}, set())
