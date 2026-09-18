from pathlib import Path

import pandas as pd

from robust_apex_qd.research.adoption import candidate_identity, common_coverage


def test_identity_ignores_headers_and_library_order_but_preserves_top_order(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "library.fasta").write_text(">a\nAAAAAAAA\n>b\nCCCCCCCC\n")
    (second / "library.fasta").write_text(">x\nCCCCCCCC\n>y\nAAAAAAAA\n")
    for path in [first, second]:
        (path / "top.fasta").write_text(">a\nAAAAAAAA\n>b\nCCCCCCCC\n")
    assert candidate_identity(first) == candidate_identity(second)
    (second / "top.fasta").write_text(">b\nCCCCCCCC\n>a\nAAAAAAAA\n")
    assert candidate_identity(first) != candidate_identity(second)


def test_common_coverage_removes_incomplete_metric_for_every_candidate() -> None:
    rows = pd.DataFrame({"safety": [1.0, 2.0], "hc50_complete_median": [10.0, float("nan")]})
    result, omitted = common_coverage(rows, ["safety", "hc50_complete_median"])
    assert omitted == ["hc50_complete_median"]
    assert result.hc50_complete_median.isna().all()
    assert result.safety.tolist() == [1.0, 2.0]
    assert rows.hc50_complete_median.notna().sum() == 1
