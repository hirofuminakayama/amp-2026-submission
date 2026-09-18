import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 99),
        ("count", 100),
        ("min_length", 9),
        ("max_length", 30),
        ("filter_out", False),
        ("n_attempts", 2),
        ("softmax", False),
    ],
)
def test_hydramp_protocol_rejects_unregistered_settings(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("evaluate_research_generation")
    config = json.loads(Path("configs/research_generation.json").read_text())
    manifest: dict[str, object] = dict(
        seed=42,
        count=1000,
        min_length=10,
        max_length=25,
        filter_out=True,
        n_attempts=1,
        softmax=True,
    )
    module.validate_run_protocol("hydramp-s42", manifest, config)
    manifest[field] = value
    with pytest.raises(ValueError, match="protocol"):
        module.validate_run_protocol("hydramp-s42", manifest, config)


def test_generation_protocol_checks_identity_and_amp_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("evaluate_research_generation")
    config = json.loads(Path("configs/research_generation.json").read_text())
    job = module.jobs(config)["paired-s42"]
    module.validate_run_protocol("paired-s42", {"config": config, "job": job}, config)
    for name, manifest in [
        ("paired-s43", {"config": config, "job": job}),
        ("hydramp-s99", {}),
        ("paired-s42", {}),
    ]:
        with pytest.raises(ValueError, match="protocol"):
            module.validate_run_protocol(name, manifest, config)


@pytest.mark.parametrize("condition", ["developability", "hemopi2", "both"])
def test_safety_top_cannot_fill_from_outside_prefilter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("run_research_selection")
    root = tmp_path
    for name in ["prepare", "oracles", "top_pilot", "result", "frozen"]:
        (root / name).mkdir()
    sequences = ["ACDEFGHIKL", "MNPQRSTVWY"]
    pool = pd.DataFrame(
        dict(
            sequence=sequences,
            candidate_id=["a", "b"],
            valid=True,
            raw_order=[0, 1],
            B1=[-1.0, -2.0],
            length=10,
            embedding_cluster=[0, 1],
            hard_reject=False,
        )
    )
    pool.to_csv(root / "prepare/pool.csv.gz", index=False)
    (root / "prepare/L2.fasta").write_text(">a\nACDEFGHIKL\n>b\nMNPQRSTVWY\n")
    (root / "frozen/top.fasta").write_text(">a\nACDEFGHIKL\n")
    (root / "reference.fasta").write_text(">r\nAAAAAAAAAA\n")
    (root / "prepare/selection_manifest.json").write_text(json.dumps({"variants": []}))
    pd.DataFrame(
        dict(sequence=sequences, hemopi2_hc50_u_m=[200.0, 200.0], hard_filter_pass=True)
    ).to_csv(root / "oracles/oracle_predictions.csv.gz", index=False)
    pool.iloc[:1][["candidate_id", "sequence"]].to_csv(root / "oracles/prefilter.csv", index=False)
    (root / "top_pilot/similarity_cache.json").write_text(
        json.dumps({key: dict.fromkeys(sequences, 0.0) for key in ["challenge", "known"]})
    )
    config = dict(
        challenge_reference=str(root / "reference.fasta"),
        training_reference=str(root / "reference.fasta"),
        frozen_run=str(root / "frozen"),
        rankers=["B1"],
        filter_conditions=[condition],
        top_k=2,
        hc50_boundary_um=100.0,
        maximum_oracle_length=40,
        oracle_prefilter=1,
    )
    module.tops(config, root, root / "result")
    result = pd.read_csv(root / "result/selection_comparison.csv")
    assert result.status.tolist() == ["cannot_collect_top100"]
    assert result.external_filter_rejected_rows.tolist() == [1]
