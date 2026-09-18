import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, write_prediction_archive
from robust_apex_qd.evaluation.run import run_submission_evaluation
from robust_apex_qd.io.fasta import FastaRecord, write_fasta


def _small_run(path: Path) -> Path:
    path.mkdir()
    work = path / "work"
    work.mkdir()
    records = [
        FastaRecord(f"cand_{index}", sequence)
        for index, sequence in enumerate(
            ("ACDEFGHI", "KLMNPQRS", "TVWYACDE", "FGHIKLMN"),
            start=1,
        )
    ]
    write_fasta(records, path / "library.fasta")
    write_fasta(records[:3], path / "top.fasta")
    with (path / "ranking.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("rank", "candidate_id", "sequence", "final_score"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, record in enumerate(records[:3], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": record.header,
                    "sequence": record.sequence,
                    "final_score": 1 - rank / 10,
                }
            )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "library_count": 4,
                "top_count": 3,
                "manual_intervention": False,
            }
        )
    )
    with gzip.open(work / "candidates.csv.gz", "wt", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("candidate_id", "sequence", "raw_order", "valid"),
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "candidate_id": record.header,
                    "sequence": record.sequence,
                    "raw_order": index,
                    "valid": "True",
                }
            )
    with gzip.open(work / "candidate_physchem.csv.gz", "wt", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("candidate_id", "sequence", "physchem_ood", "hard_reject"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "candidate_id": record.header,
                    "sequence": record.sequence,
                    "physchem_ood": "0.2",
                    "hard_reject": "False",
                }
            )
    with gzip.open(work / "candidate_embedding_diagnostics.csv.gz", "wt", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("candidate_id", "sequence", "embedding_ood", "embedding_cluster"),
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "candidate_id": record.header,
                    "sequence": record.sequence,
                    "embedding_ood": "0.3",
                    "embedding_cluster": index,
                }
            )
    tensor = np.full((4, 8, 11), 8.0, dtype=np.float32)
    tensor[1] = 32.0
    write_prediction_archive(
        work / "apex_predictions.npz",
        [record.sequence for record in records],
        [f"model_{index}" for index in range(8)],
        APEX_PATHOGENS,
        tensor,
    )
    return path


def test_run_submission_evaluation_writes_all_report_formats(tmp_path: Path) -> None:
    run_dir = _small_run(tmp_path / "run")
    challenge = tmp_path / "challenge.fasta"
    challenge.write_text(">challenge\nAAAAAAAA\n")
    report_dir = tmp_path / "report"

    report = run_submission_evaluation(
        run_dir=run_dir,
        report_dir=report_dir,
        challenge_fasta=challenge,
        draws=5,
        sample_size=2,
        seed=42,
        include_seqme=False,
    )

    assert report.counts == {"library": 4, "top": 3}
    for name in (
        "evaluation.json",
        "library_metrics.csv",
        "top100_candidates.csv",
        "random25_draws.csv",
        "oracle_predictions.csv.gz",
        "summary.md",
        "summary.html",
    ):
        assert (report_dir / name).is_file()
    assert "official hidden aggregation" in (report_dir / "summary.md").read_text()


def test_required_oracle_fails_when_environment_is_missing(tmp_path: Path) -> None:
    run_dir = _small_run(tmp_path / "run")
    challenge = tmp_path / "challenge.fasta"
    challenge.write_text(">challenge\nAAAAAAAA\n")

    with pytest.raises(RuntimeError, match="prepare-evaluation-oracles"):
        run_submission_evaluation(
            run_dir=run_dir,
            report_dir=tmp_path / "report",
            challenge_fasta=challenge,
            draws=2,
            sample_size=2,
            include_seqme=False,
            oracle_dir=tmp_path / "missing-oracle",
            require_oracles=True,
        )
