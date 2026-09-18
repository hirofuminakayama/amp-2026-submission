import csv
import gzip
import html
import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from robust_apex_qd.apex.ensemble import (
    ApexAggregates,
    aggregate_predictions,
    load_prediction_archive,
)
from robust_apex_qd.evaluation.developability import DevelopabilityResult, evaluate_developability
from robust_apex_qd.evaluation.models import (
    CandidateEvaluation,
    EvaluationCoverage,
    EvaluationReport,
    HemoPI2Prediction,
    NumericSummary,
)
from robust_apex_qd.evaluation.oracles import (
    hemopi2_environment_status,
    load_hemopi2_predictions,
    run_hemopi2,
    write_hemopi2_predictions,
)
from robust_apex_qd.evaluation.random25 import simulate_random_draws
from robust_apex_qd.evaluation.submission import pathogen_group_values, summarize_numeric
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import FastaRecord, read_fasta, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.selection.top import (
    TopCandidate,
    select_top_with_fallback,
    top_selection_config_from_mapping,
)
from robust_apex_qd.validation.compliance import validate_submission

REPORT_FILES = (
    "library_metrics.csv",
    "top100_candidates.csv",
    "random25_draws.csv",
    "oracle_predictions.csv.gz",
    "rerank_comparison.csv",
    "summary.md",
    "summary.html",
)


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as file:
            return list(csv.DictReader(file, delimiter=delimiter))
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def _unique_by(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def _require_alignment(
    records: Sequence[FastaRecord],
    rows_by_id: Mapping[str, Mapping[str, str]],
    *,
    source: str,
) -> None:
    for record in records:
        candidate_id = str(record.header)
        sequence = str(record.sequence)
        if candidate_id not in rows_by_id:
            raise ValueError(f"{source} is missing candidate_id {candidate_id}")
        if (
            "sequence" in rows_by_id[candidate_id]
            and rows_by_id[candidate_id]["sequence"] != sequence
        ):
            raise ValueError(f"{source} sequence differs for {candidate_id}")


def _write_gzip_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_file,
    ):
        writer = csv.DictWriter(text_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(
    scope: str,
    summaries: Mapping[str, NumericSummary],
) -> list[dict[str, object]]:
    return [
        {"scope": scope, "metric": name, **summary.model_dump()}
        for name, summary in summaries.items()
    ]


def _format_metric_table(
    library_metrics: Mapping[str, NumericSummary],
    pathogen_groups: Mapping[str, NumericSummary],
    top_metrics: Mapping[str, NumericSummary],
) -> str:
    lines = [
        "| Scope | Metric | Mean | P05 | P50 | P95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope, metrics in (
        ("Library", library_metrics),
        ("Library", pathogen_groups),
        ("Top", top_metrics),
    ):
        for name, summary in metrics.items():
            lines.append(
                f"| {scope} | {name} | {summary.mean:.6g} | {summary.p05:.6g} | "
                f"{summary.p50:.6g} | {summary.p95:.6g} |"
            )
    return "\n".join(lines)


def _format_random_table(report: EvaluationReport) -> str:
    lines = [
        "| Variant | Metric | Mean | P05 | P50 | P95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, metrics in report.random25.items():
        for name, summary in metrics.items():
            lines.append(
                f"| {variant} | {name} | {summary.mean:.6g} | {summary.p05:.6g} | "
                f"{summary.p50:.6g} | {summary.p95:.6g} |"
            )
    return "\n".join(lines)


def _render_markdown(report: EvaluationReport) -> str:
    table = _format_metric_table(
        report.library_metrics,
        report.pathogen_groups,
        report.top_metrics,
    )
    coverage = "\n".join(
        f"- `{name}`: **{item.status}** — {item.detail}" for name, item in report.coverage.items()
    )
    limitations = "\n".join(f"- {value}" for value in report.limitations)
    random_table = _format_random_table(report)
    return (
        "# Local submission evaluation\n\n"
        f"Run: `{report.run_dir}`\n\n"
        f"Library: {report.counts['library']:,}; Top: {report.counts['top']:,}.\n\n"
        "## Evidence coverage\n\n"
        f"{coverage}\n\n"
        "## Numeric summary\n\n"
        f"{table}\n\n"
        "## Random-25 proxy\n\n"
        f"{random_table}\n\n"
        "## Oracle-filtered Top\n\n"
        f"- Status: **{report.rerank['status']}**\n"
        f"- Detail: {report.rerank['detail']}\n"
        f"- Recommended submission Top: **{report.rerank['recommended_submission_top']}**\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def _render_html(markdown_text: str, report: EvaluationReport) -> str:
    coverage_rows = "".join(
        "<tr>"
        f"<th>{html.escape(name)}</th><td>{html.escape(item.status)}</td>"
        f"<td>{html.escape(item.detail)}</td></tr>"
        for name, item in report.coverage.items()
    )
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(scope)}</td><th>{html.escape(name)}</th>"
        f"<td>{summary.mean:.6g}</td><td>{summary.p05:.6g}</td>"
        f"<td>{summary.p50:.6g}</td><td>{summary.p95:.6g}</td></tr>"
        for scope, metrics in (
            ("Library", report.library_metrics),
            ("Library", report.pathogen_groups),
            ("Top", report.top_metrics),
        )
        for name, summary in metrics.items()
    )
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in report.limitations)
    random_rows = "".join(
        "<tr>"
        f"<td>{html.escape(variant)}</td><th>{html.escape(name)}</th>"
        f"<td>{summary.mean:.6g}</td><td>{summary.p05:.6g}</td>"
        f"<td>{summary.p50:.6g}</td><td>{summary.p95:.6g}</td></tr>"
        for variant, metrics in report.random25.items()
        for name, summary in metrics.items()
    )
    status = html.escape(str(report.rerank["status"]))
    detail = html.escape(str(report.rerank["detail"]))
    recommendation = html.escape(str(report.rerank["recommended_submission_top"]))
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width">',
            "<title>Local submission evaluation</title><style>",
            "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;",
            "padding:2rem;color:#172033}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0}",
            "th,td{border:1px solid #ccd3df;padding:.45rem;text-align:left}",
            "thead{background:#eef3f8}code{background:#eef3f8;padding:.1rem .3rem}",
            ".note{border-left:4px solid #c47f00;padding:.7rem;background:#fff7e6}",
            "</style></head><body><h1>Local submission evaluation</h1>",
            f"<p><code>{html.escape(report.run_dir)}</code> — library "
            f"{report.counts['library']:,}, Top {report.counts['top']:,}</p>",
            '<p class="note">These are local proxy metrics. They do not reproduce the '
            "official hidden aggregation or wet-lab results.</p>",
            "<h2>Evidence coverage</h2><table><thead><tr>",
            "<th>Source</th><th>Status</th><th>Detail</th></tr></thead>",
            f"<tbody>{coverage_rows}</tbody></table>",
            "<h2>Numeric summary</h2><table><thead><tr><th>Scope</th>",
            "<th>Metric</th><th>Mean</th><th>P05</th><th>P50</th><th>P95</th>",
            f"</tr></thead><tbody>{metric_rows}</tbody></table>",
            "<h2>Random-25 proxy</h2><table><thead><tr><th>Variant</th>",
            "<th>Metric</th><th>Mean</th><th>P05</th><th>P50</th><th>P95</th>",
            f"</tr></thead><tbody>{random_rows}</tbody></table>",
            f"<h2>Oracle-filtered Top</h2><p>Status: <strong>{status}</strong>. "
            f"{detail}</p><p>Recommended submission Top: "
            f"<strong>{recommendation}</strong>.</p>",
            f"<h2>Limitations</h2><ul>{limitations}</ul>",
            "<details><summary>Markdown source</summary>",
            f"<pre>{html.escape(markdown_text)}</pre></details></body></html>",
            "",
        )
    )


def _publish(temporary: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    try:
        temporary.rename(target)
    except OSError:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _candidate_evaluation(
    *,
    rank: int,
    record: FastaRecord,
    final_score: float,
    index: int,
    aggregates: ApexAggregates,
    consensus16: np.ndarray,
    group_values: Mapping[str, np.ndarray],
    embeddings_by_id: Mapping[str, Mapping[str, str]],
    developability: Mapping[str, DevelopabilityResult],
    hemo_predictions: Mapping[str, HemoPI2Prediction],
) -> CandidateEvaluation:
    hemo = hemo_predictions.get(record.header)
    broad_mic = float(aggregates.official_broad_mean_mic_u_m[index])
    return CandidateEvaluation(
        rank=rank,
        candidate_id=record.header,
        sequence=record.sequence,
        final_score=final_score,
        apex_vote16=float(aggregates.vote16[index]),
        apex_consensus16=float(consensus16[index]),
        gram_negative_success16=float(group_values["gram_negative"][index]),
        gram_positive_success16=float(group_values["gram_positive"][index]),
        mdr_proxy=float(group_values["mdr_proxy"][index]),
        apex_broad_mean_mic_u_m=broad_mic,
        apex_model_disagreement=float(aggregates.model_disagreement_mad_log2[index]),
        embedding_cluster=int(embeddings_by_id[record.header]["embedding_cluster"]),
        pep_hard_filter_pass=developability[record.header].hard_filter_pass,
        spps_difficulty_score=developability[record.header].spps_difficulty_score,
        hemopi2_hc50_u_m=hemo.hc50_u_m if hemo else None,
        hemopi2_hemolytic=hemo.hemolytic if hemo else None,
        selectivity_proxy=hemo.hc50_u_m / broad_mic if hemo else None,
    )


def _select_oracle_filtered_top(
    *,
    library_records: Sequence[FastaRecord],
    b1_scores: np.ndarray,
    aggregates: ApexAggregates,
    physchem_by_id: Mapping[str, Mapping[str, str]],
    embeddings_by_id: Mapping[str, Mapping[str, str]],
    developability: Mapping[str, DevelopabilityResult],
    hemo_predictions: Mapping[str, HemoPI2Prediction],
    challenge_sequences: Sequence[str],
    known_sequences: Sequence[str],
    ranking_config: Mapping[str, object],
    top_k: int,
) -> tuple[FastaRecord, ...]:
    candidates = tuple(
        TopCandidate(
            candidate_id=record.header,
            sequence=record.sequence,
            raw_order=int(index),
            final_score=float(b1_scores[index]),
            embedding_cluster=int(embeddings_by_id[record.header]["embedding_cluster"]),
            physchem_hard_reject=physchem_by_id[record.header]["hard_reject"] == "True",
            external_hard_reject=(
                not developability[record.header].hard_filter_pass
                or record.header not in hemo_predictions
                or hemo_predictions[record.header].hemolytic
            ),
            median_log2_mic=float(aggregates.median_log2_mic[index]),
        )
        for index, record in enumerate(library_records)
    )
    result = select_top_with_fallback(
        candidates,
        challenge_references=challenge_sequences,
        known_references=known_sequences,
        top_k=top_k,
        config=top_selection_config_from_mapping(ranking_config),
    )
    return tuple(
        FastaRecord(selected.candidate.candidate_id, selected.candidate.sequence)
        for selected in result.selected
    )


def run_submission_evaluation(
    *,
    run_dir: Path,
    report_dir: Path,
    challenge_fasta: Path,
    draws: int = 10_000,
    sample_size: int = 25,
    seed: int = 42,
    include_seqme: bool = True,
    hemopi2_predictions_path: Path | None = None,
    oracle_dir: Path | None = None,
    require_oracles: bool = False,
    config_path: Path | None = None,
    known_fasta: Path | None = None,
    seqme_reference_fasta: Path | None = None,
    seqme_subset_size: int = 1_000,
) -> EvaluationReport:
    source = run_dir.resolve()
    target = report_dir.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    library_count = int(manifest["library_count"])
    top_count = int(manifest["top_count"])
    references = set(read_fasta_sequences(challenge_fasta.resolve()))
    validation = validate_submission(
        source,
        references,
        library_size=library_count,
        top_k=top_count,
    )
    if not validation.is_valid:
        messages = "; ".join(issue.message for issue in validation.issues)
        raise ValueError(f"Run directory is not a valid submission: {messages}")
    library_records = read_fasta(source / "library.fasta")
    top_records = read_fasta(source / "top.fasta")
    ranking_rows = _read_csv(source / "ranking.tsv", delimiter="\t")
    ranking_by_id = _unique_by(ranking_rows, "candidate_id")
    _require_alignment(top_records, ranking_by_id, source="ranking.tsv")

    work = source / "work"
    candidates_by_id = _unique_by(_read_csv(work / "candidates.csv.gz"), "candidate_id")
    physchem_by_id = _unique_by(_read_csv(work / "candidate_physchem.csv.gz"), "candidate_id")
    embeddings_by_id = _unique_by(
        _read_csv(work / "candidate_embedding_diagnostics.csv.gz"),
        "candidate_id",
    )
    _require_alignment(library_records, candidates_by_id, source="candidates.csv.gz")
    _require_alignment(library_records, physchem_by_id, source="candidate_physchem.csv.gz")
    _require_alignment(
        library_records,
        embeddings_by_id,
        source="candidate_embedding_diagnostics.csv.gz",
    )

    archive = load_prediction_archive(work / "apex_predictions.npz")
    row_by_sequence = {sequence: index for index, sequence in enumerate(archive.sequences)}
    if len(row_by_sequence) != len(archive.sequences):
        raise ValueError("APEX archive sequences must be unique")
    missing_apex = [
        record.sequence for record in library_records if record.sequence not in row_by_sequence
    ]
    if missing_apex:
        raise ValueError(f"APEX archive is missing library sequence {missing_apex[0]}")
    library_indices = np.asarray(
        [row_by_sequence[record.sequence] for record in library_records],
        dtype=np.int64,
    )
    library_tensor = archive.mic_u_m[library_indices]
    aggregates = aggregate_predictions(library_tensor)
    group_values = pathogen_group_values(aggregates.pathogen_success16)
    consensus16 = (aggregates.official_pathogen_mean_mic_u_m <= 16).mean(axis=1)

    developability: dict[str, DevelopabilityResult] = {
        record.header: evaluate_developability(record.sequence) for record in library_records
    }
    b1_scores = percentile_score(aggregates.median_log2_mic, higher_is_better=False)
    ranking_config: Mapping[str, object] | None = None
    known_sequences: tuple[str, ...] = ()
    if config_path is not None and known_fasta is not None:
        loaded_config = yaml.safe_load(config_path.resolve().read_text())
        if not isinstance(loaded_config, dict) or not isinstance(
            loaded_config.get("ranking"), dict
        ):
            raise ValueError("Evaluation ranking config must contain a ranking mapping")
        ranking_config = loaded_config["ranking"]
        known_sequences = tuple(read_fasta_sequences(known_fasta.resolve()))
    hemo_predictions: dict[str, HemoPI2Prediction] = {}
    preselected_reranked_records: tuple[FastaRecord, ...] = ()
    hemo_detail = "No HemoPI2 prediction file or prepared oracle environment was supplied."
    if hemopi2_predictions_path is not None:
        expected = {record.header: record.sequence for record in library_records}
        hemo_predictions = load_hemopi2_predictions(
            hemopi2_predictions_path,
            expected,
            require_complete=False,
        )
        hemo_detail = "HemoPI2 predictions were loaded from a supplied CSV."
    elif oracle_dir is not None:
        ready, hemo_detail = hemopi2_environment_status(oracle_dir.resolve())
        if ready:
            ordered_indices = np.argsort(-b1_scores, kind="stable")[: min(10_000, library_count)]
            cache_path = (
                oracle_dir.resolve() / "cache" / f"{file_sha256(source / 'library.fasta')}.csv"
            )
            expected = {record.header: record.sequence for record in library_records}
            if cache_path.is_file():
                hemo_predictions = load_hemopi2_predictions(
                    cache_path,
                    expected,
                    require_complete=False,
                )
            try:
                for batch_start in range(0, len(ordered_indices), 500):
                    batch_indices = ordered_indices[batch_start : batch_start + 500]
                    oracle_sequences = {
                        library_records[int(index)].header: library_records[int(index)].sequence
                        for index in batch_indices
                        if len(library_records[int(index)].sequence) <= 40
                        and library_records[int(index)].header not in hemo_predictions
                    }
                    if oracle_sequences:
                        hemo_predictions.update(run_hemopi2(oracle_dir.resolve(), oracle_sequences))
                        write_hemopi2_predictions(cache_path, hemo_predictions)
                    if ranking_config is None:
                        continue
                    try:
                        preselected_reranked_records = _select_oracle_filtered_top(
                            library_records=library_records,
                            b1_scores=b1_scores,
                            aggregates=aggregates,
                            physchem_by_id=physchem_by_id,
                            embeddings_by_id=embeddings_by_id,
                            developability=developability,
                            hemo_predictions=hemo_predictions,
                            challenge_sequences=tuple(references),
                            known_sequences=known_sequences,
                            ranking_config=ranking_config,
                            top_k=top_count,
                        )
                        break
                    except RuntimeError:
                        continue
                hemo_detail = (
                    f"HemoPI2 regression covered {len(hemo_predictions)} B1-prefilter rows; "
                    "HC50 <100 µM is the upstream hemolytic boundary."
                )
            except (OSError, RuntimeError, ValueError) as error:
                hemo_detail = f"HemoPI2 execution failed: {error}"
                if require_oracles:
                    raise
    if require_oracles and not hemo_predictions:
        raise RuntimeError(hemo_detail)

    library_metrics = {
        "apex_vote16": summarize_numeric(aggregates.vote16),
        "apex_consensus16": summarize_numeric(consensus16),
        "apex_broad_mean_mic_u_m": summarize_numeric(aggregates.official_broad_mean_mic_u_m),
        "apex_median_log2_mic": summarize_numeric(aggregates.median_log2_mic),
        "apex_q90_log2_mic": summarize_numeric(aggregates.q90_log2_mic),
        "apex_worst3_log2_mic": summarize_numeric(aggregates.worst3_log2_mic),
        "apex_model_disagreement": summarize_numeric(aggregates.model_disagreement_mad_log2),
        "physchem_ood": summarize_numeric(
            np.asarray(
                [float(physchem_by_id[record.header]["physchem_ood"]) for record in library_records]
            )
        ),
        "embedding_ood": summarize_numeric(
            np.asarray(
                [
                    float(embeddings_by_id[record.header]["embedding_ood"])
                    for record in library_records
                ]
            )
        ),
        "developability_score": summarize_numeric(
            np.asarray(
                [developability[record.header].developability_score for record in library_records]
            )
        ),
        "spps_difficulty_score": summarize_numeric(
            np.asarray(
                [developability[record.header].spps_difficulty_score for record in library_records]
            )
        ),
    }
    pathogen_groups = {name: summarize_numeric(values) for name, values in group_values.items()}

    library_position = {record.sequence: index for index, record in enumerate(library_records)}
    top_candidates: list[CandidateEvaluation] = []
    for rank, record in enumerate(top_records, start=1):
        index = library_position[record.sequence]
        top_candidates.append(
            _candidate_evaluation(
                rank=rank,
                record=record,
                final_score=float(ranking_by_id[record.header]["final_score"]),
                index=index,
                aggregates=aggregates,
                consensus16=consensus16,
                group_values=group_values,
                embeddings_by_id=embeddings_by_id,
                developability=developability,
                hemo_predictions=hemo_predictions,
            )
        )
    random_result = simulate_random_draws(
        tuple(top_candidates),
        draws=draws,
        sample_size=sample_size,
        seed=seed,
        variant="B1",
    )
    top_metrics = {
        "apex_vote16": summarize_numeric(np.asarray([row.apex_vote16 for row in top_candidates])),
        "apex_consensus16": summarize_numeric(
            np.asarray([row.apex_consensus16 for row in top_candidates])
        ),
        "gram_negative_success16": summarize_numeric(
            np.asarray([row.gram_negative_success16 for row in top_candidates])
        ),
        "gram_positive_success16": summarize_numeric(
            np.asarray([row.gram_positive_success16 for row in top_candidates])
        ),
        "mdr_proxy": summarize_numeric(np.asarray([row.mdr_proxy for row in top_candidates])),
    }
    seqme_metrics: dict[str, float] = {}
    seqme_detail = "Disabled by CLI option."
    seqme_status = "skipped"
    if include_seqme:
        required_seqme_paths = (
            work / "candidate_embeddings.npy",
            work / "reference_embeddings.npy",
            work / "embedding_manifest.json",
        )
        if (
            all(path.is_file() for path in required_seqme_paths)
            and seqme_reference_fasta is not None
        ):
            try:
                from robust_apex_qd.evaluation.seqme_eval import evaluate_single_library

                seqme_metrics, provenance = evaluate_single_library(
                    library_path=source / "library.fasta",
                    candidates_path=work / "candidates.csv.gz",
                    candidate_embeddings_path=work / "candidate_embeddings.npy",
                    reference_fasta_path=seqme_reference_fasta.resolve(),
                    reference_embeddings_path=work / "reference_embeddings.npy",
                    embedding_manifest_path=work / "embedding_manifest.json",
                    seed=seed,
                    subset_size=seqme_subset_size,
                )
                seqme_status = "complete"
                seqme_detail = (
                    f"Exact count/uniqueness/novelty plus sampled metrics on "
                    f"{provenance['subset_count']} fixed rows."
                )
            except ImportError as error:
                seqme_status = "unavailable"
                seqme_detail = f"seqme optional dependency is unavailable: {error}"
        else:
            seqme_status = "unavailable"
            seqme_detail = "Run-scoped embeddings or the seqme reference FASTA are missing."

    reranked_records: tuple[FastaRecord, ...] = ()
    reranked_candidates: list[CandidateEvaluation] = []
    rerank_detail = "HemoPI2 predictions are unavailable."
    if hemo_predictions:
        if ranking_config is None:
            rerank_detail = "Ranking config or known-AMP FASTA was not supplied."
        else:
            try:
                reranked_records = preselected_reranked_records or _select_oracle_filtered_top(
                    library_records=library_records,
                    b1_scores=b1_scores,
                    aggregates=aggregates,
                    physchem_by_id=physchem_by_id,
                    embeddings_by_id=embeddings_by_id,
                    developability=developability,
                    hemo_predictions=hemo_predictions,
                    challenge_sequences=tuple(references),
                    known_sequences=known_sequences,
                    ranking_config=ranking_config,
                    top_k=top_count,
                )
                for rank, record in enumerate(reranked_records, start=1):
                    index = library_position[record.sequence]
                    reranked_candidates.append(
                        _candidate_evaluation(
                            rank=rank,
                            record=record,
                            final_score=float(b1_scores[index]),
                            index=index,
                            aggregates=aggregates,
                            consensus16=consensus16,
                            group_values=group_values,
                            embeddings_by_id=embeddings_by_id,
                            developability=developability,
                            hemo_predictions=hemo_predictions,
                        )
                    )
                rerank_detail = "Oracle hard filters passed and B1 order was preserved."
            except RuntimeError as error:
                rerank_detail = (
                    f"Oracle-filtered selector could not collect Top-{top_count}: {error}"
                )
                if require_oracles:
                    raise

    random_results = {"B1": random_result}
    if reranked_candidates:
        random_results["oracle_filtered"] = simulate_random_draws(
            tuple(reranked_candidates),
            draws=draws,
            sample_size=sample_size,
            seed=seed,
            variant="oracle_filtered",
        )
    input_paths = (
        source / "library.fasta",
        source / "top.fasta",
        source / "ranking.tsv",
        source / "manifest.json",
        work / "candidates.csv.gz",
        work / "candidate_physchem.csv.gz",
        work / "candidate_embedding_diagnostics.csv.gz",
        work / "apex_predictions.npz",
    )
    coverage = {
        "compliance": EvaluationCoverage(
            status="complete",
            row_count=library_count,
            detail="Local submission contract and challenge-reference checks passed.",
        ),
        "apex_11_pathogen_proxy": EvaluationCoverage(
            status="complete",
            row_count=library_count,
            detail="Eight-model APEX tensor; proxy, not the hidden wet-lab panel.",
        ),
        "developability_proxy": EvaluationCoverage(
            status="complete",
            row_count=library_count,
            detail="Transparent sequence-level PepPredictor v1.2.5-compatible rules.",
        ),
        "hemopi2": EvaluationCoverage(
            status="complete" if hemo_predictions else "unavailable",
            row_count=len(hemo_predictions),
            detail=hemo_detail,
        ),
        "seqme": EvaluationCoverage(
            status=seqme_status,
            row_count=library_count if seqme_status == "complete" else 0,
            detail=seqme_detail,
        ),
    }
    limitations = (
        "The official hidden aggregation weights and score cannot be reproduced locally.",
        "The competition's random 25 selections and 20-strain wet-lab measurements are "
        "unavailable.",
        "APEX, HemoPI2, developability, HC50/MIC selectivity, and Random-25 values are "
        "computational proxies.",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        metric_rows = [
            *_summary_rows("library", library_metrics),
            *_summary_rows("library_pathogen_group", pathogen_groups),
            *_summary_rows("top", top_metrics),
        ]
        with (temporary / "library_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(metric_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(metric_rows)
        top_rows = [row.model_dump(mode="json") for row in top_candidates]
        with (temporary / "top100_candidates.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(top_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(top_rows)
        draw_rows = []
        for result in random_results.values():
            for row in result.rows:
                payload = row.model_dump(mode="json")
                payload["selected_ranks"] = ";".join(str(value) for value in row.selected_ranks)
                draw_rows.append(payload)
        with (temporary / "random25_draws.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(draw_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(draw_rows)
        oracle_rows = []
        for record in library_records:
            result = developability[record.header]
            hemo = hemo_predictions.get(record.header)
            oracle_rows.append(
                {
                    "candidate_id": record.header,
                    "sequence": record.sequence,
                    **result.model_dump(mode="json", exclude={"sequence"}),
                    "hard_filter_reasons": ";".join(result.hard_filter_reasons),
                    "hemopi2_hc50_u_m": hemo.hc50_u_m if hemo else "",
                    "hemopi2_hemolytic": hemo.hemolytic if hemo else "",
                }
            )
        _write_gzip_csv(
            temporary / "oracle_predictions.csv.gz",
            tuple(oracle_rows[0]),
            oracle_rows,
        )
        current_rank = {record.header: rank for rank, record in enumerate(top_records, start=1)}
        reranked_rank = {
            record.header: rank for rank, record in enumerate(reranked_records, start=1)
        }
        comparison_rows = [
            {
                "candidate_id": record.header,
                "sequence": record.sequence,
                "current_rank": current_rank.get(record.header, ""),
                "reranked_rank": reranked_rank.get(record.header, ""),
                "in_current_top": record.header in current_rank,
                "in_reranked_top": record.header in reranked_rank,
                "pep_hard_filter_pass": developability[record.header].hard_filter_pass,
                "hemopi2_hemolytic": (
                    hemo_predictions[record.header].hemolytic
                    if record.header in hemo_predictions
                    else ""
                ),
            }
            for record in library_records
            if record.header in current_rank or record.header in reranked_rank
        ]
        with (temporary / "rerank_comparison.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=tuple(comparison_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(comparison_rows)
        if reranked_records:
            write_fasta(reranked_records, temporary / "reranked_top.fasta")
            reranked_rows = [row.model_dump(mode="json") for row in reranked_candidates]
            with (temporary / "reranked_ranking.tsv").open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=tuple(reranked_rows[0]),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(reranked_rows)
            rerank_manifest = {
                "schema_version": 1,
                "source_ranker": "B1",
                "top_count": len(reranked_records),
                "replaces_submission_top": False,
                "hemopi2_hc50_boundary_u_m": 100.0,
                "pep_hard_filters": True,
                "output_sha256": {
                    name: file_sha256(temporary / name)
                    for name in ("reranked_top.fasta", "reranked_ranking.tsv")
                },
            }
            (temporary / "rerank_manifest.json").write_text(
                json.dumps(rerank_manifest, indent=2, sort_keys=True) + "\n"
            )
        provisional = EvaluationReport(
            run_dir=str(source),
            counts={"library": library_count, "top": top_count},
            input_sha256={
                path.relative_to(source).as_posix(): file_sha256(path) for path in input_paths
            },
            compliance={"valid": True, "issue_counts": {}},
            coverage=coverage,
            library_metrics=library_metrics,
            pathogen_groups=pathogen_groups,
            seqme_metrics=seqme_metrics,
            top_metrics=top_metrics,
            random25={name: result.summary for name, result in random_results.items()},
            rerank={
                "status": "complete" if reranked_records else "unavailable",
                "detail": rerank_detail,
                "top_count": len(reranked_records),
                "overlap_with_current": len(set(current_rank) & set(reranked_rank)),
                "replaces_submission_top": False,
                "adoption_gate_passed": False,
                "recommended_submission_top": "B1",
                "top_metrics": (
                    {
                        "apex_vote16": summarize_numeric(
                            np.asarray([row.apex_vote16 for row in reranked_candidates])
                        ).model_dump(mode="json"),
                        "mdr_proxy": summarize_numeric(
                            np.asarray([row.mdr_proxy for row in reranked_candidates])
                        ).model_dump(mode="json"),
                    }
                    if reranked_candidates
                    else {}
                ),
            },
            limitations=limitations,
            output_sha256={},
        )
        markdown = _render_markdown(provisional)
        (temporary / "summary.md").write_text(markdown)
        (temporary / "summary.html").write_text(_render_html(markdown, provisional))
        output_sha256 = {
            path.name: file_sha256(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        report = provisional.model_copy(update={"output_sha256": output_sha256})
        (temporary / "evaluation.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _publish(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
