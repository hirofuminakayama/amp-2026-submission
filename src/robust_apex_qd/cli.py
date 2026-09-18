import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from robust_apex_qd.generation.sampler import LengthPolicy
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.pipeline import ROOT, PipelineOptions, run_pipeline
from robust_apex_qd.validation.compliance import validate_submission


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def _resolve_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _defaults(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    generation = config["generation"]
    runtime = config["runtime"]
    references = config["references"]
    ranking = config["ranking"]
    return {
        "output_dir": config["output_dir"],
        "raw_pool_size": generation["raw_pool_size"],
        "library_size": generation["library_size"],
        "top_k": 100,
        "batch_size": generation["batch_size"],
        "seed": config["seed"],
        "sampler": generation["sampler"],
        "device": runtime["device"],
        "checkpoint": generation["checkpoint"],
        "training_fasta": references["known_amp_fasta"],
        "challenge_fasta": references["challenge_fasta"],
        "length_policy": generation["length_policy"],
        "minimum_length": generation["min_generation_length"],
        "maximum_length": generation["max_generation_length"],
        "length_temperature": generation["length_temperature"],
        "official_similarity_max": ranking["official_challenge_similarity_max"],
        "selection_similarity_max": ranking["selection_challenge_similarity_max"],
        "cpu_threads": runtime["cpu_threads"],
    }


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-pool-size", type=int)
    parser.add_argument("--library-size", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampler", choices=("official", "deterministic"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--length-policy", choices=tuple(policy.value for policy in LengthPolicy))
    parser.add_argument("--show-defaults", action="store_true")


def _pipeline_options(arguments: argparse.Namespace, *, smoke: bool) -> PipelineOptions:
    config_path = arguments.config.resolve()
    defaults = _defaults(config_path)
    if smoke:
        defaults.update(
            {
                "output_dir": "smoke_output",
                "raw_pool_size": 320,
                "library_size": 256,
                "top_k": 20,
                "batch_size": 8,
                "sampler": "deterministic",
                "device": "cpu",
            }
        )
    for name in (
        "output_dir",
        "raw_pool_size",
        "library_size",
        "top_k",
        "batch_size",
        "seed",
        "sampler",
        "device",
        "length_policy",
    ):
        value = getattr(arguments, name)
        if value is not None:
            defaults[name] = value
    return PipelineOptions(
        config_path=config_path,
        output_dir=_resolve_root_path(str(defaults["output_dir"])),
        raw_pool_size=int(defaults["raw_pool_size"]),
        library_size=int(defaults["library_size"]),
        top_k=int(defaults["top_k"]),
        batch_size=int(defaults["batch_size"]),
        seed=int(defaults["seed"]),
        sampler=str(defaults["sampler"]),
        device=str(defaults["device"]),
        checkpoint=_resolve_root_path(str(defaults["checkpoint"])),
        training_fasta=_resolve_root_path(str(defaults["training_fasta"])),
        challenge_fasta=_resolve_root_path(str(defaults["challenge_fasta"])),
        length_policy=LengthPolicy(str(defaults["length_policy"])),
        minimum_length=int(defaults["minimum_length"]),
        maximum_length=int(defaults["maximum_length"]),
        length_temperature=float(defaults["length_temperature"]),
        official_similarity_max=float(defaults["official_similarity_max"]),
        selection_similarity_max=float(defaults["selection_similarity_max"]),
        cpu_threads=int(defaults["cpu_threads"]),
    )


def _run_pipeline_command(arguments: argparse.Namespace, *, smoke: bool) -> int:
    options = _pipeline_options(arguments, smoke=smoke)
    if arguments.show_defaults:
        print(
            json.dumps(
                {
                    "raw_pool_size": options.raw_pool_size,
                    "library_size": options.library_size,
                    "top_k": options.top_k,
                    "batch_size": options.batch_size,
                    "seed": options.seed,
                    "output_dir": str(options.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    output = run_pipeline(options)
    print(f"Validated output written to {output}")
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    references = set(read_fasta_sequences(_resolve_root_path(arguments.challenge_fasta)))
    report = validate_submission(
        arguments.output_dir.resolve(),
        references,
        library_size=arguments.library_size,
        top_k=arguments.top_k,
    )
    if report.is_valid:
        print("Local verification passed")
        return 0
    for issue in report.issues:
        print(f"{issue.reason.value}: {issue.message}", file=sys.stderr)
    return 1


def _placeholder(command: str) -> int:
    print(json.dumps({"command": command, "status": "reserved_for_later_phase"}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust APEX-QD pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    _add_pipeline_arguments(generate_parser)
    smoke_parser = subparsers.add_parser("smoke")
    _add_pipeline_arguments(smoke_parser)
    verify_parser = subparsers.add_parser("verify-local")
    verify_parser.add_argument("--output-dir", type=Path, default=ROOT / "generate")
    verify_parser.add_argument("--library-size", type=int, default=50_000)
    verify_parser.add_argument("--top-k", type=int, default=100)
    verify_parser.add_argument("--challenge-fasta", default="data/antibacterial.fasta")
    ranker_parser = subparsers.add_parser("evaluate-ranker")
    ranker_parser.add_argument("--config", type=Path, default=ROOT / "configs/candidate.yaml")
    ranker_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ranker_parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    library_parser = subparsers.add_parser("evaluate-library")
    library_parser.add_argument(
        "--variants",
        nargs="+",
        choices=("L0", "L1", "L2"),
        default=["L0", "L1", "L2"],
    )
    library_parser.add_argument("--seed", type=int, default=42)
    library_parser.add_argument("--subset-size", type=int, default=1000)
    submission_parser = subparsers.add_parser("evaluate-submission")
    submission_parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs/evaluation.yaml",
    )
    submission_parser.add_argument("--run-dir", type=Path)
    submission_parser.add_argument("--report-dir", type=Path)
    submission_parser.add_argument("--draws", type=int)
    submission_parser.add_argument("--sample-size", type=int)
    submission_parser.add_argument("--seed", type=int)
    submission_parser.add_argument("--seqme-subset-size", type=int)
    submission_parser.add_argument("--skip-seqme", action="store_true")
    submission_parser.add_argument("--hemopi2-predictions", type=Path)
    submission_parser.add_argument("--oracle-dir", type=Path)
    submission_parser.add_argument("--require-oracles", action="store_true")
    submission_parser.add_argument(
        "--ranking-config",
        type=Path,
        default=ROOT / "configs/final.yaml",
    )
    submission_parser.add_argument(
        "--challenge-fasta",
        type=Path,
        default=ROOT / "data/antibacterial.fasta",
    )
    submission_parser.add_argument(
        "--known-fasta",
        type=Path,
        default=ROOT / "data/training/training.fasta",
    )
    oracle_parser = subparsers.add_parser("prepare-evaluation-oracles")
    oracle_parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=ROOT / "work/oracles/hemopi2",
    )
    calibrator_parser = subparsers.add_parser("train-calibrator")
    calibrator_parser.add_argument("--config", type=Path, default=ROOT / "configs/candidate.yaml")
    calibrator_parser.add_argument(
        "--measurements",
        type=Path,
        default=ROOT / "experimental/mic.csv",
    )
    calibrator_parser.add_argument(
        "--apex-predictions",
        type=Path,
        default=ROOT / "work/calibration_apex_predictions.npz",
    )
    calibrator_parser.add_argument(
        "--oof-output",
        type=Path,
        default=ROOT / "reports/calibration_oof.csv",
    )
    calibrator_parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "reports/calibration_summary.json",
    )
    calibrator_parser.add_argument(
        "--artifact-output",
        type=Path,
        default=ROOT / "checkpoint/calibration.json",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if parsed.command == "generate":
        return _run_pipeline_command(parsed, smoke=False)
    if parsed.command == "smoke":
        return _run_pipeline_command(parsed, smoke=True)
    if parsed.command == "verify-local":
        return _verify(parsed)
    if parsed.command == "train-calibrator":
        from robust_apex_qd.calibration.run import run_calibration

        summary = run_calibration(
            config_path=parsed.config.resolve(),
            measurements_path=parsed.measurements.resolve(),
            apex_predictions_path=parsed.apex_predictions.resolve(),
            oof_path=parsed.oof_output.resolve(),
            summary_path=parsed.summary_output.resolve(),
            artifact_path=parsed.artifact_output.resolve(),
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if parsed.command == "evaluate-ranker":
        from robust_apex_qd.ranking.evaluate import run_ranker_evaluation

        report, bootstrap = run_ranker_evaluation(
            measurements_path=ROOT / "experimental/mic.csv",
            apex_predictions_path=ROOT / "work/calibration_apex_predictions.npz",
            calibration_path=ROOT / "checkpoint/calibration.json",
            physchem_reference_path=ROOT / "work/physchem_reference.json",
            reference_embeddings_path=ROOT / "work/reference_embeddings.npy",
            peptide_embeddings_path=ROOT / "work/calibration_peptide_embeddings.npy",
            report_path=ROOT / "reports/ranker_b0_b6.csv",
            bootstrap_path=ROOT / "reports/ranker_bootstrap.json",
            device=parsed.device,
            seed=int(_load_config(parsed.config.resolve())["seed"]),
            bootstrap_iterations=parsed.bootstrap_iterations,
        )
        print(
            json.dumps(
                {
                    "rankers": len(report),
                    "adopted_ranker": bootstrap["adoption"]["adopted_ranker"],
                },
                sort_keys=True,
            )
        )
        return 0
    if parsed.command == "evaluate-library":
        from robust_apex_qd.evaluation.seqme_eval import run_seqme_evaluation

        report, adoption = run_seqme_evaluation(
            variant_paths={
                variant: ROOT / f"work/library_{variant.lower()}/library.fasta"
                for variant in parsed.variants
            },
            candidates_path=ROOT / "work/candidates.csv.gz",
            candidate_embeddings_path=ROOT / "work/candidate_embeddings.npy",
            reference_fasta_path=ROOT / "data/training/training.fasta",
            reference_embeddings_path=ROOT / "work/reference_embeddings.npy",
            embedding_manifest_path=ROOT / "work/embedding_manifest.json",
            seed=parsed.seed,
            subset_size=parsed.subset_size,
            csv_path=ROOT / "reports/seqme_l0_l1_l2.csv",
            markdown_path=ROOT / "reports/seqme_l0_l1_l2.md",
            subset_ids_path=ROOT / "reports/seqme_subset_ids.txt",
            adoption_path=ROOT / "reports/seqme_adoption.json",
        )
        print(
            json.dumps(
                {
                    "variants": len(report),
                    "adopted_variant": adoption["adopted_variant"],
                },
                sort_keys=True,
            )
        )
        return 0
    if parsed.command == "evaluate-submission":
        from robust_apex_qd.evaluation.run import run_submission_evaluation

        evaluation_config = _load_config(parsed.evaluation_config.resolve())
        random_config = evaluation_config.get("random25", {})
        seqme_config = evaluation_config.get("seqme", {})
        oracle_config = evaluation_config.get("oracles", {})
        if not all(
            isinstance(value, dict) for value in (random_config, seqme_config, oracle_config)
        ):
            raise ValueError("Evaluation config sections must be mappings")
        run_dir = (
            parsed.run_dir.resolve()
            if parsed.run_dir is not None
            else _resolve_root_path(str(evaluation_config["run_dir"])).resolve()
        )
        report_dir = (
            parsed.report_dir.resolve()
            if parsed.report_dir is not None
            else _resolve_root_path(str(evaluation_config["report_dir"])).resolve()
        )
        oracle_dir = (
            parsed.oracle_dir.resolve()
            if parsed.oracle_dir is not None
            else _resolve_root_path(str(oracle_config["hemopi2_dir"])).resolve()
        )
        report = run_submission_evaluation(
            run_dir=run_dir,
            report_dir=report_dir,
            challenge_fasta=parsed.challenge_fasta.resolve(),
            draws=(parsed.draws if parsed.draws is not None else int(random_config["draws"])),
            sample_size=(
                parsed.sample_size
                if parsed.sample_size is not None
                else int(random_config["sample_size"])
            ),
            seed=parsed.seed if parsed.seed is not None else int(evaluation_config["seed"]),
            include_seqme=not parsed.skip_seqme and bool(seqme_config.get("enabled", True)),
            hemopi2_predictions_path=(
                parsed.hemopi2_predictions.resolve()
                if parsed.hemopi2_predictions is not None
                else None
            ),
            oracle_dir=oracle_dir,
            require_oracles=parsed.require_oracles or bool(oracle_config.get("require", False)),
            config_path=parsed.ranking_config.resolve(),
            known_fasta=parsed.known_fasta.resolve(),
            seqme_reference_fasta=parsed.known_fasta.resolve(),
            seqme_subset_size=(
                parsed.seqme_subset_size
                if parsed.seqme_subset_size is not None
                else int(seqme_config["subset_size"])
            ),
        )
        print(
            json.dumps(
                {
                    "library_count": report.counts["library"],
                    "top_count": report.counts["top"],
                    "report_dir": str(report_dir),
                    "hemopi2": report.coverage["hemopi2"].status,
                    "rerank": report.rerank["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    if parsed.command == "prepare-evaluation-oracles":
        from robust_apex_qd.evaluation.oracles import prepare_hemopi2_environment

        manifest = prepare_hemopi2_environment(parsed.oracle_dir.resolve())
        print(json.dumps(manifest, sort_keys=True))
        return 0
    return _placeholder(parsed.command)


def _entry(command: str) -> None:
    raise SystemExit(main((command, *sys.argv[1:])))


def generate_main() -> None:
    _entry("generate")


def generate_broad_spectrum_main() -> None:
    raise SystemExit(
        main(
            (
                "generate",
                "--output-dir",
                str(ROOT / "generate_broad_spectrum"),
                *sys.argv[1:],
            )
        )
    )


def smoke_main() -> None:
    _entry("smoke")


def verify_local_main() -> None:
    _entry("verify-local")


def evaluate_ranker_main() -> None:
    _entry("evaluate-ranker")


def evaluate_library_main() -> None:
    _entry("evaluate-library")


def evaluate_submission_main() -> None:
    _entry("evaluate-submission")


def prepare_evaluation_oracles_main() -> None:
    _entry("prepare-evaluation-oracles")


def train_calibrator_main() -> None:
    _entry("train-calibrator")


if __name__ == "__main__":
    raise SystemExit(main())
