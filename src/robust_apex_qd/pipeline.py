import csv
import gc
import gzip
import hashlib
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from robust_apex_qd.generation.sampler import (
    Candidate,
    LengthPolicy,
    SamplingBackend,
    build_length_quotas,
    generate_candidates,
    make_official_backend,
    set_process_determinism,
    write_candidates_csv,
)
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.validation.compliance import (
    OFFICIAL_CHALLENGE_SIMILARITY_MAX,
    passes_selection_similarity,
    require_valid_submission,
)
from robust_apex_qd.validation.inference_assets import prepare_default_encoders

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    config_path: Path
    output_dir: Path
    raw_pool_size: int
    library_size: int
    top_k: int
    batch_size: int
    seed: int
    sampler: str
    device: str
    checkpoint: Path
    training_fasta: Path
    challenge_fasta: Path
    length_policy: LengthPolicy
    minimum_length: int
    maximum_length: int
    length_temperature: float
    official_similarity_max: float
    selection_similarity_max: float
    cpu_threads: int

    @classmethod
    def smoke(
        cls,
        *,
        output_dir: Path,
        raw_pool_size: int = 320,
        library_size: int = 256,
        top_k: int = 20,
        batch_size: int = 8,
        seed: int = 42,
    ) -> "PipelineOptions":
        return cls(
            config_path=ROOT / "configs/candidate.yaml",
            output_dir=output_dir,
            raw_pool_size=raw_pool_size,
            library_size=library_size,
            top_k=top_k,
            batch_size=batch_size,
            seed=seed,
            sampler="deterministic",
            device="cpu",
            checkpoint=ROOT / "checkpoint/model.pt",
            training_fasta=ROOT / "data/training/training.fasta",
            challenge_fasta=ROOT / "data/antibacterial.fasta",
            length_policy=LengthPolicy.EMPIRICAL_TEMPERED,
            minimum_length=10,
            maximum_length=40,
            length_temperature=0.75,
            official_similarity_max=OFFICIAL_CHALLENGE_SIMILARITY_MAX,
            selection_similarity_max=0.78,
            cpu_threads=8,
        )


@dataclass(frozen=True)
class GitSourceState:
    commit: str
    dirty: bool
    diff_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_source_state(repository: Path = ROOT) -> GitSourceState:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return GitSourceState(commit_result.stdout.strip(), False, None)

    digest = hashlib.sha256()
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    for relative_bytes in sorted(filter(None, untracked_output.split(b"\0"))):
        relative_path = Path(os.fsdecode(relative_bytes))
        path = repository / relative_path
        digest.update(b"untracked\0")
        digest.update(relative_bytes)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode())
            continue
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return GitSourceState(commit_result.stdout.strip(), True, digest.hexdigest())


def _mark_rejections(candidates: list[Candidate], references: set[str]) -> list[Candidate]:
    seen: set[str] = set()
    marked: list[Candidate] = []
    for candidate in candidates:
        reason = candidate.rejection_reason
        if not reason and candidate.sequence in seen:
            reason = "duplicate_sequence"
        if not reason and candidate.sequence in references:
            reason = "exact_reference_overlap"
        seen.add(candidate.sequence)
        marked.append(replace(candidate, valid=not reason, rejection_reason=reason))
    return marked


def _select_smoke_top(
    library: list[Candidate], top_k: int, references: set[str], similarity_max: float
) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in library:
        if passes_selection_similarity(candidate.sequence, references, similarity_max):
            selected.append(candidate)
            if len(selected) == top_k:
                return selected
    raise RuntimeError(f"Only {len(selected)} candidates passed the {similarity_max:.2f} margin")


def _select_official_top(
    library: list[Candidate], options: PipelineOptions, references: set[str], work_dir: Path
) -> list[Candidate]:
    from ampdiffusion_starter_kit.generate import select_top

    known_amps = read_fasta_sequences(options.training_fasta)
    sequences = select_top(
        [candidate.sequence for candidate in library],
        options.top_k,
        references,
        known_amps,
        work_dir,
        challenge_similarity_threshold=options.selection_similarity_max,
    )
    by_sequence = {candidate.sequence: candidate for candidate in library}
    return [by_sequence[sequence] for sequence in sequences]


def _write_ranking(top: list[Candidate], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("rank", "candidate_id", "sequence", "final_score"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        denominator = max(len(top), 1)
        for rank, candidate in enumerate(top, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "sequence": candidate.sequence,
                    "final_score": f"{1.0 - (rank - 1) / denominator:.6f}",
                }
            )


def _peak_cuda_memory() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _write_manifest(
    options: PipelineOptions,
    quotas: dict[int, int],
    output_dir: Path,
    *,
    runtime_seconds: float,
    advanced_selection: bool,
) -> None:
    source_state = _git_source_state()
    config = yaml.safe_load(options.config_path.read_text())
    manifest = {
        "schema_version": 1,
        "commit": source_state.commit,
        "source_dirty": source_state.dirty,
        "source_diff_sha256": source_state.diff_sha256,
        "seed": options.seed,
        "config_sha256": _sha256(options.config_path),
        "checkpoint_sha256": _sha256(options.checkpoint),
        "challenge_reference_sha256": _sha256(options.challenge_fasta),
        "training_fasta_sha256": _sha256(options.training_fasta),
        "raw_count": options.raw_pool_size,
        "library_count": options.library_size,
        "top_count": options.top_k,
        "generation_policy": {
            "sampler": options.sampler,
            "length_policy": options.length_policy.value,
            "length_temperature": options.length_temperature,
            "length_quotas": {str(key): value for key, value in quotas.items()},
            "batch_size": options.batch_size,
        },
        "ranking_policy": {
            "official_challenge_similarity_max": options.official_similarity_max,
            "selection_challenge_similarity_max": options.selection_similarity_max,
        },
        "peak_cuda_memory_bytes": _peak_cuda_memory(),
        "peak_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "runtime_seconds": runtime_seconds,
        "advanced_selection": advanced_selection,
        "calibration_sha256": (
            _sha256(ROOT / "checkpoint/calibration.json")
            if (ROOT / "checkpoint/calibration.json").is_file()
            else None
        ),
        "manual_intervention": False,
        "output_sha256": {
            name: _sha256(output_dir / name)
            for name in ("library.fasta", "top.fasta", "ranking.tsv")
        },
    }
    if _adopted_selection_enabled(options):
        manifest["inference_policy"] = config["inference"]
        manifest["sampling_steps"] = config["generation"]["sampling_steps"]
        manifest["encoder_manifest_sha256"] = _sha256(ROOT / "configs/inference_encoders.json")
        manifest["inference_assets_sha256"] = json.loads(
            (output_dir / "work/deployed/models/assets_sha256.json").read_text()
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _advanced_selection_enabled(options: PipelineOptions) -> bool:
    config = yaml.safe_load(options.config_path.read_text())
    library_selection = config.get("library_selection", {})
    minimum_size = int(library_selection.get("minimum_full_pool_size", 50_000))
    return (
        bool(library_selection.get("enabled", False))
        and str(library_selection.get("adopted_variant")) == "L2"
        and options.library_size >= minimum_size
    )


def _adopted_selection_enabled(options: PipelineOptions) -> bool:
    config = yaml.safe_load(options.config_path.read_text())
    return (
        options.sampler == "official"
        and config.get("inference", {}).get("policy") == "ddim-lref-rankmean"
    )


def _run_adopted_selection(options: PipelineOptions, temporary: Path) -> None:
    _run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/run_adopted_selection.py"),
            "--config",
            str(options.config_path),
            "--root",
            str(temporary / "work/deployed"),
            "--raw",
            str(temporary / "work/candidates.csv.gz"),
            "--output",
            str(temporary),
            "--size",
            str(options.library_size),
            "--top-k",
            str(options.top_k),
            "--reference",
            str(options.training_fasta),
            "--challenge",
            str(options.challenge_fasta),
            "--device",
            options.device,
        ]
    )


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _release_generation_models() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_advanced_selection(options: PipelineOptions, temporary: Path) -> None:
    work = temporary / "work"
    candidates_path = work / "candidates.csv.gz"
    physchem_path = work / "candidate_physchem.csv.gz"
    embeddings_path = work / "candidate_embeddings.npy"
    reference_embeddings_path = work / "reference_embeddings.npy"
    embedding_diagnostics_path = work / "candidate_embedding_diagnostics.csv.gz"
    apex_fasta_path = work / "apex_candidates.fasta"
    apex_path = work / "apex_predictions.npz"
    valid_rows: list[FastaRecord] = []
    with gzip.open(candidates_path, "rt", newline="") as file:
        for row in csv.DictReader(file):
            if row["valid"] == "True":
                valid_rows.append(FastaRecord(row["candidate_id"], row["sequence"]))
    write_fasta(valid_rows, apex_fasta_path)
    _run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/compute_physchem.py"),
            "--input",
            str(candidates_path),
            "--output",
            str(physchem_path),
            "--reference-output",
            str(work / "physchem_reference.json"),
            "--reference-fasta",
            str(options.training_fasta),
        ]
    )
    _run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/compute_embeddings.py"),
            "--candidates",
            str(candidates_path),
            "--candidate-embeddings",
            str(embeddings_path),
            "--reference-embeddings",
            str(reference_embeddings_path),
            "--diagnostics-output",
            str(embedding_diagnostics_path),
            "--manifest-output",
            str(work / "embedding_manifest.json"),
            "--reference-fasta",
            str(options.training_fasta),
            "--device",
            options.device,
        ]
    )
    _run_checked(
        [
            sys.executable,
            str(ROOT / "apex/APEX_predict_ensemble.py"),
            "--input",
            str(apex_fasta_path),
            "--output",
            str(apex_path),
            "--aggregates",
            str(work / "apex_mean.csv"),
            "--manifest",
            str(work / "apex_manifest.json"),
        ]
    )
    staged_library = work / "library_l2"
    _run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/select_library.py"),
            "--variant",
            "L2",
            "--output-dir",
            str(staged_library),
            "--candidates",
            str(candidates_path),
            "--physchem",
            str(physchem_path),
            "--embeddings",
            str(embedding_diagnostics_path),
            "--apex",
            str(work / "apex_mean.csv"),
            "--size",
            str(options.library_size),
            "--top-k",
            str(options.top_k),
            "--report",
            str(work / "library_selection.csv"),
            "--training-fasta",
            str(options.training_fasta),
            "--challenge-fasta",
            str(options.challenge_fasta),
        ]
    )
    _run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/select_top.py"),
            "--config",
            str(options.config_path),
            "--candidates",
            str(candidates_path),
            "--apex",
            str(work / "apex_mean.csv"),
            "--physchem",
            str(physchem_path),
            "--embeddings",
            str(embedding_diagnostics_path),
            "--library-fasta",
            str(staged_library / "library.fasta"),
            "--output-dir",
            str(temporary),
            "--library-size",
            str(options.library_size),
            "--top-k",
            str(options.top_k),
            "--challenge-fasta",
            str(options.challenge_fasta),
            "--known-fasta",
            str(options.training_fasta),
        ]
    )


def _publish_output(temporary: Path, output_dir: Path) -> None:
    backup = output_dir.with_name(f"{output_dir.name}.backup-{temporary.name.rsplit('-', 1)[-1]}")
    if output_dir.exists():
        output_dir.rename(backup)
    try:
        temporary.rename(output_dir)
    except OSError:
        if backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as error:
            LOGGER.warning(
                "Published %s but retained backup %s: %s",
                output_dir,
                backup,
                error,
            )


def run_pipeline(
    options: PipelineOptions,
    *,
    backend: SamplingBackend | None = None,
) -> Path:
    started = time.perf_counter()
    if options.official_similarity_max != OFFICIAL_CHALLENGE_SIMILARITY_MAX:
        raise ValueError("official_challenge_similarity_max is fixed at 0.80")
    if options.raw_pool_size < options.library_size:
        raise ValueError("raw_pool_size must be at least library_size")
    if options.library_size < options.top_k:
        raise ValueError("library_size must be at least top_k")

    set_process_determinism(options.seed, options.cpu_threads)
    references = set(read_fasta_sequences(options.challenge_fasta))
    quotas = build_length_quotas(
        options.raw_pool_size,
        options.length_policy,
        options.minimum_length,
        options.maximum_length,
        options.training_fasta,
        options.length_temperature,
    )
    output_dir = options.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        active_backend = backend
        if active_backend is None and options.sampler == "official":
            if _adopted_selection_enabled(options):
                prepare_default_encoders(ROOT)
            config = yaml.safe_load(options.config_path.read_text())
            active_backend = make_official_backend(
                options.checkpoint,
                options.device,
                sampling_steps=int(config.get("generation", {}).get("sampling_steps", 1000)),
            )
        candidates = generate_candidates(
            options.raw_pool_size,
            options.batch_size,
            options.seed,
            quotas,
            backend=active_backend,
        )
        active_backend = None
        _release_generation_models()
        candidates = _mark_rejections(candidates, references)
        write_candidates_csv(candidates, temporary / "work/candidates.csv.gz")
        library = [candidate for candidate in candidates if candidate.valid][: options.library_size]
        if len(library) != options.library_size:
            raise RuntimeError(
                f"Only {len(library)} valid unique candidates for "
                f"library_size={options.library_size}"
            )
        adopted_selection = _adopted_selection_enabled(options)
        advanced_selection = adopted_selection or _advanced_selection_enabled(options)
        if adopted_selection:
            _run_adopted_selection(options, temporary)
        elif advanced_selection:
            _run_advanced_selection(options, temporary)
        else:
            if options.sampler == "official" and backend is None:
                top = _select_official_top(library, options, references, temporary / "work/apex")
            else:
                top = _select_smoke_top(
                    library, options.top_k, references, options.selection_similarity_max
                )
            write_fasta(
                [FastaRecord(candidate.candidate_id, candidate.sequence) for candidate in library],
                temporary / "library.fasta",
            )
            write_fasta(
                [FastaRecord(candidate.candidate_id, candidate.sequence) for candidate in top],
                temporary / "top.fasta",
            )
            _write_ranking(top, temporary / "ranking.tsv")
        _write_manifest(
            options,
            quotas,
            temporary,
            runtime_seconds=time.perf_counter() - started,
            advanced_selection=advanced_selection,
        )
        require_valid_submission(
            temporary,
            references,
            library_size=options.library_size,
            top_k=options.top_k,
        )
        if output_dir.name == "generate":
            shared_metadata = ROOT / "work/candidates.csv.gz"
            shared_metadata.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporary / "work/candidates.csv.gz", shared_metadata)
        _publish_output(temporary, output_dir)
        return output_dir
    except Exception:
        if temporary.exists():
            failed = temporary.with_name(
                f"{output_dir.name}.failed-{temporary.name.rsplit('-', 1)[-1]}"
            )
            temporary.rename(failed)
        raise


PipelineCallable = Callable[[PipelineOptions], Path]
