"""Preserve all eight APEX base-learner predictions in deterministic [N, 8, 11] form."""

import argparse
import csv
import glob
import hashlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

APEX_DIR = Path(__file__).resolve().parent
ROOT = APEX_DIR.parent
sys.path.insert(0, str(APEX_DIR))
sys.path.insert(0, str(ROOT / "src"))

import APEX_models
from robust_apex_qd.apex.ensemble import (
    APEX_MODEL_COUNT,
    APEX_PATHOGENS,
    aggregate_predictions,
    write_json_manifest,
    write_prediction_archive,
)
from robust_apex_qd.io.fasta import read_fasta
from utils import make_vocab, onehot_encoding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_paths() -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(str(APEX_DIR / "APEX_pathogen_models/APEX_*")))]
    if len(paths) != APEX_MODEL_COUNT:
        raise ValueError(f"Expected {APEX_MODEL_COUNT} APEX weights, found {len(paths)}")
    return paths


def _load_models(device: torch.device) -> tuple[list[torch.nn.Module], list[Path]]:
    paths = _model_paths()
    models: list[torch.nn.Module] = []
    for path in paths:
        model = torch.load(path, map_location="cpu", weights_only=False)
        models.append(model.eval().to(device))
    return models, paths


def _predict(
    sequences: list[str],
    models: list[torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    word2idx, _ = make_vocab()
    predictions = np.empty(
        (len(sequences), len(models), len(APEX_PATHOGENS)),
        dtype=np.float32,
    )
    for model_index, model in enumerate(models):
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = onehot_encoding(batch, 52, word2idx)
            tensor = torch.as_tensor(encoded, dtype=torch.long, device=device)
            with torch.no_grad():
                transformed = model(tensor).detach().cpu().numpy()
            mic_u_m = np.power(10.0, 6.0 - transformed).astype(np.float32)
            predictions[start : start + len(batch), model_index, :] = mic_u_m
    return predictions


def _write_aggregates(path: Path, sequences: list[str], mic_u_m: np.ndarray) -> None:
    aggregates = aggregate_predictions(mic_u_m)
    pathogen_mean_names = tuple(f"mean_mic_uM__{name}" for name in APEX_PATHOGENS)
    pathogen_success_names = tuple(f"success16__{name}" for name in APEX_PATHOGENS)
    fieldnames = (
        "sequence",
        *pathogen_mean_names,
        *pathogen_success_names,
        "official_broad_mean_mic_uM",
        "vote16",
        "median_log2_mic",
        "q90_log2_mic",
        "worst3_log2_mic",
        "model_disagreement_mad_log2",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_index, sequence in enumerate(sequences):
            row: dict[str, str] = {"sequence": sequence}
            row.update(
                {
                    name: f"{float(value):.10g}"
                    for name, value in zip(
                        pathogen_mean_names,
                        aggregates.official_pathogen_mean_mic_u_m[row_index],
                    )
                }
            )
            row.update(
                {
                    name: f"{float(value):.10g}"
                    for name, value in zip(
                        pathogen_success_names,
                        aggregates.pathogen_success16[row_index],
                    )
                }
            )
            row.update(
                {
                    "official_broad_mean_mic_uM": f"{float(aggregates.official_broad_mean_mic_u_m[row_index]):.10g}",
                    "vote16": f"{float(aggregates.vote16[row_index]):.10g}",
                    "median_log2_mic": f"{float(aggregates.median_log2_mic[row_index]):.10g}",
                    "q90_log2_mic": f"{float(aggregates.q90_log2_mic[row_index]):.10g}",
                    "worst3_log2_mic": f"{float(aggregates.worst3_log2_mic[row_index]):.10g}",
                    "model_disagreement_mad_log2": f"{float(aggregates.model_disagreement_mad_log2[row_index]):.10g}",
                }
            )
            writer.writerow(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all eight APEX base learners")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    if options.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if options.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(options.cpu_threads)
    device = torch.device(options.device)
    records = read_fasta(options.input.resolve())
    sequences = [record.sequence for record in records if len(record.sequence) <= 50]
    if not sequences:
        raise ValueError("No APEX-compatible sequences were loaded")
    started = time.perf_counter()
    models, model_paths = _load_models(device)
    predictions = _predict(sequences, models, device, options.batch_size)
    output_path = options.output.resolve()
    model_names = [path.name for path in model_paths]
    write_prediction_archive(
        output_path,
        sequences,
        model_names,
        APEX_PATHOGENS,
        predictions,
    )
    aggregates_path = (
        options.aggregates.resolve()
        if options.aggregates is not None
        else output_path.with_name("apex_mean.csv")
    )
    _write_aggregates(aggregates_path, sequences, predictions)
    runtime_seconds = time.perf_counter() - started
    manifest_path = (
        options.manifest.resolve()
        if options.manifest is not None
        else output_path.with_name("apex_manifest.json")
    )
    manifest = {
        "schema_version": 1,
        "input_sha256": _sha256(options.input.resolve()),
        "sequences": len(sequences),
        "tensor_shape": list(predictions.shape),
        "tensor_dtype": "float32",
        "model_names": model_names,
        "model_sha256": {path.name: _sha256(path) for path in model_paths},
        "pathogens": list(APEX_PATHOGENS),
        "device": options.device,
        "batch_size": options.batch_size,
        "cpu_threads": options.cpu_threads,
        "runtime_seconds": runtime_seconds,
        "predictions_sha256": _sha256(output_path),
        "aggregates_sha256": _sha256(aggregates_path),
    }
    write_json_manifest(manifest_path, manifest)
    print(
        f"Wrote APEX tensor {predictions.shape} to {output_path} "
        f"in {runtime_seconds:.3f} seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
