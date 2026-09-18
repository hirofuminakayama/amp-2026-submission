import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

APEX_PATHOGENS = (
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
)
APEX_MODEL_COUNT = 8
APEX_PATHOGEN_COUNT = 11


@dataclass(frozen=True)
class ApexAggregates:
    official_pathogen_mean_mic_u_m: NDArray[np.float32]
    official_broad_mean_mic_u_m: NDArray[np.float32]
    pathogen_success16: NDArray[np.float32]
    vote16: NDArray[np.float32]
    median_log2_mic: NDArray[np.float32]
    q90_log2_mic: NDArray[np.float32]
    worst3_log2_mic: NDArray[np.float32]
    model_disagreement_mad_log2: NDArray[np.float32]


@dataclass(frozen=True)
class ApexPredictionArchive:
    sequences: tuple[str, ...]
    model_names: tuple[str, ...]
    pathogens: tuple[str, ...]
    mic_u_m: NDArray[np.float32]


def _validated_tensor(mic_u_m: NDArray[np.floating]) -> NDArray[np.float32]:
    tensor = np.asarray(mic_u_m, dtype=np.float32)
    if tensor.ndim != 3:
        raise ValueError("APEX predictions must have shape [sequence, model, pathogen]")
    if tensor.shape[1:] != (APEX_MODEL_COUNT, APEX_PATHOGEN_COUNT):
        raise ValueError("APEX predictions must have shape [sequence, 8 models, 11 pathogens]")
    if not np.isfinite(tensor).all() or np.any(tensor <= 0):
        raise ValueError("APEX MIC values must be positive and finite")
    return tensor


def write_json_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def aggregate_predictions(mic_u_m: NDArray[np.floating]) -> ApexAggregates:
    tensor = _validated_tensor(mic_u_m)
    official_pathogen_mean = tensor.mean(axis=1, dtype=np.float32)
    official_broad_mean = official_pathogen_mean.mean(axis=1, dtype=np.float32)
    pathogen_success16 = (tensor <= 16.0).mean(axis=1, dtype=np.float32)
    vote16 = (tensor <= 16.0).mean(axis=(1, 2), dtype=np.float32)

    pathogen_log2 = np.log2(official_pathogen_mean).astype(np.float32)
    median_log2 = np.median(pathogen_log2, axis=1).astype(np.float32)
    q90_log2 = np.quantile(pathogen_log2, 0.90, axis=1).astype(np.float32)
    worst_count = min(3, pathogen_log2.shape[1])
    worst3_log2 = np.sort(pathogen_log2, axis=1)[:, -worst_count:].mean(
        axis=1,
        dtype=np.float32,
    )

    model_log2 = np.log2(tensor).astype(np.float32)
    model_median = np.median(model_log2, axis=1, keepdims=True)
    pathogen_mad = np.median(np.abs(model_log2 - model_median), axis=1)
    disagreement = pathogen_mad.mean(axis=1, dtype=np.float32)
    return ApexAggregates(
        official_pathogen_mean_mic_u_m=official_pathogen_mean,
        official_broad_mean_mic_u_m=official_broad_mean,
        pathogen_success16=pathogen_success16,
        vote16=vote16,
        median_log2_mic=median_log2,
        q90_log2_mic=q90_log2,
        worst3_log2_mic=worst3_log2,
        model_disagreement_mad_log2=disagreement,
    )


def _unicode_array(values: Sequence[str]) -> NDArray[np.str_]:
    maximum_length = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{maximum_length}")


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)


def write_prediction_archive(
    path: Path,
    sequences: Sequence[str],
    model_names: Sequence[str],
    pathogens: Sequence[str],
    mic_u_m: NDArray[np.floating],
) -> None:
    tensor = _validated_tensor(mic_u_m)
    if tensor.shape != (len(sequences), len(model_names), len(pathogens)):
        raise ValueError("APEX tensor axes do not align with sequence/model/pathogen names")
    if len(model_names) != APEX_MODEL_COUNT:
        raise ValueError(f"Expected {APEX_MODEL_COUNT} APEX model names")
    if tuple(pathogens) != APEX_PATHOGENS:
        raise ValueError("APEX pathogen order differs from the fixed canonical order")
    _write_deterministic_npz(
        path,
        {
            "sequences": _unicode_array(sequences),
            "model_names": _unicode_array(model_names),
            "pathogens": _unicode_array(pathogens),
            "mic_uM": tensor,
        },
    )


def load_prediction_archive(path: Path) -> ApexPredictionArchive:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sequences", "model_names", "pathogens", "mic_uM"}
        if set(archive.files) != required:
            raise ValueError(f"APEX archive keys {set(archive.files)} != {required}")
        sequences = tuple(str(value) for value in archive["sequences"].tolist())
        model_names = tuple(str(value) for value in archive["model_names"].tolist())
        pathogens = tuple(str(value) for value in archive["pathogens"].tolist())
        tensor = _validated_tensor(archive["mic_uM"])
    if tensor.shape != (len(sequences), len(model_names), len(pathogens)):
        raise ValueError("Stored APEX tensor axes do not align with archive names")
    return ApexPredictionArchive(
        sequences=sequences,
        model_names=model_names,
        pathogens=pathogens,
        mic_u_m=tensor,
    )
