import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from robust_apex_qd.evaluation.models import HemoPI2Prediction

HEMOPI2_VERSION = "1.3"
HEMOPI2_WHEEL_SHA256 = "53ee834b39ffa26b271674412d2b43ebaadcb27c5eedba376e5d8f4ab6feb34b"
PYPI_JSON_URL = f"https://pypi.org/pypi/hemopi2/{HEMOPI2_VERSION}/json"


def load_hemopi2_predictions(
    path: Path,
    expected_sequences: Mapping[str, str],
    *,
    require_complete: bool = True,
) -> dict[str, HemoPI2Prediction]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    required = {"SeqID", "Sequence", "HC50(μM)", "Prediction"}
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError(f"HemoPI2 CSV must contain {sorted(required)}")
    predictions: dict[str, HemoPI2Prediction] = {}
    for row in rows:
        candidate_id = row["SeqID"].strip().lstrip(">")
        if candidate_id in predictions:
            raise ValueError(f"HemoPI2 CSV contains duplicate SeqID {candidate_id}")
        if candidate_id not in expected_sequences:
            raise ValueError(f"HemoPI2 CSV contains unexpected SeqID {candidate_id}")
        sequence = row["Sequence"].strip().upper()
        if sequence != expected_sequences[candidate_id]:
            raise ValueError(f"HemoPI2 sequence differs for {candidate_id}")
        try:
            hc50 = float(row["HC50(μM)"])
        except ValueError as error:
            raise ValueError(f"HemoPI2 HC50 is not numeric for {candidate_id}") from error
        if not math.isfinite(hc50) or hc50 <= 0:
            raise ValueError(f"HemoPI2 HC50 must be positive and finite for {candidate_id}")
        label = row["Prediction"].strip().lower().replace("_", "-")
        if label not in {"hemolytic", "non-hemolytic"}:
            raise ValueError(f"Unexpected HemoPI2 prediction for {candidate_id}: {label}")
        predictions[candidate_id] = HemoPI2Prediction(
            candidate_id=candidate_id,
            sequence=sequence,
            hc50_u_m=hc50,
            hemolytic=label == "hemolytic",
        )
    missing = set(expected_sequences) - set(predictions)
    if require_complete and missing:
        raise ValueError(f"HemoPI2 CSV is missing SeqID {sorted(missing)[0]}")
    return predictions


def write_hemopi2_predictions(
    path: Path,
    predictions: Mapping[str, HemoPI2Prediction],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("SeqID", "Sequence", "HC50(μM)", "Prediction"),
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate_id in sorted(predictions):
            prediction = predictions[candidate_id]
            writer.writerow(
                {
                    "SeqID": candidate_id,
                    "Sequence": prediction.sequence,
                    "HC50(μM)": f"{prediction.hc50_u_m:.10g}",
                    "Prediction": "Hemolytic" if prediction.hemolytic else "Non-Hemolytic",
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_path(oracle_dir: Path) -> Path:
    return oracle_dir / ".venv" / "bin" / "python"


def find_hemopi2_script(oracle_dir: Path) -> Path | None:
    venv = oracle_dir / ".venv"
    scripts = sorted(venv.glob("lib/python*/site-packages/**/hemopi2_regression.py"))
    if scripts:
        return scripts[0]
    entry = venv / "bin" / "hemopi2_regression"
    return entry if entry.is_file() else None


def hemopi2_environment_status(oracle_dir: Path) -> tuple[bool, str]:
    manifest = oracle_dir / "manifest.json"
    python = _python_path(oracle_dir)
    script = find_hemopi2_script(oracle_dir)
    if not manifest.is_file() or not python.is_file() or script is None:
        return False, "Run prepare-evaluation-oracles to create the isolated HemoPI2 environment."
    try:
        payload = json.loads(manifest.read_text())
    except json.JSONDecodeError as error:
        return False, f"Invalid HemoPI2 manifest: {error}"
    if payload.get("hemopi2_version") != HEMOPI2_VERSION:
        return False, "HemoPI2 manifest version differs from the pinned adapter version."
    return True, "HemoPI2 isolated environment is ready."


def prepare_hemopi2_environment(oracle_dir: Path) -> dict[str, object]:
    target = oracle_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    wheel = target / f"hemopi2-{HEMOPI2_VERSION}-py3-none-any.whl"
    if not wheel.is_file() or _sha256(wheel) != HEMOPI2_WHEEL_SHA256:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=60) as response:
            metadata = json.load(response)
        candidates = [
            item
            for item in metadata["urls"]
            if item["filename"] == wheel.name and item["digests"]["sha256"] == HEMOPI2_WHEEL_SHA256
        ]
        if len(candidates) != 1:
            raise RuntimeError("Pinned HemoPI2 wheel was not found in the PyPI release metadata")
        temporary = wheel.with_suffix(".download")
        urllib.request.urlretrieve(candidates[0]["url"], temporary)
        if _sha256(temporary) != HEMOPI2_WHEEL_SHA256:
            raise RuntimeError("Downloaded HemoPI2 wheel checksum differs from the pinned value")
        temporary.replace(wheel)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is required to prepare the HemoPI2 environment")
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(target / "uv-cache")
    subprocess.run(
        [uv, "venv", "--python", "3.10", str(target / ".venv")],
        check=True,
        env=environment,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(_python_path(target)),
            str(wheel),
            "numpy<2",
            "pandas<3",
            "scikit-learn==1.3.1",
            "tqdm",
        ],
        check=True,
        env=environment,
    )
    script = find_hemopi2_script(target)
    if script is None:
        raise RuntimeError("Installed HemoPI2 distribution does not contain its regression runner")
    manifest = {
        "schema_version": 1,
        "hemopi2_version": HEMOPI2_VERSION,
        "wheel_sha256": HEMOPI2_WHEEL_SHA256,
        "runner_relative_path": script.relative_to(target).as_posix(),
        "runner_sha256": _sha256(script),
        "maximum_sequence_length": 40,
        "hemolytic_hc50_threshold_u_m": 100.0,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_hemopi2(
    oracle_dir: Path,
    sequences: Mapping[str, str],
) -> dict[str, HemoPI2Prediction]:
    target = oracle_dir.resolve()
    ready, detail = hemopi2_environment_status(target)
    if not ready:
        raise RuntimeError(detail)
    overlength = [
        candidate_id for candidate_id, sequence in sequences.items() if len(sequence) > 40
    ]
    if overlength:
        raise ValueError(
            f"HemoPI2 truncates sequences above 40 aa; refusing candidate {overlength[0]}"
        )
    script = find_hemopi2_script(target)
    if script is None:
        raise RuntimeError("HemoPI2 regression runner disappeared after environment validation")
    with tempfile.TemporaryDirectory(prefix="hemopi2-") as raw_work:
        work = Path(raw_work)
        fasta = work / "input.fasta"
        fasta.write_text(
            "".join(
                f">{candidate_id}\n{sequence}\n" for candidate_id, sequence in sequences.items()
            )
        )
        command = [
            str(_python_path(target)),
            str(script),
            "-i",
            str(fasta),
            "-o",
            "final_output.csv",
            "-wd",
            str(work),
            "-d",
            "2",
        ]
        environment = os.environ.copy()
        environment["PATH"] = f"{_python_path(target).parent}:{environment.get('PATH', '')}"
        subprocess.run(
            command,
            check=True,
            cwd=script.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = work / "final_output.csv"
        if not output.is_file():
            raise RuntimeError("HemoPI2 completed without final_output.csv")
        return load_hemopi2_predictions(output, sequences)
