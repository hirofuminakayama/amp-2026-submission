"""Compare generator screens on native and common-length cohorts with explicit denominators."""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_research_generation import validate_run_protocol
from run_competition_models import write_json
from run_research_models import load_esm
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import load_prediction_archive
from robust_apex_qd.evaluation.oracles import run_hemopi2
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.research.competition_generators import validate_screen
from robust_apex_qd.research.competition_models import predict_ridge_heads
from robust_apex_qd.research.selection import keyed_subset


def artifact_hashes(directory: Path, manifest: Path) -> dict[str, str]:
    """Exclude replaced metadata and the enclosing stage record on resumed writes."""
    excluded = {manifest.resolve(), (directory / "stage_manifest.json").resolve()}
    return {
        p.name: file_sha256(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.resolve() not in excluded
    }


def validate_optimization_pair(control: dict, alternative: dict) -> None:
    """Require matching sampling conditions before attributing a paired change to updates."""
    for field in ["seed", "device", "requested", "min_length", "max_length", "batch_size"]:
        if control[field] != alternative[field]:
            raise ValueError(f"Optimization comparison differs in {field}")


def nondominated(values: np.ndarray) -> np.ndarray:
    """Select finite Pareto points after every axis is oriented higher-is-better."""
    finite = np.asarray(np.isfinite(values).all(axis=1), dtype=bool)
    result = finite.copy()
    for i in np.flatnonzero(finite):
        competitors = values[finite]
        result[i] = not np.any(
            (competitors >= values[i]).all(axis=1) & (competitors > values[i]).any(axis=1)
        )
    return result


def raw_sequences(path: Path) -> list[str]:
    if (path / "raw_sequences.json").exists():
        return json.loads((path / "raw_sequences.json").read_text())
    sequences = []
    current = None
    for line in (path / "raw.fasta").read_text().splitlines():
        if line.startswith(">"):
            if current is not None:
                sequences.append(current)
            current = ""
        else:
            if current is None:
                raise ValueError("Missing FASTA header")
            current += line.strip()
    if current is not None:
        sequences.append(current)
    return sequences


def collect(root: Path, output: Path) -> None:
    old = Path("work/measured_activity_research/generation-20260911-b")
    old_config = json.loads(Path("configs/research_generation.json").read_text())
    runs = []
    tables = []
    paths = sorted(old.glob("*/run_manifest.json")) + sorted(root.glob("*/run_manifest.json"))
    paths = [p for p in paths if "smoke" not in p.parent.name]
    for manifest_path in paths:
        path = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest.get("artifacts_sha256", {}).items():
            if file_sha256(path / name) != digest:
                raise ValueError(f"Changed generation artifact {path / name}")
        if path.parent == old:
            expected = validate_run_protocol(path.name, manifest, old_config)
            sequences = raw_sequences(path)
            minimum, maximum = expected["min_length"], expected["max_length"]
            count = expected["count"]
            seed = expected["seed"]
            mode = "legacy-" + ("hydramp" if path.name.startswith("hydramp") else "diffusion")
        elif path.name.startswith("hydramp-conditional"):
            payload = json.loads((path / "raw.json").read_text())
            sequences = [
                s["sequence"]
                for value in payload.values()
                for s in value.get("generated_sequences") or []
            ]
            minimum, maximum = 10, 25
            count = manifest["count"]
            seed = manifest["seed"]
            mode = "hydramp-analogue"
            expected = dict(
                seed=int(path.name.rsplit("s", 1)[-1]),
                count=1000,
                mode="analogue",
                temperature=5.0,
                attempts=1,
                filtering_criteria="discovery",
                min_length=10,
                max_length=25,
            )
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError("HydrAMP analogue protocol differs from registration")
        else:
            sequences = raw_sequences(path)
            expected_mode = (
                "general_masked_completion"
                if path.name.startswith("deepamp-")
                else path.name.rsplit("-s", 1)[0]
            )
            ranges = {
                "designer": (10, 32),
                "prompt": (10, 34),
                "ampgen": (15, 35),
                "evodiff": (15, 35),
                "general_masked_completion": (10, 40),
                "evodiff-sft": (15, 35),
                "evodiff-cuda": (15, 35),
                "evodiff-rl": (15, 35),
            }
            minimum, maximum = ranges[expected_mode]
            expected_mode = "evodiff" if expected_mode.startswith("evodiff-") else expected_mode
            if expected_mode == "evodiff":
                device = "cpu" if path.name.rsplit("-s", 1)[0] == "evodiff" else "cuda"
                if manifest.get("device") != device:
                    raise ValueError("EvoDiff device differs from registration")
            if path.name.startswith("deepamp-common-") and manifest.get("conditioning") != "common":
                raise ValueError("DeepAMP conditioning differs from registration")
            count = 1000
            seed = int(path.name.rsplit("s", 1)[-1])
            mode = manifest["mode"]
            validate_screen(
                sequences,
                manifest,
                dict(
                    seed=seed,
                    count=count,
                    min_length=minimum,
                    max_length=maximum,
                    mode=expected_mode,
                ),
            )
        if seed not in [42, 43] or count not in [1000, 2000]:
            raise ValueError("Unregistered screen size or seed")
        if path.parent == old and len(sequences) != count:
            raise ValueError("Legacy sample count differs")
        valid = [
            s
            for s in sequences
            if minimum <= len(s) <= maximum and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
        ]
        write_fasta(
            [FastaRecord(f"s{i}", s) for i, s in enumerate(dict.fromkeys(valid))],
            output / f"{path.name}-valid.fasta",
        )
        checked = read_fasta_sequences(output / f"{path.name}-valid.fasta")
        if len(checked) != len(set(valid)):
            raise ValueError("Filtered FASTA validation failed")
        tables.extend(dict(run=path.name, sequence=s, seed=seed, mode=mode) for s in valid)
        runs.append(
            dict(
                run=path.name,
                seed=seed,
                mode=mode,
                device=manifest.get("device", "upstream default"),
                batch_size=manifest.get("batch_size"),
                conditioning=manifest.get("conditioning", "prefix")
                if path.name.startswith("deepamp-")
                else None,
                requested=count,
                emitted=len(sequences),
                returned_before_length_filter=manifest.get("raw_emitted_count", len(sequences)),
                valid=len(valid),
                unique=len(set(valid)),
                min_length=minimum,
                max_length=maximum,
                seconds=manifest.get("seconds", manifest.get("runtime_seconds")),
                peak_cuda_bytes=manifest.get("peak_cuda_bytes", 0),
                internal_filter_denominator="unknown" if "hydramp" in path.name else "raw attempts",
                manifest=str(manifest_path),
                manifest_sha256=file_sha256(manifest_path),
            )
        )
    # Expose supply-limited mixture ratios rather than silently filling an intended quota.
    frame = pd.DataFrame(tables)
    for seed in [42, 43]:
        left = list(dict.fromkeys(frame[frame.run.eq(f"paired-s{seed}")].sequence))
        right = list(dict.fromkeys(frame[frame.run.eq(f"hydramp-s{seed}")].sequence))
        for fraction in [0.1, 0.25, 0.5]:
            n = min(round(1000 * fraction), len(right))
            samples = right[:n] + [s for s in left if s not in set(right[:n])][: 1000 - n]
            name = f"hydramp-mix{fraction}-s{seed}"
            tables.extend(dict(run=name, sequence=s, seed=seed, mode="mixture") for s in samples)
            runs.append(
                dict(
                    run=name,
                    seed=seed,
                    mode="mixture",
                    requested=1000,
                    emitted=len(samples),
                    valid=len(samples),
                    unique=len(set(samples)),
                    min_length=10,
                    max_length=25,
                    desired_fraction=fraction,
                    actual_fraction=n / len(samples),
                    internal_filter_denominator="source HydrAMP decoder unknown",
                )
            )
    # Reward iterations have their own seed schedule and preserve the SFT-only iteration zero.
    optimization = root / "evodiff-optimization"
    if optimization.exists():
        completed = json.loads((optimization / "manifest.json").read_text())
        for name, digest in completed["artifacts_sha256"].items():
            if file_sha256(optimization / name) != digest:
                raise ValueError("Reward optimization artifact changed")
    for path in sorted((root / "evodiff-optimization").glob("iteration*/iteration.json")):
        manifest = json.loads(path.read_text())
        sequences = raw_sequences(path.parent)
        iteration = int(path.parent.name.removeprefix("iteration"))
        if iteration not in range(6):
            raise ValueError("Unregistered reward iteration")
        validation = validate_screen(
            sequences,
            {**manifest, "min_length": 15, "max_length": 35},
            dict(
                iteration=iteration, seed=42 + iteration, count=1000, min_length=15, max_length=35
            ),
        )
        if validation["valid"] != len(sequences):
            raise ValueError("Reward iteration contains invalid peptides")
        name = "evodiff-" + path.parent.name
        tables.extend(
            dict(run=name, sequence=s, seed=manifest["seed"], mode="reward-update")
            for s in sequences
        )
        runs.append(
            dict(
                run=name,
                seed=manifest["seed"],
                mode="reward-update",
                requested=1000,
                emitted=len(sequences),
                valid=len(sequences),
                unique=len(set(sequences)),
                min_length=15,
                max_length=35,
                reward_model="new physchem",
                internal_filter_denominator="raw attempts",
                manifest=str(path),
                manifest_sha256=file_sha256(path),
            )
        )
    pd.DataFrame(tables).to_csv(output / "samples.csv.gz", index=False)
    pd.DataFrame(runs).to_csv(output / "runs.csv", index=False)
    sequences = sorted(set(pd.DataFrame(tables).sequence))
    write_fasta(
        [FastaRecord(f"g{i}", s) for i, s in enumerate(sequences)], output / "sequences.fasta"
    )
    write_json(
        output / "manifest.json",
        dict(
            runs=len(runs),
            unique_sequences=len(sequences),
            artifacts_sha256=artifact_hashes(output, output / "manifest.json"),
        ),
    )


def score(root: Path, output: Path) -> None:
    sequences = read_fasta_sequences(root / "collect/sequences.fasta")
    archive_path = output / "apex.npz"
    if not archive_path.exists():
        subprocess.run(
            [
                sys.executable,
                "apex/APEX_predict_ensemble.py",
                "--input",
                str(root / "collect/sequences.fasta"),
                "--output",
                str(archive_path),
                "--device",
                "cpu",
                "--batch-size",
                "64",
            ],
            check=True,
        )
    archive = load_prediction_archive(archive_path)
    if tuple(sequences) != archive.sequences:
        raise ValueError("APEX sequence alignment changed")
    props = [compute_features(s) for s in sequences]
    features = np.asarray(
        [
            [*p.values(), *[s.count(a) / len(s) for a in "ACDEFGHIKLMNPQRSTVWY"]]
            for s, p in zip(sequences, props, strict=True)
        ]
    )
    reward = Path("work/competition_exploration/20260912-b/phase4/reward/weights.npz")
    predictions = predict_ridge_heads(dict(np.load(reward)), features)
    frame = pd.DataFrame(
        dict(
            sequence=sequences,
            apex_vote=(archive.mic_u_m <= 16).mean((1, 2)),
            apex_log2=np.median(np.log2(archive.mic_u_m.mean(1)), axis=1),
            new_physchem_log2=np.nanmean(predictions, axis=1),
        )
    )
    for name in ["charge_ph_7_4", "gravy", "shannon_entropy"]:
        frame[name] = [p[name] for p in props]
    frame.to_csv(output / "predictions.csv.gz", index=False)
    write_json(
        output / "manifest.json",
        dict(
            reward_sha256=file_sha256(reward),
            artifacts_sha256=artifact_hashes(output, output / "manifest.json"),
        ),
    )


def hemopi_batches(sequences: list[str], output: Path, batch_size: int = 1000) -> dict[str, float]:
    def one(start: int) -> pd.DataFrame:
        path = output / f"batch{start:05}.csv"
        batch = sequences[start : start + batch_size]
        if path.exists():
            frame = pd.read_csv(path)
            if frame.sequence.tolist() != batch:
                raise ValueError("Oracle resumed batch mismatch")
        else:
            items = {f"h{i}": s for i, s in enumerate(batch)}
            result = run_hemopi2(Path("work/oracles/hemopi2"), items)
            frame = pd.DataFrame(dict(sequence=batch, hc50=[result[i].hc50_u_m for i in items]))
            frame.to_csv(path, index=False)
        print(f"HC50 {start + len(batch)}/{len(sequences)}", flush=True)
        return frame

    values = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for frame in executor.map(one, range(0, len(sequences), batch_size)):
            values.update(dict(zip(frame.sequence, frame.hc50, strict=True)))
    return values


def safety(root: Path, output: Path) -> None:
    sequences = read_fasta_sequences(root / "collect/sequences.fasta")
    baseline_cache = Path(
        "work/competition_exploration/20260912-a/phase2/oracles/predictions.csv.gz"
    )
    existing = pd.read_csv(baseline_cache).set_index("sequence")
    cache_inputs = {str(baseline_cache): file_sha256(baseline_cache)}
    values = existing.hemopi2_hc50_u_m.to_dict()
    identity = {
        str(p): file_sha256(p)
        for p in [
            Path("work/oracles/hemopi2/manifest.json"),
            Path("src/robust_apex_qd/evaluation/oracles.py"),
        ]
    }
    cache_root = Path("work/competition_exploration/20260912-b")
    for cache in [
        cache_root / "phase4-evaluation-preflight/safety",
        cache_root / "phase4-evaluation-preflight-r2/safety",
    ]:
        if cache.resolve() != output.resolve() and (cache / "completion.json").exists():
            saved = json.loads((cache / "completion.json").read_text())
            if (
                saved["oracle_identity"] != identity
                or file_sha256(cache / "predictions.csv.gz") != saved["predictions_sha256"]
            ):
                raise ValueError("Safety cache identity changed")
            cached = pd.read_csv(cache / "predictions.csv.gz").dropna(subset=["hc50"])
            values.update(dict(zip(cached.sequence, cached.hc50, strict=True)))
            cache_inputs.update(
                {
                    str(cache / name): file_sha256(cache / name)
                    for name in ["predictions.csv.gz", "oracle_protocol.json", "completion.json"]
                }
            )
    missing = [s for s in sequences if s not in values and len(s) <= 40]
    protocol = dict(
        oracle_identity=identity,
        missing_sequences=missing,
        workers=4,
        batch_size=1000,
        cache_inputs_sha256=cache_inputs,
        cache_semantics="frozen predictions with retained original ordered batches",
        upstream_limitation="RRI feature loop retains row state; do not claim batch-invariant HC50",
    )
    protocol_path = output / "oracle_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Safety resume protocol changed")
    write_json(protocol_path, protocol)
    values.update(hemopi_batches(missing, output))
    pd.DataFrame(dict(sequence=sequences, hc50=[values.get(s) for s in sequences])).to_csv(
        output / "predictions.csv.gz", index=False
    )
    write_json(
        output / "completion.json",
        dict(
            oracle_identity=identity,
            predictions_sha256=file_sha256(output / "predictions.csv.gz"),
            evaluated=len(missing),
            reused=len(sequences) - len(missing),
        ),
    )


def embedding(root: Path, output: Path) -> None:
    samples = pd.read_csv(root / "collect/samples.csv.gz")
    refs = read_fasta_sequences(Path("data/training/training.fasta"))
    refs = sorted(
        set(s for s in refs if 8 <= len(s) <= 50 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))
    )
    selections = {}
    for name, frame in samples.groupby("run"):
        for scope in ["native", "common"]:
            seqs = sorted(set(frame.sequence))
            if scope == "common":
                seqs = [s for s in seqs if 15 <= len(s) <= 25]
            if len(seqs) >= 2:
                selections[f"{name}/{scope}"] = keyed_subset(seqs, min(1000, len(seqs)), 42)
    selections["reference/native"] = keyed_subset(refs, 1000, 42)
    selections["reference/common"] = keyed_subset([s for s in refs if 15 <= len(s) <= 25], 1000, 42)
    seqs = sorted(set().union(*map(set, selections.values())))
    checkpoint = Path("/home/hnaka/.cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt")
    model, alphabet = load_esm(checkpoint)
    model.eval()
    vectors = []
    for start in range(0, len(seqs), 32):
        batch = seqs[start : start + 32]
        _, _, tokens = alphabet.get_batch_converter()([(str(i), s) for i, s in enumerate(batch)])
        with torch.no_grad():
            rep = model(tokens, repr_layers=[6])["representations"][6]
        vectors.extend(rep[i, 1 : len(s) + 1].mean(0).numpy() for i, s in enumerate(batch))
    array = np.asarray(vectors)
    np.save(output / "embeddings.npy", array)
    write_json(output / "sequences.json", seqs)
    write_json(output / "subsets.json", selections)
    index = {s: i for i, s in enumerate(seqs)}
    metrics = []
    for name, selected in selections.items():
        if name.startswith("reference/"):
            continue
        run, scope = name.split("/")
        x = array[[index[s] for s in selected]].astype(float)
        r = array[[index[s] for s in selections[f"reference/{scope}"]]].astype(float)
        c1, c2 = np.cov(x, rowvar=False), np.cov(r, rowvar=False)
        value = np.sum((x.mean(0) - r.mean(0)) ** 2) + np.trace(c1 + c2 - 2 * sqrtm(c1 @ c2).real)
        matched = None
        if len(x) >= 100 and len(r) >= 100:
            a, b = x[:100], r[:100]
            ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
            matched = float(
                max(
                    0,
                    np.sum((a.mean(0) - b.mean(0)) ** 2)
                    + np.trace(ca + cb - 2 * sqrtm(ca @ cb).real),
                )
            )
        normalized = x / np.linalg.norm(x, axis=1)[:, None]
        diversity = (len(x) ** 2 - np.sum(normalized.sum(0) ** 2)) / (len(x) * (len(x) - 1))
        metrics.append(
            dict(
                run=run,
                scope=scope,
                fbd=float(max(0, value)),
                fbd_equal100=matched,
                embedding_cosine_diversity=float(diversity),
                subset_count=len(selected),
                reference_count=len(r),
            )
        )
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    write_json(
        output / "manifest.json",
        dict(
            checkpoint_sha256=file_sha256(checkpoint),
            device="cpu",
            artifacts_sha256=artifact_hashes(output, output / "manifest.json"),
        ),
    )


def report(root: Path, output: Path) -> None:
    samples = pd.read_csv(root / "collect/samples.csv.gz")
    samples.assign(length=samples.sequence.str.len()).groupby(["run", "length"]).agg(
        count=("sequence", "size")
    ).to_csv(output / "length_histograms.csv")
    runs = pd.read_csv(root / "collect/runs.csv").set_index("run")
    frame = samples.merge(
        pd.read_csv(root / "score/predictions.csv.gz"), on="sequence", validate="many_to_one"
    )
    frame = frame.merge(
        pd.read_csv(root / "safety/predictions.csv.gz"), on="sequence", validate="many_to_one"
    )
    known = set(read_fasta_sequences(Path("data/training/training.fasta")))
    challenge = set(read_fasta_sequences(Path("data/antibacterial.fasta")))
    development_path = Path(
        "work/competition_exploration/20260912-b/phase3-r3/prepare/sequences.fasta"
    )
    sft_path = Path("work/competition_exploration/20260912-b/phase4/reward/sft.fasta")
    own_development = set(read_fasta_sequences(development_path))
    own_sft = set(read_fasta_sequences(sft_path))
    write_json(
        output / "novelty_inputs.json",
        {
            str(p): file_sha256(p)
            for p in [
                Path("data/training/training.fasta"),
                Path("data/antibacterial.fasta"),
                development_path,
                sft_path,
            ]
        },
    )
    metrics = []
    for name, group in frame.groupby("run"):
        for scope in ["native", "common"]:
            cohort = group if scope == "native" else group[group.sequence.str.len().between(15, 25)]
            metrics.append(
                dict(
                    run=name,
                    scope=scope,
                    **runs.loc[name].dropna().to_dict(),
                    retained=len(cohort),
                    unique_retained=cohort.sequence.nunique(),
                    exact_novel=int((~cohort.sequence.isin(known | challenge)).sum()),
                    unique_exact_novel=len(set(cohort.sequence) - known - challenge),
                    unique_own_development_overlap=len(set(cohort.sequence) & own_development),
                    unique_own_sft_overlap=len(set(cohort.sequence) & own_sft),
                    challenge_exact_novel=int((~cohort.sequence.isin(challenge)).sum()),
                    mean_length=cohort.sequence.str.len().mean(),
                    apex_vote=cohort.apex_vote.mean(),
                    new_physchem_log2=cohort.new_physchem_log2.mean(),
                    hc50_coverage=int(cohort.hc50.notna().sum()),
                    hc50_median=cohort.hc50.median(),
                    mean_charge=cohort.charge_ph_7_4.mean(),
                    mean_gravy=cohort.gravy.mean(),
                )
            )
    comparison = pd.DataFrame(metrics).merge(
        pd.read_csv(root / "embedding/metrics.csv"),
        on=["run", "scope"],
        how="left",
        validate="one_to_one",
    )
    comparison.to_csv(output / "generator_comparison.csv", index=False)
    decisions = comparison.copy()
    decisions["pareto"] = False
    axes = ["apex_vote", "new_physchem_log2", "hc50_median", "fbd_equal100"]
    for _scope, indices in decisions.groupby("scope").groups.items():
        values = decisions.loc[indices, axes].to_numpy(float) * [1, -1, 1, -1]
        decisions.loc[indices, "pareto"] = nondominated(values)
    decisions["family"] = decisions.run.str.replace(r"-s\d+$", "", regex=True)
    decisions["screen_decision"] = np.where(
        decisions.pareto,
        "retain computational tradeoff for expansion",
        "retain family representative; lower expansion priority",
    )
    decisions.loc[decisions.unique_retained.lt(100), "screen_decision"] = (
        "supply limited; no equal100 FBD comparison"
    )
    decisions.to_csv(output / "screen_decisions.csv", index=False)
    paired = []
    indexed = comparison.set_index(["run", "scope"])
    for seed in [42, 43]:
        pairs = [
            (f"evodiff-cuda-s{seed}", f"evodiff-sft-s{seed}"),
            (f"evodiff-sft-s{seed}", f"evodiff-rl-s{seed}"),
        ]
        pairs += [
            (f"paired-s{seed}", name)
            for name in comparison.run.unique()
            if name.endswith(f"-s{seed}") and not name.startswith("paired-")
        ]
        for control, alternative in pairs:
            for scope in ["native", "common"]:
                if (control, scope) not in indexed.index or (
                    alternative,
                    scope,
                ) not in indexed.index:
                    continue
                before, after = indexed.loc[(control, scope)], indexed.loc[(alternative, scope)]
                if control.startswith("evodiff-"):
                    validate_optimization_pair(before.to_dict(), after.to_dict())
                paired.append(
                    dict(
                        control=control,
                        alternative=alternative,
                        control_device=before.device,
                        alternative_device=after.device,
                        comparison="matched_optimization"
                        if control.startswith("evodiff-")
                        else "cross_generator",
                        seed=seed,
                        scope=scope,
                        **{f"delta_{a}": after[a] - before[a] for a in axes},
                    )
                )
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    write_json(
        output / "interpretation.json",
        dict(
            exploratory=True,
            activity_truth="not measured",
            length_comparison="common range15-25; within-range length histograms are not matched",
            hc50=(
                "frozen original ordered-batch predictions; upstream RRI row-state dependence "
                "observed, so small differences are not evidence of improved biological safety"
            ),
            novelty=(
                "exact novelty against official training/reference FASTA; own development/SFT "
                "exact overlaps are separate and other pretrained-model overlap is not exhaustive; "
                "official Levenshtein novelty "
                "deferred to final library validation"
            ),
            fbd=(
                "ESM2-8M shared seed42, at most1000 unique per native/common scope; "
                "finite-sample size differences explicit"
            ),
            reward_included="new physchem only",
            reward_excluded="APEX and HC50",
            screening=(
                "retain family representatives and native/common Pareto tradeoffs; "
                "no old independent-evidence veto"
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("work/competition_exploration/20260912-b/phase4")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["collect", "score", "safety", "embedding", "report"], required=True
    )
    args = parser.parse_args()
    path = args.output / args.stage
    path.mkdir(parents=True, exist_ok=args.stage in ["score", "safety"])
    torch.set_num_threads(4)
    start = time.monotonic()
    with threadpool_limits(limits=4):
        if args.stage == "collect":
            collect(args.runs, path)
        else:
            {"score": score, "safety": safety, "embedding": embedding, "report": report}[
                args.stage
            ](args.output, path)
    write_json(
        path / "stage_manifest.json",
        dict(
            seconds=time.monotonic() - start,
            source_sha256=file_sha256(Path(__file__)),
            artifacts_sha256=artifact_hashes(path, path / "stage_manifest.json"),
        ),
    )


if __name__ == "__main__":
    main()
