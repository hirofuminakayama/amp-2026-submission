"""Prepare and score registered expanded pools without overwriting earlier runs."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_competition_models import predict, write_json
from run_research_models import load_esm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import load_prediction_archive
from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.research.competition_models import supported_blend
from robust_apex_qd.research.scale import align_cached_rows, merge_pool_sources
from robust_apex_qd.research.selection import align_apex_scores


def prepare(config: dict[str, Any], name: str, output: Path) -> None:
    sources = {}
    inputs = {}
    for key in config["pools"][name]:
        source = config["sources"][key]
        path = Path(source["path"])
        digest = file_sha256(path)
        if source.get("sha256") is not None and digest != source["sha256"]:
            raise ValueError(f"Changed registered source: {key}")
        if source.get("manifest"):
            manifest_path = Path(source["manifest"])
            manifest = json.loads(manifest_path.read_text())
            if manifest["artifacts_sha256"][path.name] != digest:
                raise ValueError(f"Generation artifact differs from manifest: {key}")
            inputs[str(manifest_path)] = file_sha256(manifest_path)
        rows = pd.read_csv(path)
        if source.get("runs"):
            rows = rows[rows.run.isin(source["runs"])]
        if len(rows) != source["count"]:
            raise ValueError(f"Incomplete registered raw count: {key}")
        sources[key] = rows.sequence.tolist()
        inputs[str(path)] = digest
    reference = Path(config["challenge"])
    pool, provenance, inventory = merge_pool_sources(sources, set(read_fasta_sequences(reference)))
    if len(pool) < config.get("minimum_pool_size", config["size"]):
        raise ValueError("Insufficient pool for a full library")
    pool.to_csv(output / "candidates.csv.gz", index=False)
    provenance.to_csv(output / "source_rows.csv.gz", index=False)
    inventory.to_csv(output / "pool_inventory.csv", index=False)
    write_fasta(
        [FastaRecord(i, s) for i, s in zip(pool.candidate_id, pool.sequence, strict=True)],
        output / "sequences.fasta",
    )
    write_json(output / "input_sha256.json", {**inputs, str(reference): file_sha256(reference)})


def features(config: dict[str, Any], name: str, root: Path, output: Path, device: str) -> None:
    pool = pd.read_csv(root / "prepare/candidates.csv.gz")
    if name == "baseline":
        old = Path(config["baseline"]) / "work"
        original = pd.read_csv(old / "candidates.csv.gz")
        indexed = original.drop_duplicates("sequence").set_index("sequence")
        ids = indexed.loc[pool.sequence].index
        # Duplicates can carry different cluster labels; preserve the retained first row.
        positions = {s: int(indexed.loc[s].raw_order) for s in ids}
        for filename in ["candidate_physchem.csv.gz", "candidate_embedding_diagnostics.csv.gz"]:
            data = pd.read_csv(old / filename).set_index("candidate_id")
            aligned = data.loc[indexed.loc[pool.sequence].candidate_id].reset_index(drop=True)
            aligned["candidate_id"] = pool.candidate_id
            aligned.to_csv(output / filename, index=False)
        np.save(
            output / "candidate_embeddings.npy",
            np.load(old / "candidate_embeddings.npy")[[positions[s] for s in pool.sequence]],
        )
        for filename in ["reference_embeddings.npy", "embedding_manifest.json"]:
            (output / filename).write_bytes((old / filename).read_bytes())
        # The registered selection protocol owns its earlier reference assignments.
        prior_config = json.loads((Path(config["prior_selection"]) / "protocol.json").read_text())
        clusters = Path(prior_config["old_selection"]) / "prepare/reference_clusters.csv"
        (output / "reference_clusters.csv").write_bytes(clusters.read_bytes())
        paths = [clusters, old / "candidates.csv.gz"] + [
            old / n
            for n in [
                "candidate_physchem.csv.gz",
                "candidate_embedding_diagnostics.csv.gz",
                "candidate_embeddings.npy",
                "reference_embeddings.npy",
                "embedding_manifest.json",
            ]
        ]
        write_json(output / "input_sha256.json", {str(p): file_sha256(p) for p in paths})
        return
    subprocess.run(
        [
            sys.executable,
            "scripts/compute_physchem.py",
            "--input",
            str(root / "prepare/candidates.csv.gz"),
            "--output",
            str(output / "candidate_physchem.csv.gz"),
            "--reference-output",
            str(output / "physchem_reference.json"),
            "--reference-fasta",
            config["reference"],
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compute_embeddings.py",
            "--candidates",
            str(root / "prepare/candidates.csv.gz"),
            "--candidate-embeddings",
            str(output / "candidate_embeddings.npy"),
            "--reference-embeddings",
            str(output / "reference_embeddings.npy"),
            "--diagnostics-output",
            str(output / "candidate_embedding_diagnostics.csv.gz"),
            "--manifest-output",
            str(output / "embedding_manifest.json"),
            "--reference-fasta",
            config["reference"],
            "--device",
            device,
        ],
        check=True,
    )
    em = json.loads((output / "embedding_manifest.json").read_text())
    reference = read_fasta_sequences(Path(config["reference"]))
    reference = [
        s for s in reference if 8 <= len(s) <= 50 and not set(s) - set("ACDEFGHIKLMNPQRSTVWY")
    ]
    vectors = np.load(output / "candidate_embeddings.npy")
    refs = np.load(output / "reference_embeddings.npy")
    ids = np.sort(
        np.random.default_rng(em["pca_seed"]).choice(
            len(refs), em["pca_reference_subset"], replace=False
        )
    )
    pca = PCA(n_components=em["pca_components"], svd_solver="full").fit(refs[ids])
    clusters = MiniBatchKMeans(
        n_clusters=em["cluster_count"],
        random_state=em["clustering_seed"],
        n_init=10,
        batch_size=2048,
        reassignment_ratio=0.0,
    )
    labels = clusters.fit_predict(pca.transform(vectors).astype(np.float32))
    expected = pd.read_csv(output / "candidate_embedding_diagnostics.csv.gz").embedding_cluster
    np.testing.assert_array_equal(labels, expected)
    pd.DataFrame(
        dict(
            sequence=reference,
            length=list(map(len, reference)),
            cluster=clusters.predict(pca.transform(refs).astype(np.float32)),
        )
    ).to_csv(output / "reference_clusters.csv", index=False)


def apex(config: dict[str, Any], name: str, root: Path, output: Path) -> None:
    pool = pd.read_csv(root / "prepare/candidates.csv.gz")
    if name == "baseline":
        archive_path = Path(config["baseline"]) / "work/apex_predictions.npz"
    else:
        archive_path = output / "apex.npz"
        subprocess.run(
            [
                sys.executable,
                "apex/APEX_predict_ensemble.py",
                "--input",
                str(root / "prepare/sequences.fasta"),
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
    index = {s: i for i, s in enumerate(archive.sequences)}
    tensor = archive.mic_u_m[[index[s] for s in pool.sequence]]
    np.save(output / "tensor.npy", tensor)
    scores = align_apex_scores(
        pool.sequence.tolist(), tensor, pool.sequence.tolist(), [True] * len(pool)
    )
    for key, value in scores.items():
        pool[key] = value
    for filename in ["candidate_physchem.csv.gz", "candidate_embedding_diagnostics.csv.gz"]:
        data = pd.read_csv(root / "features" / filename)
        shared = [c for c in data if c in pool and c != "candidate_id"]
        pool = pool.merge(data.drop(columns=shared), on="candidate_id", validate="one_to_one")
    votes = (tensor <= 16).mean(1)
    species = np.column_stack(
        [
            votes[:, 0],
            votes[:, 1:4].mean(1),
            votes[:, 4],
            votes[:, 5:7].mean(1),
            votes[:, 7:9].mean(1),
            votes[:, 9],
            votes[:, 10],
        ]
    )
    for i in range(7):
        pool[f"species_{i}"] = species[:, i]
    pool["species"] = species.mean(1)
    pool["worst3"] = np.sort(species, axis=1)[:, :3].mean(1)
    pool.to_csv(output / "pool.csv.gz", index=False)
    write_json(output / "input_sha256.json", {str(archive_path): file_sha256(archive_path)})


def models(config: dict[str, Any], root: Path, output: Path) -> None:
    pool = pd.read_csv(root / "apex/pool.csv.gz")
    refits = Path(config["refits"])
    cfg = json.loads(Path(config["model_config"]).read_text())
    tensor = np.load(root / "apex/tensor.npy")
    original = np.log2(tensor.mean(1))
    arms = json.loads((refits / "selected_models.json").read_text())
    ranks = []
    inputs = {}
    caches = []
    for name in config["pools"]:
        directory = root.parent / name / "models"
        record = directory / "manifest.json"
        if name == root.name or not record.exists():
            continue
        manifest = json.loads(record.read_text())
        if manifest["config"] != config:
            raise ValueError("Cached pool model configuration differs")
        verify_hashes({str(directory / k): v for k, v in manifest["artifacts_sha256"].items()})
        cached_sequences = pd.read_csv(directory / "pool.csv.gz").sequence.tolist()
        cached_inputs = json.loads((directory / "input_sha256.json").read_text())
        inputs[str(record)] = file_sha256(record)
        caches.append((directory, cached_sequences, cached_inputs))
    accounting = []
    for arm in arms:
        if arm["artifact_key"].startswith("ablation"):
            continue
        family = arm["family"]
        directory = refits / arm["artifact_key"]
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / (
            "weights.npz" if family in ["physchem", "linear8", "linear650"] else "weights.pt"
        )
        if file_sha256(path) != manifest["artifacts_sha256"][path.name]:
            raise ValueError("Frozen refit weights changed")
        inputs[str(path)] = file_sha256(path)
        state = (
            dict(np.load(path)) if path.suffix == ".npz" else torch.load(path, weights_only=False)
        )
        species = np.full((len(pool), 7), np.nan)
        strain = np.full((len(pool), 11), np.nan)
        available = np.zeros(len(pool), dtype=bool)
        vectors650 = (
            np.full((len(pool), 1280), np.nan, dtype=np.float32) if family == "linear650" else None
        )
        for cached_dir, cached_sequences, cached_inputs in caches:
            if cached_inputs.get(str(path)) != inputs[str(path)]:
                raise ValueError("Cached pool predictions used different model weights")
            cached = np.load(cached_dir / f"{family}.npz")
            aligned, present = align_cached_rows(
                pool.sequence.tolist(), cached_sequences, cached["species"]
            )
            use = present & ~available
            species[use] = aligned[use]
            aligned, _ = align_cached_rows(
                pool.sequence.tolist(), cached_sequences, cached["strain"]
            )
            strain[use] = aligned[use]
            if vectors650 is not None:
                if cached_dir.parent.name == "baseline":
                    embedding_path = refits / "esm650.npy"
                    sequence_path = refits / "pool_sequences.csv"
                    retained = (
                        pd.read_csv(sequence_path)
                        .reset_index()
                        .drop_duplicates("sequence", keep="last")
                    )
                    vector_sequences = retained.sequence.tolist()
                    cached_vectors = np.load(embedding_path, mmap_mode="r")[retained["index"]]
                    inputs[str(sequence_path)] = file_sha256(sequence_path)
                else:
                    embedding_path = cached_dir / "esm650.npy"
                    vector_sequences = cached_sequences
                    cached_vectors = np.load(embedding_path, mmap_mode="r")
                aligned, vector_present = align_cached_rows(
                    pool.sequence.tolist(), vector_sequences, cached_vectors
                )
                if not vector_present[use].all():
                    raise ValueError("Cached representation coverage differs from predictions")
                vectors650[use] = aligned[use]
                inputs[str(embedding_path)] = file_sha256(embedding_path)
            available |= present
        missing = ~available
        pending = pool.loc[missing].reset_index(drop=True)
        x = None
        if root.name == "baseline":
            saved = pd.read_csv(refits / "pool_sequences.csv")
            positions = {s: i for i, s in enumerate(saved.sequence)}
            archive_path = directory / "candidate_predictions.npz"
            if file_sha256(archive_path) != manifest["artifacts_sha256"][archive_path.name]:
                raise ValueError("Saved predictor values changed")
            archive = np.load(archive_path)
            rows = [positions[s] for s in pool.sequence]
            species, strain = archive["species"][rows], archive["strain"][rows]
            inputs[str(archive_path)] = file_sha256(archive_path)
            inputs[str(refits / "pool_sequences.csv")] = file_sha256(refits / "pool_sequences.csv")
            x = None
        elif len(pending) == 0:
            pass
        elif family == "physchem":
            x = np.asarray(
                [
                    [
                        *compute_features(s).values(),
                        *[s.count(a) / len(s) for a in "ACDEFGHIKLMNPQRSTVWY"],
                    ]
                    for s in pending.sequence
                ],
                dtype=np.float32,
            )
        elif family == "linear650":
            model, alphabet = load_esm(Path(cfg["esm650_checkpoint"]))
            model.eval().cuda()
            vectors = []
            for start in range(0, len(pending), 64):
                batch = pending.sequence.iloc[start : start + 64].tolist()
                _, _, tokens = alphabet.get_batch_converter()(
                    [(str(i), s) for i, s in enumerate(batch)]
                )
                with torch.no_grad():
                    rep = model(tokens.cuda(), repr_layers=[33])["representations"][33]
                vectors.extend(
                    rep[i, 1 : len(s) + 1].mean(0).cpu().numpy() for i, s in enumerate(batch)
                )
            x = np.asarray(vectors)
            del model
            torch.cuda.empty_cache()
            if vectors650 is None:
                raise ValueError("Missing representation buffer")
            vectors650[missing] = x
        else:
            x = np.load(root / "features/candidate_embeddings.npy")[missing]
        if x is not None:
            new_species, new_strain = predict(cfg, state, x, pending.sequence.tolist(), arm)
            species[missing], strain[missing] = new_species, new_strain
        if vectors650 is not None and root.name != "baseline":
            if not np.isfinite(vectors650).all():
                raise ValueError("Incomplete representation coverage")
            np.save(output / "esm650.npy", vectors650)
        if species.shape != (len(pool), 7) or strain.shape != (len(pool), 11):
            raise ValueError("Prediction shape changed")
        np.savez(output / f"{family}.npz", species=species, strain=strain)
        if not np.isfinite(species).all():
            raise ValueError("Incomplete species prediction coverage")
        accounting.append(
            dict(
                family=family,
                reused=int(available.sum()),
                inferred=int(missing.sum()) if root.name != "baseline" else 0,
                original_refit_reuse=len(pool) if root.name == "baseline" else 0,
            )
        )
        pool[f"{family}_mean_log2"] = np.nanmean(species, axis=1)
        ranks.append(percentile_score(-np.nanmean(species, axis=1)))
        for weight in [0.5, 0.75]:
            blended, _ = supported_blend(strain, original, weight)
            pool[f"{family}-w{weight}"] = -np.median(blended, axis=1)
        del state
        torch.cuda.empty_cache()
        print(f"Scored {family}: {len(pool)} sequences", flush=True)
    pool["rankmean"] = np.mean([percentile_score(pool.species.to_numpy()), *ranks], axis=0)
    pool.to_csv(output / "pool.csv.gz", index=False)
    write_json(output / "input_sha256.json", inputs)
    write_json(output / "prediction_reuse.json", accounting)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_scale.json"))
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "features", "apex", "models"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.pool not in config["pools"]:
        raise ValueError("Unregistered pool")
    root = args.output / args.pool
    output = root / args.stage
    dependencies = {
        "prepare": [],
        "features": ["prepare"],
        "apex": ["prepare", "features"],
        "models": ["prepare", "features", "apex"],
    }
    for stage in dependencies[args.stage]:
        previous = root / stage
        manifest = json.loads((previous / "manifest.json").read_text())
        if manifest["config"] != config:
            raise ValueError("Pool processing configuration changed")
        for name, digest in manifest["artifacts_sha256"].items():
            if file_sha256(previous / name) != digest:
                raise ValueError("Pool processing input changed")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_hash = file_sha256(Path(__file__))
    (output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare(config, args.pool, output)
        elif args.stage == "features":
            features(config, args.pool, root, output, args.device)
        elif args.stage == "apex":
            apex(config, args.pool, root, output)
        else:
            models(config, root, output)
    write_json(
        output / "manifest.json",
        dict(
            seconds=time.monotonic() - started,
            source_sha256=source_hash,
            config=config,
            artifacts_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
