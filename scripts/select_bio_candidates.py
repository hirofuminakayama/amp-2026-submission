"""Refit portable endpoint heads and compare validated Tops on complete saved pools."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from compare_competition_pool import tops
from run_competition_bioaccuracy import archive_sources, checked_manifest, finish_stage, write_json
from run_competition_models import fit, predict
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.biofeatures import research_features
from robust_apex_qd.research.biomodels import HC50Bundle, predict_hc50
from robust_apex_qd.research.bioscenarios import (
    biological_rank_scores,
    joint_trials,
    marginal_joint_probability,
    random25_scenarios,
)
from robust_apex_qd.research.competition_models import SPECIES
from robust_apex_qd.research.mic_models import fit_regressor, predict_regressor


def run(
    root: Path,
    mic_root: Path,
    pool_root: Path,
    output: Path,
    mic_models: Path | None = None,
    nested: Path | None = None,
) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for stage in ["split", "features", "hc50-embeddings", "hc50-measured", "nested-selection"]:
        inputs.update(checked_manifest(root / stage / "manifest.json"))
    inputs.update(checked_manifest(mic_root / "manifest.json"))
    nested = nested or root / "nested-selection"
    inputs.update(checked_manifest(nested / "manifest.json"))
    mic_family = "linear8" if mic_models is None else "esm8-exact"
    nested_protocol = json.loads((nested / "protocol.json").read_text())
    if nested_protocol.get("mic_family", "linear8") != mic_family:
        raise ValueError("MIC family and nested HC50 selection disagree")
    widths = {}
    if mic_models is not None:
        inputs.update(checked_manifest(mic_models / "manifest.json"))
        metadata = json.loads((mic_models / "manifest.json").read_text())
        split_hash = file_sha256(root / "split/split_manifest.json")
        if split_hash not in [
            h for p, h in metadata["inputs_sha256"].items() if p.endswith("split_manifest.json")
        ]:
            raise ValueError("Shared endpoint split differs from MIC training")
        choices = json.loads((mic_models / "esm8-exact-selection.json").read_text())
        widths = {int(r["fold"]): r["selected_width"] for r in choices}
    for stage in ["prepare", "features", "models", "libraries"]:
        inputs.update(checked_manifest(pool_root / stage / "manifest.json"))
    config_path = Path("configs/competition_scale.json")
    config = json.loads(config_path.read_text())
    inputs[str(config_path)] = file_sha256(config_path)
    pool = pd.read_csv(pool_root / "models/pool.csv.gz")
    original = read_fasta_sequences(pool_root / "prepare/sequences.fasta")
    embeddings = np.load(pool_root / "features/candidate_embeddings.npy", mmap_mode="r")
    if len(original) != len(embeddings):
        raise ValueError("Pool embedding source row alignment mismatch")
    source_index = {s: i for i, s in enumerate(original)}
    values = np.asarray(embeddings[[source_index[s] for s in pool.sequence]])
    sequences = json.loads((root / "features/sequences.json").read_text())
    train_index = {s: i for i, s in enumerate(sequences)}
    training = np.load(root / "hc50-embeddings/features.npy")
    rows = pd.read_json(root / "split/rows.jsonl", lines=True)
    rows["sequence_index"] = rows.sequence.map(train_index)
    mask = (rows.objective.eq("measured_mic") & rows.exact_regression).to_numpy()
    selection = json.loads((mic_root / "selection.json").read_text())
    alpha = float(np.median([s["alpha"] for s in selection]))
    bundle = output / "endpoint-bundle"
    bundle.mkdir()
    if mic_models is None:
        state = fit({}, rows, training, mask, dict(family="linear8", alpha=alpha), 42, bundle)
        mic, _strains = predict({}, state, values, [], dict(family="linear8"))
        loaded = dict(np.load(bundle / "weights.npz"))
        again = predict({}, loaded, values[::-1], [], dict(family="linear8"))[0][::-1]
    else:
        protocol = json.loads((mic_models / "protocol.json").read_text())["config"]
        selected_rows = rows.loc[mask]
        heads = np.where(
            selected_rows.strain_index >= 0,
            selected_rows.strain_index + 7,
            selected_rows.species_index,
        ).astype(int)
        labels = np.log2(selected_rows.mic_um.to_numpy(float))
        settings = dict(
            width=int(np.median(list(widths.values()))),
            heads=18,
            scale_floor=protocol["scale_floor"],
            device="cpu",
            epochs=protocol["epochs"],
            batch_size=protocol["batch_size"],
            learning_rate=protocol["learning_rate"],
            loss="interval",
            family="esm8-exact",
        )
        state = fit_regressor(
            training[selected_rows.sequence_index], heads, labels, labels, settings, 42
        )
        torch.save(state, bundle / "weights.pt")
        mic = predict_regressor(state, values)[0][:, :7]
        loaded = torch.load(bundle / "weights.pt", weights_only=True)
        again = predict_regressor(loaded, values[::-1])[0][::-1, :7]
    tolerance = 1e-12 if mic_models is None else 2e-5
    np.testing.assert_allclose(mic, again, atol=tolerance, rtol=0)
    first_top = set(np.argsort(np.median(mic, axis=1), kind="stable")[:100])
    second_top = set(np.argsort(np.median(again, axis=1), kind="stable")[:100])
    if first_top != second_top:
        raise ValueError("MIC permutation changes the unconstrained Top100 set")
    audits = json.loads((nested / "selection_audit.json").read_text())
    votes = Counter(a["selected_arm"] for a in audits)
    arm = sorted(votes, key=lambda a: (-votes[a], a))[0]
    hc_path = root / "hc50-measured" / arm / "refit.json"
    model = HC50Bundle.model_validate_json(hc_path.read_text())
    (bundle / "hc50.json").write_bytes(hc_path.read_bytes())
    if arm.startswith("esm8"):
        features = values
    else:
        features = np.array(
            [list(research_features(s, boman="standard").values()) for s in pool.sequence]
        )
    hc = predict_hc50(model, features, model.feature_sha256)
    # Species residuals are computed from shared-fold OOF species heads, not strain offsets.
    splits = json.loads((root / "split/split_manifest.json").read_text())
    residual_records = []
    for fold in range(5):
        cohort = rows[mask & rows.homology_fold.eq(fold)]
        if mic_models is None:
            state = dict(np.load(mic_root / "fits" / f"outer{fold}-selected/weights.npz"))
            p = predict({}, state, training[cohort.sequence_index], [], dict(family="linear8"))[0]
        else:
            path = mic_models / "fits/esm8-exact" / f"outer{fold}-selected-w{widths[fold]}-s42"
            inputs.update(checked_manifest(path / "manifest.json"))
            state = torch.load(path / "weights.pt", weights_only=True)
            p = predict_regressor(state, training[cohort.sequence_index])[0][:, :7]
        for i, r in enumerate(cohort.itertuples()):
            residual_records.append(
                dict(
                    sequence=r.sequence,
                    species=r.species,
                    error=float(np.log2(r.mic_um) - p[i, int(r.species_index)]),
                )
            )
    residual_frame = pd.DataFrame(residual_records)
    residuals = [
        residual_frame[residual_frame.species == s].groupby("sequence").error.mean().to_numpy()
        for s in SPECIES
    ]
    hc_oof = pd.read_csv(root / "hc50-measured/measured_hc50_oof.csv")
    hc_oof = hc_oof[(hc_oof.arm == arm) & hc_oof.relation.eq("=")].copy()
    hc_oof["error"] = np.log2(hc_oof.value_um) - hc_oof.prediction_log2_um
    hc_errors = hc_oof.groupby("sequence").error.mean().to_numpy()
    np.savez_compressed(
        bundle / "residuals.npz", hc50=hc_errors, **{f"mic{i}": r for i, r in enumerate(residuals)}
    )
    probabilities = marginal_joint_probability(mic, hc, residuals, hc_errors)
    for name, scores in biological_rank_scores(mic, probabilities, pool.B1.to_numpy()).items():
        pool[name] = scores
    pool["bio_hc50_log2_um"] = hc
    for i in range(7):
        pool[f"bio_mic{i}"] = mic[:, i]
        pool[f"bio_joint{i}"] = probabilities[:, i]
    (output / "models").mkdir()
    pool.to_csv(output / "models/pool.csv.gz", index=False)
    (output / "libraries").mkdir()
    library = pool_root / "libraries/L2.fasta"
    (output / "libraries/L2.fasta").write_bytes(library.read_bytes())
    requests = [
        dict(id="library-L2", library="L2", ranker="B1", constraint="current", factor="control")
    ]
    requests.extend(
        dict(id=name, library="L2", ranker=name, constraint="current", factor="biological endpoint")
        for name in ["bio-MIC", "bio-joint", "bio-MIC-apex50"]
    )
    write_json(
        bundle / "bundle.json",
        dict(
            schema_version=2,
            mic_family=mic_family,
            mic_alpha=alpha if mic_models is None else None,
            hc50_arm=arm,
            hc50_selection="most frequent inner-selected family, ties id",
            hc50_votes=dict(votes),
            feature_dimension=320,
            feature_name="esm2_t6_8M_UR50D mean residue",
            hc50_feature_sha256=model.feature_sha256,
            species=list(SPECIES),
            units="log2_uM",
            ratio=8,
            mic_threshold_um=16,
            split_sha256=file_sha256(root / "split/split_manifest.json"),
            reload_permutation_atol=tolerance,
            artifact_sha256={p.name: file_sha256(p) for p in bundle.iterdir() if p.is_file()},
        ),
    )
    write_json(
        output / "protocol.json",
        dict(
            config=config,
            requests=requests,
            candidate_prediction_count=len(pool),
            pool=str(pool_root),
            selection_scope="existing L2 library/current constraints; fixed library diagnostic",
            covered_before_selection="every valid unique pool sequence; no B1 inference prefilter",
            probability="conditional OOF residual scenario; wet-lab calibration unverified",
            adopted=False,
            raw_mic_permutation_top100_overlap=len(first_top & second_top),
            shared_groups=len(set(splits["groups"].values())),
        ),
    )
    destination = output / "tops"
    destination.mkdir()
    tops(config, pool_root.name, output, destination, requests)
    scenarios = []
    index = {s: i for i, s in enumerate(pool.sequence)}
    for request in requests:
        path = destination / request["id"] / "top.fasta"
        if not path.exists():
            continue
        selected = read_fasta_sequences(path)
        positions = [index[s] for s in selected]
        groups = pool.embedding_cluster.iloc[positions].astype(str).tolist()
        for ratio in [4, 8, 16]:
            for correlation in [-0.5, 0.0, 0.5]:
                for dependence in ["independent", "cluster", "species"]:
                    for seed in [42, 43, 44]:
                        trials = joint_trials(
                            mic[positions],
                            hc[positions],
                            residuals,
                            hc_errors,
                            groups=groups,
                            ratio=ratio,
                            correlation=correlation,
                            dependence=dependence,
                            seed=seed,
                        )
                        scenarios.append(
                            dict(
                                candidate=request["id"],
                                ratio=ratio,
                                correlation=correlation,
                                dependence=dependence,
                                seed=seed,
                                **random25_scenarios(trials, seed=seed),
                            )
                        )
    write_json(output / "random25_scenarios.json", scenarios)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mic-baselines", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mic-models", type=Path)
    parser.add_argument("--nested-selection", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / f"pool-{args.pool.name}"
    fresh_output(output, [args.pool, args.mic_baselines])
    started = time.monotonic()
    inputs = archive_sources(
        output,
        [
            Path(__file__),
            Path("scripts/compare_competition_pool.py"),
            Path("scripts/run_competition_models.py"),
            Path("scripts/run_competition_bioaccuracy.py"),
            Path("src/robust_apex_qd/research/mic_models.py"),
            Path("uv.lock"),
            *Path("src/robust_apex_qd/research").glob("bio*.py"),
        ],
    )
    with threadpool_limits(2):
        torch.set_num_threads(2)
        inputs.update(
            run(
                args.root,
                args.mic_baselines,
                args.pool,
                output,
                args.mic_models,
                args.nested_selection,
            )
        )
    finish_stage(output, inputs, started)


if __name__ == "__main__":
    main()
