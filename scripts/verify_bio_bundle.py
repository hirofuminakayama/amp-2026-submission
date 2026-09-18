"""Verify label-free biological bundle inference against a saved generated-pool prediction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_competition_bioaccuracy import archive_sources, checked_manifest, finish_stage, write_json
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256, row_mapping_sha256
from robust_apex_qd.research.biobundle import load_bio_bundle, predict_bio_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.bundle, args.pool, args.expected.parent])
    started = time.monotonic()
    inputs = archive_sources(
        args.output,
        [
            Path(__file__),
            Path("scripts/run_competition_bioaccuracy.py"),
            Path("uv.lock"),
            *Path("src/robust_apex_qd/research").glob("bio*.py"),
        ],
    )
    for stage in ["prepare", "features"]:
        inputs.update(checked_manifest(args.pool / stage / "manifest.json"))
    inputs[str(args.expected)] = file_sha256(args.expected)
    inputs[str(args.bundle / "bundle.json")] = file_sha256(args.bundle / "bundle.json")
    bundle = load_bio_bundle(args.bundle)
    inputs.update({str(args.bundle / p): h for p, h in bundle.manifest.artifact_sha256.items()})
    candidates = pd.read_csv(args.pool / "prepare/candidates.csv.gz")
    manifest = json.loads((args.pool / "features/embedding_manifest.json").read_text())
    if (
        manifest["model_name"] != "esm2_t6_8M_UR50D"
        or manifest["model_revision"] != "fair-esm-2.0.0"
    ):
        raise ValueError("Embedding model identity mismatch")
    direct_mapping = (
        row_mapping_sha256(candidates.candidate_id.tolist(), candidates.sequence.tolist())
        == manifest["candidate_row_mapping_sha256"]
    )
    embeddings = np.load(args.pool / "features/candidate_embeddings.npy", mmap_mode="r")
    if not direct_mapping:
        # Baseline processing retains an original manifest while remapping its saved rows.
        source_hashes = json.loads((args.pool / "features/input_sha256.json").read_text())
        table_paths = [Path(p) for p in source_hashes if Path(p).name == "candidates.csv.gz"]
        vector_paths = [
            Path(p) for p in source_hashes if Path(p).name == "candidate_embeddings.npy"
        ]
        if len(table_paths) != 1 or len(vector_paths) != 1:
            raise ValueError("Unambiguous original embedding row provenance required")
        table, vector = table_paths[0], vector_paths[0]
        for path in [table, vector]:
            if file_sha256(path) != source_hashes[str(path)]:
                raise ValueError("Original embedding provenance hash mismatch")
            inputs[str(path)] = source_hashes[str(path)]
        original = pd.read_csv(table)
        if (
            row_mapping_sha256(original.candidate_id.tolist(), original.sequence.tolist())
            != manifest["candidate_row_mapping_sha256"]
        ):
            raise ValueError("Original embedding candidate row mapping mismatch")
        if file_sha256(vector) != manifest["candidate_embeddings_sha256"]:
            raise ValueError("Original embedding array hash mismatch")
        first = original.drop_duplicates("sequence").set_index("sequence").raw_order
        expected_vectors = np.load(vector, mmap_mode="r")[
            first.loc[candidates.sequence].to_numpy(int)
        ]
        np.testing.assert_array_equal(embeddings, expected_vectors)
    expected = pd.read_csv(args.expected)
    if expected.sequence.duplicated().any():
        raise ValueError("Duplicate expected prediction sequence")
    index = {s: i for i, s in enumerate(candidates.sequence)}
    sequences = expected.sequence.tolist()
    x = np.asarray(embeddings[[index[s] for s in sequences]])
    errors = {}
    tolerances = dict(
        mic_log2_um=1e-12 if bundle.manifest.mic_family == "linear8" else 2e-5,
        hc50_log2_um=1e-12,
        joint_probability=1e-12,
    )
    with threadpool_limits(2):
        prediction = predict_bio_bundle(
            bundle, sequences, x, embedding_feature_name=bundle.manifest.feature_name
        )
        expected_arrays = dict(
            mic_log2_um=expected[[f"bio_mic{i}" for i in range(7)]].to_numpy(),
            hc50_log2_um=expected.bio_hc50_log2_um.to_numpy(),
            joint_probability=expected[[f"bio_joint{i}" for i in range(7)]].to_numpy(),
        )
        for name, values in prediction.items():
            np.testing.assert_allclose(values, expected_arrays[name], atol=tolerances[name], rtol=0)
            errors[name] = float(np.abs(values - expected_arrays[name]).max())
        # Deterministic seed-selected controls are independent of score and label values.
        selected = np.random.default_rng(42).choice(len(x), size=min(64, len(x)), replace=False)
        reverse = predict_bio_bundle(
            load_bio_bundle(args.bundle),
            [sequences[i] for i in selected[::-1]],
            x[selected[::-1]],
            embedding_feature_name=bundle.manifest.feature_name,
        )
        for name, values in reverse.items():
            np.testing.assert_allclose(
                values[::-1], prediction[name][selected], atol=tolerances[name], rtol=0
            )
    np.savez_compressed(args.output / "predictions.npz", allow_pickle=False, **prediction)
    write_json(
        args.output / "inference_parity.json",
        dict(
            candidates=len(sequences),
            maximum_absolute_error=errors,
            absolute_tolerances=tolerances,
            reload_permutation_controls=len(selected),
            training_labels_loaded=False,
            generated_pool_inference=True,
            scope="load/predict/save parity; generation and entrypoint integration pending",
            embedding_identity="named model/revision and saved-array hash; checkpoint not rerun",
            original_mapping_verified=not direct_mapping,
        ),
    )
    finish_stage(args.output, inputs, started)


if __name__ == "__main__":
    main()
