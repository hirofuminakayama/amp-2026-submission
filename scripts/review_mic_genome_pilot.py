"""Recompute genome-pilot membership, OOF parity and frozen-pool fallback checks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from run_mic_genome_pilot import ARMS, load_pair_data, read_config
from run_mic_research import checked_manifest, finish_stage, write_json
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.competition_models import SPECIES
from robust_apex_qd.research.genome_pairs import (
    fallback_reason,
    fit_categories,
    pair_features,
    pair_masks,
    resolve_genome,
)
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.mic_models import predict_regressor


def review(root: Path, output: Path) -> None:
    started = time.monotonic()
    fresh_output(output, [root / "prepare", root / "models", root / "handoff"])
    capture_execution(output)
    inputs = {}
    manifests = sorted(root.glob("*/manifest.json")) + sorted(root.glob("models/*/manifest.json"))
    for path in manifests:
        inputs.update(checked_manifest(path))
        manifest = json.loads(path.read_text())
        for name, expected in manifest.get("inputs_sha256", {}).items():
            if file_sha256(Path(name)) != expected:
                raise ValueError(f"Changed input: {name}")
            inputs[name] = expected
    for path in root.glob("*/execution_manifest.json"):
        inputs.update(checked_manifest(path))
    config = read_config(root / "prepare/config.json")
    for stage in ["models", "handoff"]:
        if read_config(root / stage / "config.json") != config:
            raise ValueError("Configuration mismatch between stages")
    # Existing ESM cache order is defined by its preparation FASTA.
    source_fasta = Path(config["peptide_features"]).parent.parent / "prepare/sequences.fasta"
    sequences = json.loads((Path(config["prepared"]) / "sequences.json").read_text())
    if sequences != [str(r.seq) for r in SeqIO.parse(source_fasta, "fasta")]:
        raise ValueError("ESM feature sequence order mismatch")
    inputs[str(source_fasta)] = file_sha256(source_fasta)
    rows, genomes = load_pair_data(root / "prepare")
    peptide = np.load(config["peptide_features"])[rows.sequence_index.to_numpy(int)]
    oof = pd.read_json(root / "models/oof.jsonl", lines=True)
    costs = pd.read_csv(root / "models/costs.csv")
    if len(costs) != 4 * (5 + config["strain_folds"] + 5 * config["strain_folds"]):
        raise ValueError("Incomplete fit schedule")
    max_error = 0.0
    for fit in costs.to_dict("records"):
        genome, assay, aux = ARMS[fit["arm"]]
        parts = fit["fold"].split("-")
        p = None if parts[-2] == "pNone" else int(parts[-2][1:])
        s = None if parts[-1] == "gNone" else int(parts[-1][1:])
        training, validation = pair_masks(rows, fit["regime"], p, s)
        if not aux:
            training &= rows.primary.to_numpy(bool)
        folder = root / "models" / fit["fold"]
        member = json.loads((folder / "membership.json").read_text())
        tr, va = rows[training], rows[validation]
        if (
            member["train"] != tr.observation_id.tolist()
            or member["validation"] != va.observation_id.tolist()
        ):
            raise ValueError("Incorrect train/validation membership")
        if fit["regime"] in {"peptide", "both"} and set(tr.homology_group) & set(va.homology_group):
            raise ValueError("Homology leakage")
        if fit["regime"] in {"strain", "both"} and set(tr.genome_accession) & set(
            va.genome_accession
        ):
            raise ValueError("Assembly leakage")
        bundle = torch.load(folder / "model.pt", weights_only=True)
        categories = fit_categories(tr, assay)
        if bundle["categories"] != categories or bundle["settings"] != config["settings"]:
            raise ValueError("Unexpected learned vocabulary or model settings")
        scaler = StandardScaler().fit(
            pair_features(peptide[training], tr, genomes, categories, genome)
        )
        np.testing.assert_allclose(bundle["feature_mean"].numpy(), scaler.mean_)
        np.testing.assert_allclose(bundle["feature_scale"].numpy(), scaler.scale_)
        for masked in [False, True] if assay else [False]:
            x = pair_features(
                peptide[validation], va, genomes, categories, genome, mask_assay=masked
            )
            actual = predict_regressor(bundle, x)[0][:, 0]
            arm = fit["arm"] + ("-masked" if masked else "")
            recorded = oof[oof.fold.eq(fit["fold"]) & oof.arm.eq(arm)]
            if recorded.observation_id.tolist() != va.observation_id.tolist():
                raise ValueError("Incorrect OOF sequence")
            np.testing.assert_allclose(actual, recorded.prediction.to_numpy(), atol=1e-6)
            max_error = max(max_error, float(np.max(np.abs(actual - recorded.prediction))))
    for _key, group in oof.groupby(["arm", "regime"]):
        regime = group.regime.iloc[0]
        expected = rows[rows.primary]
        if regime != "peptide":
            expected = expected[expected.genome_status.eq("exact_label")]
        if group.observation_id.duplicated().any() or set(group.observation_id) != set(
            expected.observation_id
        ):
            raise ValueError("Incorrect primary evaluation coverage")
    sequences = pd.read_csv(config["pool_sequences"]).sequence.tolist()
    frame = pd.read_csv(root / "handoff/predictions.csv.gz", keep_default_na=False)
    with np.load(config["apex_pool"]) as archive:
        lookup = {s: i for i, s in enumerate(archive["sequences"].tolist())}
        apex = np.log2(archive["mic_uM"].mean(1))[[lookup[s] for s in sequences]]
    with np.load(root / "handoff/raw_pair_predictions.npz") as archive:
        if archive["sequences"].tolist() != sequences or archive["targets"].tolist() != list(
            APEX_PATHOGENS
        ):
            raise ValueError("Incorrect raw pair prediction keys")
        raw = archive["log2_mic"]
    bundle = torch.load(root / "handoff/model.pt", weights_only=True)
    pool = np.load(config["pool_features"])
    sample = np.unique(np.linspace(0, len(pool) - 1, 127).astype(int))
    support = json.loads((root / "handoff/target_support.json").read_text())
    for head, entry in enumerate(support):
        mapping = resolve_genome(entry["target"], entry["species"], config)
        reason = fallback_reason(mapping, genomes)
        part = frame[frame.target_id.eq(entry["target"])].set_index("sequence").loc[sequences]
        if part.index.duplicated().any() or not part.fallback_reason.eq(reason).all():
            raise ValueError("Incorrect fallback provenance")
        if not part.supported.eq(not bool(reason)).all():
            raise ValueError("Incorrect strain support flag")
        np.testing.assert_allclose(part.prediction, apex[:, head] if reason else raw[:, head])
        targets = pd.DataFrame(
            dict(
                species=[mapping.species] * len(sample),
                genome_accession=mapping.accession,
                genome_status=mapping.status,
            )
        )
        x = pair_features(pool[sample], targets, genomes, bundle["categories"], True)
        np.testing.assert_allclose(
            predict_regressor(bundle, x)[0][:, 0], raw[sample, head], atol=1e-5
        )
    if len(frame) != len(sequences) * len(APEX_PATHOGENS) or not frame.unit.eq("log2_uM").all():
        raise ValueError("Invalid adapter output contract")
    write_json(
        output / "verification.json",
        dict(
            fits=len(costs),
            checked_files=len(inputs),
            primary_rows=int(rows.primary.sum()),
            auxiliary_rows=int((~rows.primary).sum()),
            oof_max_reload_error=max_error,
            sequences=len(sequences),
            prediction_rows=len(frame),
            supported_rows=int(frame.supported.sum()),
            development_only=True,
            adopted=False,
        ),
    )
    comparison = pd.read_csv(root / "models/comparison.csv")
    primary = comparison[comparison.cohort.eq("primary7")]
    table = [
        "| Arm | Peptide-unseen | Strain-unseen | Both-unseen |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in sorted(primary.arm.unique()):
        scores = primary[primary.arm.eq(arm)].set_index("regime").macro_mae
        table.append(
            f"| {arm} | {scores['peptide']:.4f} | {scores['strain']:.4f} | {scores['both']:.4f} |"
        )
    counts = rows[rows.primary & rows.genome_status.eq("exact_label")].groupby("species").size()
    phases = {
        stage: json.loads((root / stage / "manifest.json").read_text())
        for stage in ["prepare", "models", "handoff"]
    }
    report = [
        "# Genome pair MIC pilot",
        "",
        "Fixed-setting, single-seed development comparison; no independent holdout claim.",
        "",
        "Macro MAE in log2 µM (lower is better):",
        "",
        *table,
        "",
        f"Peptide-unseen: {int(rows.primary.sum()):,} measured competition rows.",
        f"Strain/both-unseen: {int(counts.sum()):,} rows across {len(counts)} species.",
        "Zero exact-label coverage: " + ", ".join(sorted(set(SPECIES) - set(counts.index))) + ".",
        "Zero-coverage species are not included in macro MAE.",
        "Compare arms within a regime; evaluation populations differ across regimes.",
        "",
        f"Auxiliary training: {int((~rows.primary).sum())} measurements; exclusion checks passed.",
        f"Fits: {len(costs)} plus one full-data genome refit. CPU only; fixed peptide embeddings.",
        "Stage elapsed seconds: "
        + ", ".join(f"{s}={m['seconds']:.2f}" for s, m in phases.items())
        + ".",
        "Parallel stages share CPU resources; elapsed times are not additive allocations.",
        "",
        f"Common adapter: {len(sequences):,} sequences, {len(frame):,} strain rows, "
        f"{int(frame.supported.sum()):,} supported pair rows; other rows retain APEX predictions.",
        "Assembly-label matches do not prove identity with the measured laboratory isolate.",
        "Raw pair output and explicit mapping/fallback metadata are preserved in handoff/.",
        "",
        "Assess genome and assay effects against the species-ID and no-auxiliary controls.",
        "Review mapping coverage and condition provenance before adding proteome features.",
        "This pilot does not automatically adopt a predictor. No defaults changed.",
        "",
        "Verification recomputed split membership, training-only scaling, OOF checkpoint parity,",
        "full-pool fallback and sampled inference parity. See verification.json and manifest.json.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    finish_stage(output, inputs, started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        review(args.root, args.output)


if __name__ == "__main__":
    main()
