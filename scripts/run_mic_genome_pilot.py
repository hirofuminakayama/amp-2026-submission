"""Versioned genome preparation, fixed-setting pair pilot and frozen-pool handoff."""

import argparse
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from run_mic_research import checked_manifest, finish_stage, write_json
from threadpoolctl import threadpool_limits
from train_mic_models import evaluate

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.competition_models import SPECIES, STRAIN_SPECIES
from robust_apex_qd.research.genome_pairs import (
    dna_fourmers,
    fallback_reason,
    fit_categories,
    genome_prediction_records,
    pair_features,
    pair_masks,
    resolve_genome,
)
from robust_apex_qd.research.mic_data import measured_bounds
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.mic_models import fit_regressor, predict_regressor

ARMS = {
    "species": (False, False, True),
    "genome": (True, False, True),
    "genome-assay": (True, True, True),
    "genome-noaux": (True, False, False),
}


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    accessions = [a["accession"] for a in config["assemblies"]]
    labels = [label for a in config["assemblies"] for label in a["labels"]]
    if len(set(accessions)) != len(accessions) or len(set(labels)) != len(labels):
        raise ValueError("Duplicate accession or label in genome configuration")
    if not set(config["species_references"].values()) <= set(accessions):
        raise ValueError("Reference genomes must be explicitly registered")
    if config["settings"]["heads"] != 1 or config["settings"]["device"] != "cpu":
        raise ValueError("The fixed pair pilot uses one CPU regression head")
    return config


def fetch(config: dict[str, Any], output: Path) -> None:
    fresh_output(output, [Path(config["prepared"]), Path(config["lineage"])])
    for assembly in config["assemblies"]:
        accession = assembly["accession"]
        for suffix, endpoint in [
            (".json", "/dataset_report"),
            (".zip", "/download?include_annotation_type=GENOME_FASTA"),
        ]:
            subprocess.run(
                [
                    "curl",
                    "-fLsS",
                    "--max-time",
                    "120",
                    "--retry",
                    "2",
                    f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{accession}{endpoint}",
                    "-o",
                    str(output / (accession + suffix)),
                ],
                check=True,
            )
        print(accession, flush=True)


def prepare(config: dict[str, Any], sources: Path, output: Path, config_path: Path) -> None:
    started = time.monotonic()
    prepared = Path(config["prepared"])
    lineage = Path(config["lineage"])
    fresh_output(output, [prepared, lineage, sources])
    capture_execution(output)
    write_json(output / "config.json", config)
    inputs = {str(config_path): file_sha256(config_path)}
    for path in [
        prepared / "rows.jsonl",
        prepared / "sequences.json",
        prepared / "split_manifest.json",
        lineage / "observations.jsonl",
        Path(config["peptide_features"]),
        Path(config["peptide_sequences"]),
    ]:
        inputs[str(path)] = file_sha256(path)
    genomes, provenance = {}, []
    for entry in config["assemblies"]:
        accession = entry["accession"]
        archive = sources / (accession + ".zip")
        inputs[str(archive)] = file_sha256(archive)
        with zipfile.ZipFile(archive) as package:
            metadata = [
                json.loads(line)
                for line in package.read("ncbi_dataset/data/assembly_data_report.jsonl")
                .decode()
                .splitlines()
            ]
            if len(metadata) != 1 or metadata[0]["accession"] != accession:
                raise ValueError("Downloaded assembly differs from registered accession")
            names = [n for n in package.namelist() if n.endswith("_genomic.fna")]
            if len(names) != 1 or f"/{accession}/" not in names[0]:
                raise ValueError("Expected one genome FASTA in the registered assembly directory")
            # Read members directly: no archive paths are extracted to the filesystem.
            fasta = package.read(names[0]).decode("ascii")
            records = list(SeqIO.parse(io.StringIO(fasta), "fasta"))
            genomes[accession] = dna_fourmers(str(r.seq) for r in records)
        provenance.append(
            dict(
                **entry,
                metadata=metadata[0],
                contigs=len(records),
                bases=sum(len(r) for r in records),
                archive_sha256=inputs[str(archive)],
                source_url=f"https://www.ncbi.nlm.nih.gov/datasets/genome/{accession}/",
                terms_url="https://www.ncbi.nlm.nih.gov/home/about/policies/",
                terms=(
                    "NCBI imposes no additional molecular-data use restrictions; "
                    "third-party rights are not waived."
                ),
                experimental_isolate_verified=False,
            )
        )
    np.savez_compressed(output / "genomes.npz", **genomes)
    write_json(output / "assembly_manifest.json", provenance)
    rows = pd.read_json(prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].copy()
    rows["primary"] = True
    sequences = json.loads((prepared / "sequences.json").read_text())
    if sequences != [str(r.seq) for r in SeqIO.parse(config["peptide_sequences"], "fasta")]:
        raise ValueError("Peptide feature order differs from the evaluation sequence order")
    embedding = np.load(config["peptide_features"])
    if len(embedding) != len(sequences) or not np.isfinite(embedding).all():
        raise ValueError("Invalid fixed peptide feature cache")
    index = {s: i for i, s in enumerate(sequences)}
    raw = pd.read_json(lineage / "observations.jsonl", lines=True)
    eligible = (
        raw.objective.eq("measured_mic")
        & raw.species.isin(config["auxiliary_species"])
        & raw.sequence.isin(index)
        & raw.chemical_form.eq("reported_linear_free")
        & raw.duplicate_of.isna()
        & (raw.lower_um.notna() | raw.upper_um.notna())
    )
    # Excluded primary-species observations never enter through the auxiliary pathway.
    if set(config["auxiliary_species"]) & set(SPECIES):
        raise ValueError("Auxiliary species must be disjoint from competition species")
    auxiliary = raw[eligible].copy()
    auxiliary["primary"] = False
    auxiliary["exact_regression"] = auxiliary.exact_mic & auxiliary.mic_um.notna()
    rows = pd.concat([rows, auxiliary], ignore_index=True)
    if rows.observation_id.duplicated().any():
        raise ValueError("Duplicate observation identifiers in pair data")
    _, _, usable = measured_bounds(rows)
    if not usable.all():
        raise ValueError("Unusable MIC bound in pair data")
    split = json.loads((prepared / "split_manifest.json").read_text())
    rows["sequence_index"] = rows.sequence.map(index).astype(int)
    rows["peptide_fold"] = rows.sequence.map(split["outer"]).astype(int)
    rows["homology_group"] = rows.sequence.map(split["groups"])
    targets = rows[["target", "species"]].drop_duplicates().to_dict("records")
    targets += [
        dict(target=t, species=SPECIES[STRAIN_SPECIES[i]]) for i, t in enumerate(APEX_PATHOGENS)
    ]
    mappings = {r["target"]: resolve_genome(r["target"], r["species"], config) for r in targets}
    write_json(output / "target_manifest.json", [m.model_dump() for m in mappings.values()])
    rows["genome_accession"] = rows.target.map({t: m.accession for t, m in mappings.items()})
    rows["genome_status"] = rows.target.map({t: m.status for t, m in mappings.items()})
    # Deterministic accession groups, stratified by reported species without consulting MIC.
    genome_folds = {}
    exact = rows[rows.genome_status.eq("exact_label")]
    for _species, group in exact.groupby("species", sort=True):
        for i, accession in enumerate(sorted(group.genome_accession.unique())):
            genome_folds[accession] = i % config["strain_folds"]
    rows["genome_fold"] = rows.genome_accession.map(genome_folds).fillna(-1).astype(int)
    rows.to_json(output / "rows.jsonl", orient="records", lines=True)
    coverage = (
        rows.groupby(["primary", "species", "genome_status"])
        .agg(
            rows=("observation_id", "size"),
            targets=("target", "nunique"),
            assemblies=("genome_accession", "nunique"),
            peptides=("sequence", "nunique"),
        )
        .reset_index()
    )
    coverage.to_csv(output / "coverage.csv", index=False)
    rejected = raw[raw.objective.eq("measured_mic") & ~raw.species.isin(SPECIES)].copy()
    rejected["included_auxiliary"] = rejected.observation_id.isin(auxiliary.observation_id)
    rejected.groupby(
        ["species", "chemical_form", "included_auxiliary"], dropna=False
    ).size().rename("rows").to_csv(output / "auxiliary_inventory.csv")
    write_json(
        output / "splits.json",
        dict(
            genome_folds=genome_folds,
            peptide_split_sha256=inputs[str(prepared / "split_manifest.json")],
            strain_interpretation=(
                "unseen versioned assembly with matching label, not verified wet-lab isolate"
            ),
            independent_holdout=False,
        ),
    )
    finish_stage(
        output,
        inputs,
        started,
        primary_rows=int(rows.primary.sum()),
        auxiliary_rows=len(auxiliary),
        assemblies=len(genomes),
    )


def load_pair_data(prepared: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = pd.read_json(prepared / "rows.jsonl", lines=True)
    with np.load(prepared / "genomes.npz") as archive:
        genomes = {k: archive[k] for k in archive.files}
    return rows, genomes


def validate_prepared(config: dict[str, Any], prepared: Path) -> dict[str, str]:
    inputs = checked_manifest(prepared / "manifest.json")
    manifest = json.loads((prepared / "manifest.json").read_text())
    inputs.update(verify_hashes(manifest["inputs_sha256"]))
    if config != read_config(prepared / "config.json"):
        raise ValueError("Use the prepared snapshot configuration for training and inference")
    return inputs


def score_rows(rows: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    return evaluate(rows, prediction, np.full(len(rows), np.nan))


def train(config: dict[str, Any], prepared: Path, output: Path) -> None:
    started = time.monotonic()
    inputs = validate_prepared(config, prepared)
    feature_path = Path(config["peptide_features"])
    inputs[str(feature_path)] = file_sha256(feature_path)
    fresh_output(output, [prepared, feature_path])
    capture_execution(output)
    write_json(output / "config.json", config)
    rows, genomes = load_pair_data(prepared)
    embedding = np.load(feature_path)[rows.sequence_index.to_numpy(int)]
    lo, hi, _ = measured_bounds(rows)
    folds = sorted(rows.peptide_fold.unique().tolist())
    strain_folds = list(range(config["strain_folds"]))
    schedule = [("peptide", p, None) for p in folds]
    schedule += [("strain", None, s) for s in strain_folds]
    schedule += [("both", p, s) for p in folds for s in strain_folds]
    predictions, costs = [], []
    torch.set_num_threads(config["cpu_threads"])
    for arm, (genome, assay, aux) in ARMS.items():
        for regime, p, s in schedule:
            fit_start = time.monotonic()
            training, validation = pair_masks(rows, regime, p, s)
            if not aux:
                training &= rows.primary.to_numpy(bool)
            if not validation.any() or not training.any():
                raise ValueError("Empty train/validation partition")
            train_rows, valid_rows = rows[training], rows[validation]
            if regime in {"peptide", "both"} and set(train_rows.homology_group) & set(
                valid_rows.homology_group
            ):
                raise ValueError("Peptide homology group leakage")
            if regime in {"strain", "both"} and set(train_rows.genome_accession) & set(
                valid_rows.genome_accession
            ):
                raise ValueError("Genome accession leakage")
            categories = fit_categories(train_rows, assay)
            features = pair_features(embedding, rows, genomes, categories, genome)
            bundle = fit_regressor(
                features[training],
                np.zeros(training.sum(), dtype=int),
                lo[training],
                hi[training],
                config["settings"],
                config["seed"],
            )
            name = f"{arm}-{regime}-p{p}-g{s}"
            destination = output / name
            destination.mkdir()
            bundle["categories"] = categories
            bundle["genome"] = genome
            torch.save(bundle, destination / "model.pt")
            write_json(
                destination / "membership.json",
                dict(
                    train=train_rows.observation_id.tolist(),
                    validation=valid_rows.observation_id.tolist(),
                    train_groups=sorted(train_rows.homology_group.unique().tolist()),
                    validation_groups=sorted(valid_rows.homology_group.unique().tolist()),
                    train_accessions=sorted(train_rows.genome_accession.dropna().unique().tolist()),
                    validation_accessions=sorted(
                        valid_rows.genome_accession.dropna().unique().tolist()
                    ),
                    categories=categories,
                ),
            )
            conditions = [False, True] if assay else [False]
            for mask_assay in conditions:
                x = pair_features(
                    embedding[validation],
                    valid_rows,
                    genomes,
                    categories,
                    genome,
                    mask_assay=mask_assay,
                )
                mean, _ = predict_regressor(bundle, x)
                pred = valid_rows[
                    [
                        "observation_id",
                        "species",
                        "mic_um",
                        "exact_regression",
                        "active16",
                        "objective",
                        "lower_um",
                        "upper_um",
                        "medium",
                        "cfu",
                        "sequence",
                        "genome_status",
                    ]
                ].copy()
                pred["arm"] = arm + ("-masked" if mask_assay else "")
                pred["regime"] = regime
                pred["fold"] = name
                pred["prediction"] = mean[:, 0]
                predictions.append(pred)
            cost: dict[str, Any] = dict(
                arm=arm,
                regime=regime,
                fold=name,
                training=len(train_rows),
                auxiliary=int((~train_rows.primary).sum()),
                validation=len(valid_rows),
                validation_species_unseen=int((~valid_rows.species.isin(train_rows.species)).sum()),
                seconds=time.monotonic() - fit_start,
            )
            finish_stage(
                destination, {}, fit_start, **{k: v for k, v in cost.items() if k != "seconds"}
            )
            costs.append(cost)
            print(name, round(cost["seconds"], 2), flush=True)
    oof = pd.concat(predictions, ignore_index=True)
    if oof.duplicated(["arm", "regime", "observation_id"]).any():
        raise ValueError("Repeated primary validation prediction")
    oof.to_json(output / "oof.jsonl", orient="records", lines=True)
    reports = []
    for (arm, regime), group in oof.groupby(["arm", "regime"]):
        reports.append(
            dict(
                arm=arm,
                regime=regime,
                cohort="primary7",
                **score_rows(group, group.prediction.to_numpy()),
            )
        )
        for species in SPECIES:
            subset = group[group.species.eq(species)]
            reports.append(
                dict(
                    arm=arm,
                    regime=regime,
                    cohort=species,
                    **score_rows(subset, subset.prediction.to_numpy()),
                )
            )
        for missing in [False, True]:
            subset = group[(group.medium.isna() | group.cfu.isna()) == missing]
            reports.append(
                dict(
                    arm=arm,
                    regime=regime,
                    cohort=f"assay_missing={missing}",
                    **score_rows(subset, subset.prediction.to_numpy()),
                )
            )
    pd.DataFrame(reports).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(costs).to_csv(output / "costs.csv", index=False)
    # Fixed architecture/seed/epochs: the pilot compares arms, with no outer-fold tuning.
    write_json(
        output / "protocol.json",
        dict(
            settings=config["settings"],
            seed=config["seed"],
            selection="none; predeclared fixed-setting development pilot",
            independent_holdout=False,
            handoff_arm="genome",
            handoff_reason="predeclared non-assay interface demonstration; not adoption",
            proteome_plm="deferred until coverage and pilot errors justify a separate experiment",
        ),
    )
    finish_stage(output, inputs, started, fits=len(costs))


def handoff(config: dict[str, Any], prepared: Path, output: Path) -> None:
    started = time.monotonic()
    inputs = validate_prepared(config, prepared)
    for key in ["peptide_features", "pool_features", "pool_sequences", "apex_pool"]:
        inputs[config[key]] = file_sha256(Path(config[key]))
    fresh_output(
        output,
        [
            prepared,
            *[
                Path(config[k])
                for k in ["peptide_features", "pool_features", "pool_sequences", "apex_pool"]
            ],
        ],
    )
    capture_execution(output)
    write_json(output / "config.json", config)
    rows, genomes = load_pair_data(prepared)
    peptide = np.load(config["peptide_features"])[rows.sequence_index.to_numpy(int)]
    categories = fit_categories(rows, False)
    x = pair_features(peptide, rows, genomes, categories, True)
    lo, hi, _ = measured_bounds(rows)
    torch.set_num_threads(config["cpu_threads"])
    bundle = fit_regressor(
        x, np.zeros(len(rows), dtype=int), lo, hi, config["settings"], config["seed"]
    )
    bundle.update(categories=categories, genome=True)
    torch.save(bundle, output / "model.pt")
    write_json(output / "training_ids.json", rows.observation_id.tolist())
    sequences = pd.read_csv(config["pool_sequences"]).sequence.tolist()
    pool = np.load(config["pool_features"])
    if len(pool) != len(sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("Invalid pool feature alignment")
    with np.load(config["apex_pool"]) as archive:
        if archive["pathogens"].tolist() != list(APEX_PATHOGENS):
            raise ValueError("Unexpected APEX pathogen order")
        lookup = {s: i for i, s in enumerate(archive["sequences"].tolist())}
        apex = np.log2(archive["mic_uM"].mean(1))[[lookup[s] for s in sequences]]
    raw = np.empty((len(sequences), 11))
    mapping_rows: list[dict[str, Any]] = []
    for head, target in enumerate(APEX_PATHOGENS):
        mapping = resolve_genome(target, SPECIES[STRAIN_SPECIES[head]], config)
        reason = fallback_reason(mapping, genomes)
        for start in range(0, len(pool), 4096):
            stop = min(start + 4096, len(pool))
            target_rows = pd.DataFrame(
                dict(
                    species=[mapping.species] * (stop - start),
                    genome_accession=mapping.accession,
                    genome_status=mapping.status,
                )
            )
            features = pair_features(pool[start:stop], target_rows, genomes, categories, True)
            prediction, _ = predict_regressor(bundle, features)
            raw[start:stop, head] = prediction[:, 0]
        mapping_rows.append(
            dict(
                **mapping.model_dump(),
                fallback_reason=reason,
                training_exact_rows=int(
                    (
                        rows.genome_accession.eq(mapping.accession)
                        & rows.genome_status.eq("exact_label")
                    ).sum()
                ),
                assay="not_conditioned",
                uncertainty="not_calibrated",
            )
        )
    frame = genome_prediction_records(
        sequences,
        raw,
        apex,
        [resolve_genome(m["target"], m["species"], config) for m in mapping_rows],
        genomes,
        file_sha256(output / "model.pt"),
        inputs[config["apex_pool"]],
    )
    if not np.isfinite(frame.prediction).all() or frame.duplicated(["sequence", "target_id"]).any():
        raise ValueError("Invalid common-adapter handoff")
    frame.to_csv(output / "predictions.csv.gz", index=False)
    np.savez_compressed(
        output / "raw_pair_predictions.npz",
        sequences=np.array(sequences),
        targets=np.array(APEX_PATHOGENS),
        log2_mic=raw,
    )
    write_json(output / "target_support.json", mapping_rows)
    finish_stage(
        output,
        inputs,
        started,
        sequences=len(sequences),
        supported_targets=sum(not m["fallback_reason"] for m in mapping_rows),
        supported_rows=int(frame.supported.sum()),
        adopted=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["fetch", "prepare", "train", "handoff"])
    parser.add_argument("--config", type=Path, default=Path("configs/mic_genome_pilot.json"))
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    with threadpool_limits(limits=config["cpu_threads"]):
        if args.stage == "fetch":
            fetch(config, args.output)
        elif args.stage == "prepare":
            if args.sources is None:
                parser.error("--sources is required")
            prepare(config, args.sources, args.output, args.config)
        else:
            if args.prepared is None:
                parser.error("--prepared is required")
            if args.stage == "train":
                train(config, args.prepared, args.output)
            else:
                handoff(config, args.prepared, args.output)


if __name__ == "__main__":
    main()
