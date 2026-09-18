"""Train resumable public-development MIC screens and retain fold-level provenance."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_research_models import load_esm
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.research.competition_models import (
    SPECIES,
    STRAIN_SPECIES,
    bounded_loss,
    checked_masks,
    fit_ridge_heads,
    predict_ridge_heads,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def grid() -> list[dict[str, Any]]:
    result = []
    for family in ["physchem", "linear8", "linear650"]:
        for alpha in [1.0, 10.0, 100.0]:
            result.append(dict(id=f"{family}-a{alpha:g}", family=family, alpha=alpha))
    for width in [32, 64]:
        for factor in [0.3, 1.0, 3.0]:
            result.append(
                dict(
                    id=f"mlp8-w{width}-r{factor:g}",
                    family="mlp8",
                    width=width,
                    lr=0.003 * factor,
                    epochs=150,
                )
            )
    for epochs in [5, 10]:
        for factor in [0.3, 1.0, 3.0]:
            result.append(
                dict(
                    id=f"finetune8-e{epochs}-r{factor:g}",
                    family="finetune8",
                    lr=0.0001 * factor,
                    epochs=epochs,
                )
            )
    return result


def prepare(config: dict[str, Any], output: Path) -> None:
    data = Path(config["dataset"])
    manifest = json.loads((data / "split_manifest.json").read_text())
    for name, digest in manifest["artifacts_sha256"].items():
        if file_sha256(data / name) != digest:
            raise ValueError(f"Dataset changed: {name}")
    rows = pd.read_json(data / "development.jsonl", lines=True)
    excluded = rows[~rows.species.isin(SPECIES)]
    excluded.groupby(["objective", "species"]).size().to_csv(output / "non_target_species.csv")
    rows = rows[rows.species.isin(SPECIES)].copy().reset_index(drop=True)
    rows["species_index"] = rows.species.map({s: i for i, s in enumerate(SPECIES)})
    rows["strain_index"] = (
        rows.apex_pathogen.map({s: i for i, s in enumerate(APEX_PATHOGENS)}).fillna(-1).astype(int)
    )
    sequences = sorted(set(rows.sequence))
    rows["sequence_index"] = rows.sequence.map({s: i for i, s in enumerate(sequences)})
    rows.to_json(output / "rows.jsonl", orient="records", lines=True)
    write_fasta(
        [FastaRecord(f"d{i}", s) for i, s in enumerate(sequences)], output / "sequences.fasta"
    )
    feature_rows = []
    for s in sequences:
        feature_rows.append(
            {
                **compute_features(s),
                **{f"aac_{a}": s.count(a) / len(s) for a in "ACDEFGHIKLMNPQRSTVWY"},
            }
        )
    frame = pd.DataFrame(feature_rows)
    frame.to_csv(output / "physchem.csv", index=False)
    write_json(
        output / "input_manifest.json",
        dict(
            dataset_sha256=file_sha256(data / "development.jsonl"),
            folds_sha256=file_sha256(data / "fold_assignments.csv"),
            sequences=len(sequences),
            included_rows=len(rows),
            measured_rows=int(rows.objective.eq("measured_mic").sum()),
            excluded_non_target_rows=len(excluded),
            heads=list(SPECIES),
            strains=list(APEX_PATHOGENS),
        ),
    )


def encode(config: dict[str, Any], root: Path, output: Path, large: bool) -> None:
    sequences = read_fasta_sequences(root / "prepare/sequences.fasta")
    model, alphabet = load_esm(Path(config["esm650_checkpoint" if large else "esm8_checkpoint"]))
    model.eval().cuda()
    arrays = []
    for start in range(0, len(sequences), 8):
        batch = sequences[start : start + 8]
        _, _, tokens = alphabet.get_batch_converter()([(str(i), s) for i, s in enumerate(batch)])
        with torch.no_grad():
            rep = model(tokens.cuda(), repr_layers=[model.num_layers])["representations"][
                model.num_layers
            ]
        arrays.extend(rep[i, 1 : len(s) + 1].mean(0).cpu().numpy() for i, s in enumerate(batch))
    np.save(output / "features.npy", np.asarray(arrays, dtype=np.float32))


def load_features(root: Path, family: str) -> np.ndarray:
    if family == "physchem":
        return pd.read_csv(root / "prepare/physchem.csv").to_numpy(dtype=np.float32)
    return np.load(root / ("esm650" if family == "linear650" else "esm8") / "features.npy")


def training_mask(rows: pd.DataFrame, arm: dict[str, Any]) -> np.ndarray:
    measured = rows.objective.eq("measured_mic").to_numpy()
    loss = arm.get("loss", "exact")
    mask = measured & (
        rows.exact_regression.to_numpy()
        if loss == "exact"
        else rows.active16.notna().to_numpy()
        if loss == "classification"
        else (rows.lower_um.notna() | rows.upper_um.notna()).to_numpy()
    )
    if arm.get("chemistry") == "strict":
        mask &= rows.strict_chemistry.to_numpy()
    if arm.get("head") == "strain_only":
        mask &= rows.strain_index.to_numpy() >= 0
    if arm.get("consensus", False):
        mask |= rows.objective.eq("qmap_consensus").to_numpy()
    return mask


def fit(
    config: dict[str, Any],
    rows: pd.DataFrame,
    features: np.ndarray,
    train: np.ndarray,
    arm: dict[str, Any],
    seed: int,
    output: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    chosen = rows.loc[train].copy()
    seqidx = chosen.sequence_index.to_numpy(int)
    x = features[seqidx]
    species = chosen.species_index.to_numpy(int)
    strains = chosen.strain_index.to_numpy(int)
    consensus = chosen.objective.eq("qmap_consensus").to_numpy()
    y = np.log2(np.where(consensus, chosen.consensus_um, chosen.mic_um).astype(float))
    support = np.array([np.any((strains == i) & ~consensus) for i in range(11)])
    species_support = np.array([np.any((species == i) & ~consensus) for i in range(7)])
    if arm["family"] in ["physchem", "linear8", "linear650"]:
        state = fit_ridge_heads(x, species, y, 7, arm["alpha"])
        base = predict_ridge_heads(state, x)[np.arange(len(x)), species]
        offsets = np.array(
            [np.mean((y - base)[strains == i]) if support[i] else 0.0 for i in range(11)]
        )
        state.update(offsets=offsets, strain_support=support, species_support=species_support)
        np.savez(output / "weights.npz", **state)
        return state
    device = "cuda"
    fine = arm["family"] == "finetune8"
    scaler = StandardScaler().fit(x)
    center = float(np.mean(y)) if arm.get("loss") != "classification" else 0.0
    encoder = None
    if fine:
        encoder, alphabet = load_esm(Path(config["esm8_checkpoint"]))
        encoder.cuda().train()
        head = torch.nn.Linear(320, 25).cuda()
        sequences = read_fasta_sequences(Path(config["run_root"]) / "prepare/sequences.fasta")
        _, _, tokens = alphabet.get_batch_converter()(
            [(str(i), s) for i, s in enumerate(sequences)]
        )
        inputs = tokens.cuda()
        params = [*encoder.parameters(), *head.parameters()]
    else:
        inputs = torch.tensor(scaler.transform(features), dtype=torch.float32, device=device)
        head = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], arm["width"]),
            torch.nn.ReLU(),
            torch.nn.Linear(arm["width"], 25),
        ).cuda()
        params = list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=arm["lr"], weight_decay=0.01)
    lower = np.log2(chosen.lower_um.fillna(0).to_numpy(float))
    upper = np.log2(chosen.upper_um.fillna(np.inf).to_numpy(float))
    lower[consensus] = y[consensus]
    upper[consensus] = y[consensus]
    targets = torch.tensor(np.nan_to_num(y) - center, dtype=torch.float32, device=device)
    low = torch.tensor(lower - center, dtype=torch.float32, device=device)
    high = torch.tensor(upper - center, dtype=torch.float32, device=device)
    labels = torch.tensor(
        chosen.active16.fillna(False).to_numpy(float), dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(seed)
    losses = []
    for _epoch in range(arm["epochs"]):
        order = rng.permutation(len(chosen))
        total = 0.0
        for start in range(
            0, len(order), config["finetune_batch_size"] if fine else config["mlp_batch_size"]
        ):
            b = order[
                start : start
                + (config["finetune_batch_size"] if fine else config["mlp_batch_size"])
            ]
            xi = inputs[seqidx[b]]
            if encoder is not None:
                rep = encoder(xi, repr_layers=[6])["representations"][6]
                mask = (xi != 1) & (xi != 0) & (xi != 2)
                xi = (rep * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
            raw = head(xi)
            values = raw[
                torch.arange(len(b), device=device), torch.tensor(species[b], device=device)
            ]
            known = (strains[b] >= 0) & ~consensus[b]
            if arm.get("head") != "species_only":
                values = values + raw[
                    torch.arange(len(b), device=device),
                    torch.tensor(np.maximum(strains[b], 0) + 7, device=device),
                ] * torch.tensor(known, device=device)
            aux = torch.tensor(consensus[b], device=device)
            values = torch.where(
                aux,
                raw[
                    torch.arange(len(b), device=device),
                    torch.tensor(species[b] + 18, device=device),
                ],
                values,
            )
            if arm.get("loss") == "interval":
                loss = bounded_loss(values, low[b], high[b]).mean()
            elif arm.get("loss") == "classification":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(values, labels[b])
            else:
                weights = torch.where(aux, 0.25, 1.0)
                loss = ((values - targets[b]).square() * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(b)
        losses.append(total / len(order))
    state = dict(
        head=head.cpu().state_dict(),
        encoder=encoder.cpu().state_dict() if encoder is not None else None,
        mean=scaler.mean_,
        scale=scaler.scale_,
        center=center,
        strain_support=support,
        species_support=species_support,
        arm=arm,
        loss_curve=losses,
    )
    torch.save(state, output / "weights.pt")
    return state


def predict(
    config: dict[str, Any],
    state: dict[str, Any],
    features: np.ndarray,
    sequences: list[str],
    arm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if arm["family"] in ["physchem", "linear8", "linear650"]:
        species = predict_ridge_heads(state, features)
        strains = species[:, STRAIN_SPECIES] + state["offsets"]
    else:
        encoder = None
        if arm["family"] == "finetune8":
            encoder, alphabet = load_esm(Path(config["esm8_checkpoint"]))
            encoder.load_state_dict(state["encoder"])
            encoder.eval().cuda()
            head = torch.nn.Linear(320, 25)
        else:
            head = torch.nn.Sequential(
                torch.nn.Linear(features.shape[1], arm["width"]),
                torch.nn.ReLU(),
                torch.nn.Linear(arm["width"], 25),
            )
        head.load_state_dict(state["head"])
        head.eval().cuda()
        values = []
        for start in range(0, len(features), 32):
            with torch.no_grad():
                if encoder is not None:
                    batch = sequences[start : start + 32]
                    _, _, tokens = alphabet.get_batch_converter()(
                        [(str(i), s) for i, s in enumerate(batch)]
                    )
                    tokens = tokens.cuda()
                    rep = encoder(tokens, repr_layers=[6])["representations"][6]
                    mask = (tokens != 1) & (tokens != 0) & (tokens != 2)
                    x = (rep * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
                else:
                    x = torch.tensor(
                        (features[start : start + 32] - state["mean"]) / state["scale"],
                        dtype=torch.float32,
                        device="cuda",
                    )
                values.append(head(x).cpu().numpy())
        raw = np.concatenate(values)
        species = raw[:, :7] + state["center"]
        strains = species[:, STRAIN_SPECIES].copy()
        if arm.get("head") != "species_only":
            strains += raw[:, 7:18]
        if arm.get("loss") == "classification":
            # Output is negative activity logit, never a MIC concentration.
            species, strains = -species, -strains
    species[:, ~state["species_support"]] = np.nan
    strains[:, ~state["strain_support"]] = np.nan
    return species, strains


def metric(rows: pd.DataFrame, values: np.ndarray) -> dict[str, Any]:
    measured = rows.objective.eq("measured_mic").to_numpy()
    exact = measured & rows.exact_regression.to_numpy() & np.isfinite(values)
    active = measured & rows.active16.notna().to_numpy() & np.isfinite(values)
    top, ap = [], []
    for species in SPECIES:
        ids = np.flatnonzero(active & rows.species.eq(species).to_numpy())
        if len(ids):
            order = sorted(
                ids, key=lambda i: (values[i], rows.iloc[i].sequence, rows.iloc[i].observation_id)
            )
            top.append(
                float(rows.iloc[order[: max(1, int(np.ceil(0.2 * len(ids))))]].active16.mean())
            )
            if rows.iloc[ids].active16.nunique() == 2:
                ap.append(
                    float(
                        average_precision_score(rows.iloc[ids].active16.astype(int), -values[ids])
                    )
                )
    return dict(
        rows=int(measured.sum()),
        coverage=int((measured & np.isfinite(values)).sum()),
        macro_top20=float(np.mean(top)) if top else None,
        macro_ap=float(np.mean(ap)) if ap else None,
        mae=float(np.mean(np.abs(values[exact] - np.log2(rows.loc[exact, "mic_um"]))))
        if exact.any()
        else None,
    )


def run_arm(config: dict[str, Any], root: Path, arm: dict[str, Any], seed: int) -> dict[str, Any]:
    output = root / "fits" / f"{arm['id']}-s{seed}"
    if (output / "fit_manifest.json").exists():
        audit = json.loads((output / "fit_manifest.json").read_text())
        for name, digest in audit["artifacts_sha256"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Completed fit artifact changed: {output / name}")
        return audit["summary"]
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_json(root / "prepare/rows.jsonl", lines=True)
    if arm.get("split") == "exact":
        rows["homology_group"] = rows.sequence
        rows["homology_fold"] = rows.exact_fold
    features = load_features(root, arm["family"])
    sequences = read_fasta_sequences(root / "prepare/sequences.fasta")
    allowed = training_mask(rows, arm)
    prediction = np.full(len(rows), np.nan)
    species_prediction = prediction.copy()
    audits = []
    began = time.monotonic()
    for fold in sorted(rows.homology_fold.unique()):
        path = output / f"fold{fold}"
        train, valid = checked_masks(rows, int(fold))
        train &= allowed
        if (path / "audit.json").exists():
            audit = json.loads((path / "audit.json").read_text())
            for name, digest in audit["artifacts_sha256"].items():
                if file_sha256(path / name) != digest:
                    raise ValueError("Fold artifacts changed")
            saved = np.load(path / "predictions.npz")
            prediction[valid], species_prediction[valid] = saved["prediction"], saved["species"]
            audits.append(audit)
            continue
        path.mkdir(exist_ok=False)
        start = time.monotonic()
        state = fit(config, rows, features, train, arm, seed, path)
        unique = sorted(set(rows.loc[valid, "sequence_index"]))
        p, s = predict(config, state, features[unique], [sequences[i] for i in unique], arm)
        # Validate the serialized artifact, rather than relying on in-memory fit success.
        loaded = (
            dict(np.load(path / "weights.npz"))
            if (path / "weights.npz").exists()
            else torch.load(path / "weights.pt", weights_only=False)
        )
        p2, s2 = predict(
            config, loaded, features[unique[:32]], [sequences[i] for i in unique[:32]], arm
        )
        np.testing.assert_allclose(p[:32], p2, atol=1e-5, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(s[:32], s2, atol=1e-5, rtol=1e-5, equal_nan=True)
        mapping = {i: j for j, i in enumerate(unique)}
        vr = rows.loc[valid]
        ix = np.array([mapping[i] for i in vr.sequence_index])
        sp = p[ix, vr.species_index.to_numpy(int)]
        values = sp.copy()
        known = vr.strain_index.to_numpy(int) >= 0
        values[known] = s[ix[known], vr.loc[known, "strain_index"].to_numpy(int)]
        prediction[valid], species_prediction[valid] = values, sp
        np.savez(path / "predictions.npz", prediction=values, species=sp)
        audit = dict(
            fold=int(fold),
            seed=seed,
            arm=arm,
            train_ids=rows.loc[train, "observation_id"].tolist(),
            validation_ids=vr.observation_id.tolist(),
            train_groups=sorted(set(rows.loc[train, "homology_group"])),
            validation_groups=sorted(set(vr.homology_group)),
            train_rows=int(train.sum()),
            validation_rows=int(valid.sum()),
            strain_support=state["strain_support"].tolist(),
            species_support=state["species_support"].tolist(),
            seconds=time.monotonic() - start,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            reload_equal=True,
            artifacts_sha256={p.name: file_sha256(p) for p in path.iterdir() if p.is_file()},
        )
        write_json(path / "audit.json", audit)
        audits.append(audit)
        print(f"{arm['id']} seed{seed} fold{fold} {audit['seconds']:.1f}s", flush=True)
    result = rows[
        [
            "observation_id",
            "sequence",
            "species",
            "apex_pathogen",
            "homology_group",
            "homology_fold",
        ]
    ].copy()
    result["prediction"], result["species_prediction"] = prediction, species_prediction
    result.to_csv(output / "oof.csv", index=False)
    summary = dict(
        id=arm["id"],
        family=arm["family"],
        seed=seed,
        seconds=time.monotonic() - began,
        unit="negative_activity_logit" if arm.get("loss") == "classification" else "log2_uM",
        **metric(rows, prediction),
    )
    if arm.get("loss") == "classification":
        summary["mae"] = None
    write_json(
        output / "fit_manifest.json",
        dict(
            summary=summary,
            arm=arm,
            folds=audits,
            input_sha256=file_sha256(root / "prepare/rows.jsonl"),
            artifacts_sha256={
                str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()
            },
        ),
    )
    return summary


def train(config: dict[str, Any], root: Path) -> None:
    summaries = []
    arms = grid()
    write_json(root / "registered_grid.json", arms)
    for arm in arms:
        summaries.append(run_arm(config, root, arm, 42))
        pd.DataFrame(summaries).to_csv(root / "model_comparison.csv", index=False)
    winners = []
    frame = pd.DataFrame(summaries)
    for _family, group in frame.groupby("family"):
        ids = (
            group.sort_values(["macro_top20", "mae", "id"], ascending=[False, True, True])
            .head(2)
            .id
        )
        for name in ids:
            arm = next(a for a in arms if a["id"] == name)
            winners.append(arm)
            for seed in [43, 44]:
                summaries.append(run_arm(config, root, arm, seed))
    base = next(a for a in arms if a["id"] == "mlp8-w32-r1")
    factors = [
        ("strict", {"chemistry": "strict"}),
        ("strain-only", {"head": "strain_only"}),
        ("species-only", {"head": "species_only"}),
        ("interval", {"loss": "interval"}),
        ("classification", {"loss": "classification"}),
        ("consensus", {"consensus": True}),
        ("exact-fold", {"split": "exact"}),
    ]
    for name, settings in factors:
        summaries.append(run_arm(config, root, {**base, **settings, "id": f"ablation-{name}"}, 42))
    pd.DataFrame(summaries).to_csv(root / "model_comparison.csv", index=False)
    write_json(root / "screen_winners.json", winners)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_models.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "esm8", "esm650", "train"], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["run_root"] = str(args.output)
    code = {
        str(p): file_sha256(p)
        for p in [Path(__file__), Path("src/robust_apex_qd/research/competition_models.py")]
    }
    protocol = dict(config=config, code_sha256=code)
    if (args.output / "protocol.json").exists():
        if json.loads((args.output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume config or source changed")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output must be empty before registration")
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    with threadpool_limits(limits=4):
        if args.stage == "train":
            train(config, args.output)
        else:
            stage = args.output / args.stage
            stage.mkdir(exist_ok=False)
            start = time.monotonic()
            if args.stage == "prepare":
                prepare(config, stage)
            else:
                encode(config, args.output, stage, args.stage == "esm650")
            write_json(
                stage / "manifest.json",
                dict(
                    seconds=time.monotonic() - start,
                    artifacts_sha256={
                        p.name: file_sha256(p) for p in stage.iterdir() if p.is_file()
                    },
                ),
            )


if __name__ == "__main__":
    main()
