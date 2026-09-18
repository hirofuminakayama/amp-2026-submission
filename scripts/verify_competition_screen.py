"""Recheck frozen predictor/generator screens without changing their saved artifacts."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    old = args.prior
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    verified = {}

    def hashes(base: Path, items: dict[str, str]) -> None:
        for name, digest in items.items():
            path = base / name
            assert file_sha256(path) == digest, (str(path), "changed")
            verified[str(path)] = digest

    hashes(Path("."), json.loads((old / "phase0/baseline_manifest.json").read_text())["inputs"])
    protocol = json.loads((root / "phase3-r3/protocol.json").read_text())
    hashes(Path("."), protocol["code_sha256"])
    data = Path(protocol["config"]["dataset"])
    hashes(data, json.loads((data / "split_manifest.json").read_text())["artifacts_sha256"])
    for name in ["esm8", "esm650"]:
        p = root / "phase3-r3" / name
        hashes(p, json.loads((p / "manifest.json").read_text())["artifacts_sha256"])
    rows = pd.read_json(root / "phase3-r3/prepare/rows.jsonl", lines=True).set_index(
        "observation_id"
    )
    assert len(rows) == 18328 and rows.index.is_unique
    fits = list((root / "phase3-r3/fits").glob("*/fit_manifest.json"))
    assert len(fits) == 48, len(fits)
    folds = []
    for path in fits:
        fit = json.loads(path.read_text())
        hashes(path.parent, fit["artifacts_sha256"])
        seen = []
        for fold in fit["folds"]:
            train = rows.loc[fold["train_ids"]]
            valid = rows.loc[fold["validation_ids"]]
            assert not set(train.index) & set(valid.index)
            assert not set(train.sequence) & set(valid.sequence)
            if fit["arm"].get("split") != "exact":
                assert not set(train.homology_group) & set(valid.homology_group)
            assert fold["reload_equal"]
            seen += fold["validation_ids"]
            folds.append(
                dict(
                    run=path.parent.name,
                    fold=fold["fold"],
                    seconds=fold["seconds"],
                    rows=len(train),
                    validation=len(valid),
                    peak_cuda_bytes=fold["peak_cuda_bytes"],
                )
            )
        assert len(seen) == len(rows) and set(seen) == set(rows.index)
    assert len(folds) == 240
    pd.DataFrame(folds).to_csv(out / "fold_verification.csv", index=False)
    for name in [
        "phase3-auxiliary",
        "phase3-selection-audit-r2",
        "phase4-component-diagnostics",
        "phase4-deepamp-paired-report",
        "phase4-hydramp-paired-report",
    ]:
        p = root / name
        m = json.loads((p / "manifest.json").read_text())
        hashes(p, m["artifacts_sha256"])
        hashes(Path("."), m["input_sha256"])
    aux = pd.read_csv(root / "phase3-auxiliary/consensus_oof.csv.gz")
    assert len(aux) == 11168 and aux.observation_id.is_unique
    selected = json.loads((root / "phase3-refits/selected_models.json").read_text())
    assert len(selected) == 8
    pool = pd.read_csv(root / "phase3-refits/pool_sequences.csv")
    assert len(pool) == 58195
    refit_records = []
    for arm in selected:
        p = root / "phase3-refits" / arm["artifact_key"]
        m = json.loads((p / "manifest.json").read_text())
        hashes(p, m["artifacts_sha256"])
        assert m["reload_equal"]
        pred = np.load(p / "candidate_predictions.npz")
        assert pred["species"].shape == (len(pool), 7) and pred["strain"].shape == (len(pool), 11)
        support = np.asarray(m["strain_support"], bool)
        assert (
            np.isfinite(pred["strain"][:, support]).all()
            and np.isnan(pred["strain"][:, ~support]).all()
        )
        assert (
            len(set(rows.loc[m["train_ids"]].sequence) & set(pool.sequence))
            == m["train_pool_exact_overlap"]
        )
        refit_records.append(
            dict(
                model=arm["artifact_key"],
                unit=m["unit"],
                support=int(support.sum()),
                overlap=m["train_pool_exact_overlap"],
                seconds=m["seconds"],
            )
        )
    pd.DataFrame(refit_records).to_csv(out / "refit_verification.csv", index=False)
    report = root / "phase3-report"
    hashes(report, json.loads((report / "fit_manifest.json").read_text())["artifacts_sha256"])
    module_spec = importlib.util.spec_from_file_location("official", "scripts/verify_submission.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    full = module._verify_sequences(old / "phase2/prepare/libraries/L2.fasta")
    module._verify_no_overlap(full, set(read_fasta_sequences(Path("data/antibacterial.fasta"))))
    tops = []
    for p in report.glob("*-top.csv"):
        s = pd.read_csv(p).sequence.tolist()
        f = out / (p.stem + ".fasta")
        write_fasta([FastaRecord(f"rank{i + 1}", x) for i, x in enumerate(s)], f)
        module._verify_top(f, full, 100)
        tops.append(str(p))
    assert tops
    source_hashes = {
        file_sha256(p)
        for p in list((root / "executed_sources").glob("*.py"))
        + list(args.source_archive.glob("*.py"))
    }
    for p in (root / "phase4").glob("*/run_manifest.json"):
        m = json.loads(p.read_text())
        if "artifacts_sha256" in m:
            hashes(p.parent, m["artifacts_sha256"])
        else:
            retained = json.loads((root / "hydramp_preserved_output_hashes.json").read_text())[
                "runs"
            ][str(p.parent)]
            hashes(p.parent, retained)
            if "smoke" not in p.parent.name:
                raw = json.loads((p.parent / "raw.json").read_text())
                seq = [
                    v["sequence"]
                    for item in raw.values()
                    for v in (item.get("generated_sequences") or [])
                ]
                seq = [
                    v for v in seq if 10 <= len(v) <= 25 and set(v) <= set("ACDEFGHIKLMNPQRSTVWY")
                ]
                first = pd.read_csv(root / "phase4-evaluation-preflight-r2/collect/samples.csv.gz")
                assert first[first.run.eq(p.parent.name)].sequence.tolist() == seq
        if m.get("source_sha256"):
            assert m["source_sha256"] in source_hashes or m["source_sha256"] == file_sha256(
                Path("scripts/run_competition_hydramp.py")
            )
        if m.get("input_path"):
            assert file_sha256(Path(m["input_path"])) == m["input_sha256"]
        if isinstance(m.get("input_sha256"), dict):
            for name, digest in m["input_sha256"].items():
                q = Path(name)
                assert file_sha256(q) == digest or (
                    q.suffix == ".py" and digest in source_hashes
                ), (name, "input identity differs")
        if p.parent.name.startswith("evodiff-") and "-smoke" not in p.parent.name:
            family = p.parent.name.rsplit("-s", 1)[0]
            checkpoint = {"evodiff-sft": "sft.pt", "evodiff-rl": "rl5.pt"}.get(family)
            expected = (
                {}
                if checkpoint is None
                else {
                    str(root / "phase4/evodiff-optimization" / checkpoint): file_sha256(
                        root / "phase4/evodiff-optimization" / checkpoint
                    )
                }
            )
            actual = {k: v for k, v in m["input_sha256"].items() if Path(k).suffix != ".py"}
            assert actual == expected, (p.parent.name, "checkpoint differs from registered arm")

    for name in [
        "evodiff-optimization",
        "structure-r2",
        "ampgen-downstream-generated",
        "membrane-diagnostic-r2",
    ]:
        p = root / "phase4" / name
        m = json.loads((p / "manifest.json").read_text())
        hashes(p, m["artifacts_sha256"])
    assert len(list((root / "phase4/evodiff-optimization").glob("rl*.pt"))) == 5
    assert len(list((root / "phase4/structure-r2").glob("*.pdb"))) == 20
    membrane = json.loads((root / "phase4/membrane-diagnostic-r2/manifest.json").read_text())
    assert (
        membrane["status"] == "partial_with_concrete_blocker"
        and len(membrane["results"]) == 1
        and membrane["attempted_candidates"] == 2
    )
    for attempt in membrane["attempts"]:
        hashes(Path("."), attempt["input_and_output_sha256"])
    assert json.loads((root / "phase3-r3/failed_fold_reproduction.json").read_text())[
        "weights_exact"
    ]
    assert json.loads((root / "phase4/ampgen-downstream-generated/parity.json").read_text())[
        "passed"
    ]
    evalroot = root / "phase4-evaluation"
    for stage in ["collect", "score", "safety", "embedding", "report"]:
        p = evalroot / stage
        m = json.loads((p / "stage_manifest.json").read_text())
        hashes(p, m["artifacts_sha256"])
        assert m["source_sha256"] in source_hashes or m["source_sha256"] == file_sha256(
            Path("scripts/evaluate_competition_generators.py")
        )
    runs = pd.read_csv(evalroot / "collect/runs.csv")
    comparison = pd.read_csv(evalroot / "report/generator_comparison.csv")
    assert len(comparison) == 2 * len(runs)
    for family in [
        "designer",
        "prompt",
        "evodiff",
        "evodiff-cuda",
        "evodiff-sft",
        "evodiff-rl",
        "ampgen",
        "deepamp",
        "deepamp-common",
        "hydramp-conditional",
        "hydramp-conditional-common",
    ]:
        assert all(f"{family}-s{s}" in set(runs.run) for s in [42, 43]), family
    assert (comparison.hc50_coverage == comparison.retained).all()
    for p in (root / "phase4/assets").glob("*/asset_manifest.json"):
        m = json.loads(p.read_text())
        hashes(Path("."), m["files_sha256"])
    assert (
        hashlib.md5(
            (root / "assets/torch/hub/checkpoints/msa-oaar-maxsub.tar").read_bytes()
        ).hexdigest()
        == "2bfac1cba5be60a09856529b27c73dd5"
    )
    summary = dict(
        passed=True,
        baseline_files=247,
        fit_runs=len(fits),
        folds=len(folds),
        refits=len(selected),
        tops=len(tops),
        generator_runs=len(runs),
        generator_comparison_rows=len(comparison),
        hashes_checked=len(verified),
        official_validator=file_sha256(Path("scripts/verify_submission.py")),
        checked_sha256=verified,
    )
    (out / "completion_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({k: v for k, v in summary.items() if k != "checked_sha256"})


if __name__ == "__main__":
    main()
