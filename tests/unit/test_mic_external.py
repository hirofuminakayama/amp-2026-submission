import json
from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.research.mic_external import (
    ExternalObservation,
    build_external_split,
    load_training_rows,
    validate_training_membership,
)


def row(
    identifier: str, sequence: str, paper: str, exposure: str = "certified_unused"
) -> ExternalObservation:
    return ExternalObservation.model_validate(
        dict(
            observation_id=identifier,
            sequence=sequence,
            paper_ids=[paper],
            species="fixture",
            exposure=exposure,
            exposure_evidence="fixture exhaustive use audit",
            primary_eligible=True,
        )
    )


def split(
    rows: list[ExternalObservation],
    identity: np.ndarray | None = None,
    links: list[tuple[str, str]] | None = None,
) -> list[dict]:
    sequences = sorted({r.sequence for r in rows})
    return build_external_split(
        rows, sequences, np.eye(len(sequences)) if identity is None else identity, links or []
    )


def test_used_paper_and_transitive_homology_stay_in_development() -> None:
    rows = [
        row("old", "AAAA", "p1", "used"),
        row("new", "CCCC", "p1"),
        row("bridge", "DDDD", "p2"),
        row("far", "EEEE", "p3"),
    ]
    m = np.eye(4)
    m[1, 2] = m[2, 1] = 0.7
    result = split(rows, m)
    assert {r["partition"] for r in result[:3]} == {"development"}
    assert result[3]["partition"] == "final_evaluation"


def test_unknown_history_never_becomes_unused_and_odd_goes_to_evaluation() -> None:
    rows = [row("u", "AAAA", "p0", "unknown")]
    rows += [row(str(i), s, "p" + str(i)) for i, s in enumerate(["CCCC", "DDDD", "EEEE"], 1)]
    result = split(rows)
    assert result[0]["partition"] == "diagnostic"
    assert sum(r["partition"] == "final_evaluation" for r in result) == 2
    assert sum(r["partition"] == "new_training" for r in result) == 1
    assert split(list(reversed(rows))) == list(reversed(result))


def test_exact_threshold_allowed_above_threshold_and_duplicates_union() -> None:
    rows = [row("a", "AAAA", "p1", "used"), row("b", "CCCC", "p2"), row("c", "DDDD", "p3")]
    m = np.eye(3)
    m[0, 1] = m[1, 0] = 0.6
    result = split(rows, m, [("b", "c")])
    assert result[1]["partition"] == result[2]["partition"] == "final_evaluation"
    m[0, 1] = m[1, 0] = 0.6001
    assert {r["partition"] for r in split(rows, m, [("b", "c")])} == {"development"}


def test_invalid_matrix_or_unresolved_duplicate_rejected() -> None:
    rows = [row("a", "AAAA", "p1"), row("b", "CCCC", "p2")]
    for m in [np.eye(1), np.array([[1, np.nan], [np.nan, 1]]), np.array([[1, 0.7], [0, 1]])]:
        with pytest.raises(ValueError):
            split(rows, m)
    with pytest.raises(ValueError, match="duplicate"):
        split(rows, links=[("a", "missing")])
    with pytest.raises(ValueError):
        split([rows[0], rows[0]])


def test_unknown_bridge_quarantines_entire_component() -> None:
    rows = [row("a", "AAAA", "p1"), row("b", "CCCC", "p1", "unknown")]
    assert {r["partition"] for r in split(rows)} == {"diagnostic"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_id", "held"),
        ("sequence", "CCCC"),
        ("study", "held-paper"),
        ("component_id", "held-component"),
    ],
)
def test_training_guard_rejects_each_held_out_identity(field: str, value: str) -> None:
    contract = dict(
        allowed_observation_ids=["train"],
        forbidden_observation_ids=["held"],
        forbidden_sequences=["CCCC"],
        forbidden_paper_ids=["held-paper"],
        forbidden_component_ids=["held-component"],
    )
    r = dict(
        observation_id="train", sequence="AAAA", study="train-paper", component_id="train-component"
    )
    r[field] = value
    with pytest.raises(ValueError):
        validate_training_membership([r], contract)


def test_unregistered_auxiliary_and_external_label_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_training_membership(
            [dict(observation_id="aux", sequence="AAAA")],
            dict(
                allowed_observation_ids=[],
                forbidden_observation_ids=[],
                forbidden_sequences=[],
                forbidden_paper_ids=[],
                forbidden_component_ids=[],
            ),
        )
    (tmp_path / "rows.jsonl").write_text(
        json.dumps(dict(observation_id="held", external_partition="final_evaluation")) + "\n"
    )
    with pytest.raises(ValueError, match="evaluation"):
        load_training_rows(tmp_path)


def test_unmodified_legacy_training_input_remains_usable(tmp_path: Path) -> None:
    (tmp_path / "rows.jsonl").write_text('{"observation_id":"old"}\n')
    assert list(load_training_rows(tmp_path).observation_id) == ["old"]


def test_labels_cannot_enter_component_builder() -> None:
    record = row("a", "AAAA", "p1").model_dump()
    record["mic_um"] = 1.0
    with pytest.raises(ValueError):
        ExternalObservation.model_validate(record)


def test_training_file_hash_and_fresh_fold_requirement(tmp_path: Path) -> None:
    from robust_apex_qd.features.embeddings import file_sha256

    path = tmp_path / "rows.jsonl"
    path.write_text('{"observation_id":"train","external_partition":"development"}\n')
    contract = dict(
        allowed_observation_ids=["train"],
        forbidden_observation_ids=[],
        forbidden_sequences=[],
        forbidden_paper_ids=[],
        forbidden_component_ids=[],
        rows_sha256=file_sha256(path),
        folds_ready=False,
    )
    (tmp_path / "training_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="new development folds"):
        load_training_rows(tmp_path)
    path.write_text('{"observation_id":"aux","external_partition":"development"}\n')
    with pytest.raises(ValueError, match="changed"):
        load_training_rows(tmp_path)


def test_float32_boundary_is_same_as_native_development_storage() -> None:
    rows = [row("old", "AAAA", "p1", "used"), row("new", "CCCC", "p2")]
    m = np.array([[1, 0.6], [0.6, 1]], dtype=np.float32)
    assert split(rows, m)[1]["partition"] == "final_evaluation"
    m[0, 1] = m[1, 0] = np.nextafter(np.float32(0.6), np.float32(1))
    assert split(rows, m)[1]["partition"] == "development"


def test_nonempty_external_export_keeps_labels_out_of_training_and_inference(
    tmp_path: Path,
) -> None:
    import runpy

    from robust_apex_qd.features.embeddings import file_sha256

    freeze = runpy.run_path("scripts/prepare_mic_external.py")["freeze"]
    inventory = tmp_path / "inventory"
    alignment = tmp_path / "alignment"
    curation = tmp_path / "curation"
    output = tmp_path / "output"
    for p in [inventory, alignment, curation, output]:
        p.mkdir()

    def write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value))

    def lines(path: Path, values: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in values))

    def seal(path: Path, extra: dict | None = None) -> None:
        write(
            path / "manifest.json",
            dict(
                inputs_sha256={},
                artifacts_sha256={p.name: file_sha256(p) for p in path.iterdir()},
                **(extra or {}),
            ),
        )

    sequences = ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]
    records = [row("old", sequences[0], "old-paper", "used")]
    records += [row(str(i), s, "p" + str(i)) for i, s in enumerate(sequences[1:], 1)]
    lines(inventory / "observations.jsonl", [r.model_dump() for r in records])
    write(inventory / "sequences.json", sequences)
    write(inventory / "duplicate_edges.json", [])
    write(inventory / "exposure_summary.json", dict(unknowns=["pretraining unknown"]))
    seal(inventory)
    write(alignment / "sequences.json", sequences)
    np.save(alignment / "identity.npy", np.eye(4, dtype=np.float32))
    seal(
        alignment,
        dict(
            engine="parasail.nw_stats_striped_sat",
            matrix="BLOSUM45",
            gap_open=5,
            gap_extend=1,
            orientation="lexicographic",
            denominator="max(stats.length, len(left), len(right))",
        ),
    )
    m = json.loads((alignment / "manifest.json").read_text())
    m["inputs_sha256"] = {
        str(inventory / "sequences.json"): file_sha256(inventory / "sequences.json")
    }
    write(alignment / "manifest.json", m)
    originals = []
    for r in records[1:]:
        originals.append(
            dict(
                observation_id=r.observation_id,
                paper_id=r.paper_ids[0],
                sequence=r.sequence,
                peptide_name="fixture",
                target="Escherichia coli fixture",
                species="Escherichia coli",
                raw_value="123.456",
                raw_unit="uM",
                cell_text="123.456",
                chemical_form="linear_free_L",
                nterminal="free",
                cterminal="free",
                bonds="none",
                stereochemistry="all L",
                source_file="fixture",
                source_sha256="a" * 64,
                table_id="T1",
                row=0,
                column=0,
                sequence_evidence="fixture",
                chemistry_evidence="fixture",
                assay_evidence="fixture",
                review_evidence="fixture",
                footnotes="fixture",
            )
        )
    lines(curation / "paper_observations.jsonl", originals)
    lines(curation / "duplicate_links.jsonl", [])
    lines(curation / "correction_ledger.jsonl", [])
    seal(curation)
    legacy = tmp_path / "legacy.jsonl"
    lines(legacy, [dict(observation_id="old", sequence=sequences[0], mic_um=1)])
    model = tmp_path / "model"
    model.mkdir()
    write(model / "manifest.json", dict(train_ids=["old"]))
    freeze(
        dict(
            curation=str(curation),
            legacy_training_rows=str(legacy),
            comparators=[dict(name="fixture", files=[str(model / "manifest.json")])],
            comparison_settings={},
            metrics={},
        ),
        inventory,
        alignment,
        output,
    )
    s = json.loads((output / "split_manifest.json").read_text())
    assert len(s["primary_evaluation_ids"]) == 2
    assert len(s["new_training_ids"]) == 1
    target_rows = [
        json.loads(line) for line in (output / "inference/targets.jsonl").read_text().splitlines()
    ]
    assert len(target_rows) == 2
    assert all(
        set(r) == {"observation_id", "sequence", "target", "species", "component_id"}
        for r in target_rows
    )
    training = [
        json.loads(line) for line in (output / "training/rows.jsonl").read_text().splitlines()
    ]
    assert {r["observation_id"] for r in training}.isdisjoint(s["primary_evaluation_ids"])
    assert len((output / "scoring/labels.jsonl").read_text().splitlines()) == 2
    assert (
        json.loads((output / "evaluation_protocol.json").read_text())["scoring_status"]
        == "not_started"
    )
