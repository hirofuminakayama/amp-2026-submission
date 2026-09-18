import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.research.genome_pairs import (
    GenomeMapping,
    dna_fourmers,
    fallback_reason,
    fit_categories,
    genome_prediction_records,
    pair_features,
    pair_masks,
    resolve_genome,
)


def test_dna_counts_respect_contigs_ambiguity_and_reverse_complement() -> None:
    a = dna_fourmers(["AAAACGTNCCCC"])
    b = dna_fourmers(["GGGGNACGTTTT"])
    np.testing.assert_equal(a, b)
    assert a.shape == (256,) and a.sum() == pytest.approx(1)
    np.testing.assert_equal(dna_fourmers(["AAAA", "TTTT"]), dna_fourmers(["AAAA"]))
    with pytest.raises(ValueError):
        dna_fourmers(["AAA", "TTT", "NNNN"])


def test_mapping_does_not_promote_missing_or_species_reference_to_exact() -> None:
    missing = GenomeMapping(target="species isolate", species="species", status="unknown")
    assert missing.accession is None
    with pytest.raises(ValueError):
        GenomeMapping(target="species isolate", species="species", status="exact_label")
    with pytest.raises(ValueError):
        GenomeMapping(
            target="species isolate",
            species="species",
            status="unknown",
            accession="GCF_000000001.1",
        )


def test_unseen_masks_quarantine_auxiliary_rows_and_uncertain_strains() -> None:
    rows = pd.DataFrame(
        dict(
            peptide_fold=[0, 1, 0, 1, 1],
            genome_fold=[0, 1, 1, 0, -1],
            primary=[True, True, False, False, True],
            genome_status=["exact_label"] * 4 + ["species_reference"],
        )
    )
    train, valid = pair_masks(rows, "both", 0, 0)
    assert train.tolist() == [False, True, False, False, False]
    assert valid.tolist() == [True, False, False, False, False]
    train, valid = pair_masks(rows, "peptide", 0, None)
    assert train.tolist() == [False, True, False, True, True]
    assert valid.tolist() == [True, False, False, False, False]
    train, valid = pair_masks(rows, "strain", None, 0)
    assert train.tolist() == [False, True, True, False, False]
    assert valid.tolist() == [True, False, False, False, False]


def test_features_fit_categories_on_training_and_expose_missingness() -> None:
    rows = pd.DataFrame(
        dict(
            species=["known", "novel"],
            medium=["MHB", "future"],
            cfu=[None, "1e6"],
            genome_accession=["GCF_000000001.1", None],
            genome_status=["exact_label", "unknown"],
        )
    )
    categories = fit_categories(rows.iloc[:1], True)
    assert categories == {"species": ["known"], "medium": ["MHB"], "cfu": []}
    genomes = {"GCF_000000001.1": dna_fourmers(["AAAA"])}
    x = pair_features(np.ones((2, 3)), rows, genomes, categories, True)
    assert np.isfinite(x).all()
    assert x[1, -2] == 1 and x[0, -2] == 0
    assert x[1, 3] == 0 and x[1, 4] == 1
    masked = pair_features(np.ones((2, 3)), rows, genomes, categories, True, mask_assay=True)
    assert masked[0, 5] == 0 and masked[0, 6] == 1
    assert rows.medium.tolist() == ["MHB", "future"]
    np.testing.assert_equal(masked[:, 8:], x[:, 8:])


def test_exact_alias_registry_and_fallback_are_conservative() -> None:
    config = dict(
        assemblies=[dict(accession="GCF_000000001.1", labels=["S stock"])],
        species_references={"S": "GCF_000000001.1"},
    )
    exact = resolve_genome("S stock", "S", config)
    variant = resolve_genome("S stock mutant", "S", config)
    unknown = resolve_genome("T stock", "T", config)
    genomes = {"GCF_000000001.1": dna_fourmers(["AAAA"])}
    assert fallback_reason(exact, genomes) == ""
    assert fallback_reason(variant, genomes) == "species_reference_not_exact_strain"
    assert fallback_reason(unknown, genomes) == "genome_missing"
    assert fallback_reason(exact, {}) == "genome_missing"
    assert not exact.experimental_isolate_verified


def test_common_adapter_keeps_unresolved_strain_on_apex() -> None:
    from robust_apex_qd.apex.ensemble import APEX_PATHOGENS

    mappings = [GenomeMapping(target=t, species="S", status="unknown") for t in APEX_PATHOGENS]
    mappings[0] = GenomeMapping(
        target=APEX_PATHOGENS[0], species="S", status="exact_label", accession="GCF_000000001.1"
    )
    mappings[1] = GenomeMapping(
        target=APEX_PATHOGENS[1],
        species="S",
        status="species_reference",
        accession="GCF_000000001.1",
    )
    frame = genome_prediction_records(
        ["AAAA"],
        np.full((1, 11), 2.0),
        np.full((1, 11), 5.0),
        mappings,
        {"GCF_000000001.1": dna_fourmers(["AAAA"])},
        "pair",
        "apex",
    )
    assert frame.prediction.tolist() == [2.0] + [5.0] * 10
    assert frame.supported.tolist() == [True] + [False] * 10
    assert frame.model_sha256.tolist() == ["pair"] + ["apex"] * 10
    assert not frame.experimental_isolate_verified.any()
    assert frame.assay.eq("not_conditioned").all()
    assert frame.iloc[1].fallback_reason == "species_reference_not_exact_strain"
    assert frame.iloc[2].fallback_reason == "genome_missing"


def test_registered_abbreviated_apex_labels_have_useful_exact_support() -> None:
    from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
    from robust_apex_qd.research.competition_models import SPECIES, STRAIN_SPECIES

    config = json.loads(Path("configs/mic_genome_pilot.json").read_text())
    supported = [
        resolve_genome(t, SPECIES[STRAIN_SPECIES[i]], config).status == "exact_label"
        for i, t in enumerate(APEX_PATHOGENS)
    ]
    assert supported == [True, True, False, False, True, False, True, True, False, False, False]
