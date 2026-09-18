from robust_apex_qd.research.provenance import match_activity, modal_mic, training_folds


def test_blank_published_cell_is_not_inactive_or_censored() -> None:
    assert modal_mic("") == (None, None)
    assert modal_mic("64") == (64.0, 0)
    assert modal_mic("16") == (16.0, 1)


def test_assay_match_requires_conditions_and_preserves_ambiguous_references() -> None:
    observation = {
        "sequence": "ACDEFGHIK",
        "target": "Escherichia coli ATCC 11775",
        "raw_value": "16",
        "raw_unit": "µM",
        "medium": "MHB",
        "cfu": "1E5",
        "note": None,
    }
    activity = {
        "id": 17,
        "targetSpecies": {"name": observation["target"]},
        "activityMeasureGroup": {"name": "MIC"},
        "concentration": "16",
        "unit": {"name": "µM"},
        "medium": {"name": "MHB"},
        "cfu": "1E5",
        "note": "",
        "reference": "2",
    }
    metadata = {
        "sequence": "ACDEFGHIK",
        "targetActivities": [activity],
        "articles": [
            {"id": 555, "pubmed": {"pubmedId": "111"}},
            {"id": 666, "pubmed": {"pubmedId": "222"}},
        ],
    }
    result = match_activity(observation, metadata)
    assert result["activity_ids"] == [17]
    assert result["reference_tokens"] == ["2"]
    assert result["assay_publication_verified"] is False
    assert result["publication_candidates"] == ["pubmed:111", "pubmed:222"]
    # Reference tokens are not assumed to index the articles array.
    observation["cfu"] = "5E5"
    assert match_activity(observation, metadata)["activity_ids"] == []


def test_sequence_change_prevents_cross_release_assay_match() -> None:
    assert (
        match_activity({"sequence": "AAAA"}, {"sequence": "WWWW"})["status"] == "sequence_conflict"
    )


def test_inner_folds_never_move_holdout_and_keep_groups_intact() -> None:
    rows = [{"sequence": str(i), "group": str(i // 2), "split": "train"} for i in range(12)]
    rows.append({"sequence": "held", "group": "held", "split": "holdout"})
    folds = training_folds(rows, n_folds=3)
    assert "held" not in folds
    assert set(folds.values()) == {0, 1, 2}
    assert all(folds[str(i)] == folds[str(i + 1)] for i in range(0, 12, 2))
    assert folds == training_folds(list(reversed(rows)), n_folds=3)
