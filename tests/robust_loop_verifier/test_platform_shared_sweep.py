from robust_loop_verifier.platform_shared_sweep import (
    ScoreSpec,
    choose_balanced_setting,
    evaluate_group_score,
)


def test_evaluate_group_score_uses_one_score_spec_for_all_group_sequences():
    records = [
        {"sequence_key": "group/a", "label": True, "score": 3.0},
        {"sequence_key": "group/a", "label": False, "score": 2.0},
        {"sequence_key": "group/b", "label": False, "score": 2.0},
        {"sequence_key": "group/b", "label": True, "score": 3.0},
    ]
    spec = ScoreSpec(name="unit", scores=[row["score"] for row in records])

    row = evaluate_group_score(
        group_name="group",
        sequence_keys=("group/a", "group/b"),
        records=records,
        spec=spec,
    )

    assert row["group"] == "group"
    assert row["sequence_count"] == 2
    assert row["positive_count"] == 2
    assert row["macro_AP"] == 1.0
    assert row["macro_MR@100P"] == 1.0
    assert row["TP@100P"] == 2


def test_choose_balanced_setting_prefers_ap_near_best_mr():
    rows = [
        {
            "name": "best_mr_low_ap",
            "macro_AP": 0.70,
            "macro_MR@100P": 1.00,
            "TP@100P": 10,
        },
        {
            "name": "balanced",
            "macro_AP": 0.90,
            "macro_MR@100P": 0.995,
            "TP@100P": 10,
        },
        {
            "name": "high_ap_low_mr",
            "macro_AP": 0.95,
            "macro_MR@100P": 0.80,
            "TP@100P": 8,
        },
    ]

    selected = choose_balanced_setting(rows, mr_tolerance=0.01)

    assert selected["name"] == "balanced"
