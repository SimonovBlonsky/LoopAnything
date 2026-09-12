from robust_loop_verifier.hard_gate_platform_sweep import (
    BASE_THRESHOLDS_BY_GROUP,
    evaluate_hard_gate_global_shared_sweep,
    evaluate_hard_gate_platform_shared_sweep,
    score_hard_gate_records,
)


def _record(pair_id, sequence, label, features):
    return {
        "pair_id": pair_id,
        "sequence_key": sequence,
        "query_idx": 10,
        "candidate_idx": 1,
        "label": label,
        "gate_features": features,
    }


def test_score_hard_gate_records_is_negative_maximum_violation():
    thresholds = BASE_THRESHOLDS_BY_GROUP["handheld"]
    weights = [
        thresholds["salad_min_score"],
        thresholds["support_min_baseline"],
        thresholds["sim3_min_scale"],
        1.0 / thresholds["sim3_max_scale"],
        1.0 / thresholds["sim3_alignment_rmse"],
        1.0 / thresholds["sim3_direction_error"],
        1.0 / thresholds["translation_norm"],
        1.0 / thresholds["trajectory_deformation"],
        1.0 / thresholds["pgo_error_per_factor"],
        1.0 / thresholds["odom_strain_chi2"],
        1.0 / thresholds["loop_chi2"],
    ]
    features = tuple(0.5 / weight for weight in weights)
    records = [_record("p0", "group/a", True, features)]

    assert score_hard_gate_records(records, thresholds) == [-0.5]


def test_platform_sweep_uses_one_selected_threshold_set_per_group():
    groups = {"handheld": ("group/a", "group/b")}
    positive_features = (1.0,) * 11
    negative_features = (2.0,) * 11
    records = [
        _record("a-pos", "group/a", True, positive_features),
        _record("a-neg", "group/a", False, negative_features),
        _record("b-pos", "group/b", True, positive_features),
        _record("b-neg", "group/b", False, negative_features),
    ]

    result = evaluate_hard_gate_platform_shared_sweep(
        records,
        platform_groups=groups,
        threshold_multipliers=(1.0,),
        mr_tolerance=0.0,
    )

    selected = result["selections"]["handheld"]["balanced"]
    assert selected["macro_MR@100P"] == 1.0
    assert len(result["selected_sequence_rows"]) == 2
    assert {row["method"] for row in result["selected_sequence_rows"]} == {selected["name"]}


def test_global_sweep_uses_one_selected_threshold_set_for_every_sequence():
    positive_features = (1.0,) * 11
    negative_features = (2.0,) * 11
    records = [
        _record("a-pos", "group/a", True, positive_features),
        _record("a-neg", "group/a", False, negative_features),
        _record("b-pos", "group/b", True, positive_features),
        _record("b-neg", "group/b", False, negative_features),
    ]

    result = evaluate_hard_gate_global_shared_sweep(
        records,
        sequence_keys=("group/a", "group/b"),
        threshold_multipliers=(1.0,),
        mr_tolerance=0.0,
    )

    selected = result["selections"]["all"]["balanced"]
    assert result["sharing_scope"] == "global"
    assert set(result["selections"]) == {"all"}
    assert len(result["selected_sequence_rows"]) == 2
    assert {row["method"] for row in result["selected_sequence_rows"]} == {selected["name"]}
