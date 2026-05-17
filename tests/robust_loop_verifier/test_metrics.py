import math
import sys

import pytest

from robust_loop_verifier.metrics import (
    assign_failure_worst_scores,
    average_precision,
    max_recall_at_100_precision,
)


def test_average_precision_larger_score_is_better():
    labels = [True, False, True]
    scores = [0.9, 0.8, 0.1]

    assert math.isclose(average_precision(labels, scores), (1.0 + 2.0 / 3.0) / 2.0)


def test_max_recall_at_100_precision_stops_before_false_positive():
    labels = [True, True, False, True]
    scores = [0.9, 0.8, 0.7, 0.1]

    assert math.isclose(max_recall_at_100_precision(labels, scores), 2.0 / 3.0)


def test_failure_scores_rank_after_normal_scores():
    scores = [1.0, None, 0.0, None]

    fixed = assign_failure_worst_scores(scores)

    assert fixed[1] < fixed[2]
    assert fixed[3] < fixed[2]
    assert fixed[1] != fixed[3]


def test_average_precision_no_positives_returns_zero():
    labels = [False, False]
    scores = [0.9, 0.1]

    assert average_precision(labels, scores) == 0.0


def test_max_recall_at_100_precision_no_positives_returns_zero():
    labels = [False, False]
    scores = [0.9, 0.1]

    assert max_recall_at_100_precision(labels, scores) == 0.0


def test_average_precision_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        average_precision([True], [0.9, 0.1])


def test_max_recall_at_100_precision_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        max_recall_at_100_precision([True], [0.9, 0.1])


def test_no_normal_scores_still_assigns_finite_distinct_failure_scores():
    fixed = assign_failure_worst_scores([None, None, None])

    assert all(math.isfinite(score) for score in fixed)
    assert len(set(fixed)) == 3


def test_failure_assignment_rejects_impossible_score_below_lowest_finite_float():
    with pytest.raises(ValueError, match="cannot assign finite failure score below minimum"):
        assign_failure_worst_scores([-sys.float_info.max, None])


def test_equal_scores_preserve_input_order_for_average_precision():
    labels = [False, True, True]
    scores = [0.5, 0.5, 0.1]

    assert math.isclose(average_precision(labels, scores), (1.0 / 2.0 + 2.0 / 3.0) / 2.0)


def test_tied_positive_and_negative_do_not_temporarily_count_as_100_precision():
    labels = [True, False]
    scores = [0.5, 0.5]

    assert max_recall_at_100_precision(labels, scores) == 0.0


def test_tied_positive_group_counts_together_for_max_recall_at_100_precision():
    labels = [True, True, False]
    scores = [0.5, 0.5, 0.1]

    assert max_recall_at_100_precision(labels, scores) == 1.0


def test_equal_score_threshold_group_blocks_max_recall_at_100_precision():
    labels = [True, False, True]
    scores = [0.5, 0.5, 0.1]

    assert max_recall_at_100_precision(labels, scores) == 0.0


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_failure_assignment_rejects_non_finite_normal_scores(bad_score):
    with pytest.raises(ValueError, match="finite"):
        assign_failure_worst_scores([1.0, bad_score, None])


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_average_precision_rejects_non_finite_scores(bad_score):
    with pytest.raises(ValueError, match="finite"):
        average_precision([True, False], [0.9, bad_score])


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_max_recall_at_100_precision_rejects_non_finite_scores(bad_score):
    with pytest.raises(ValueError, match="finite"):
        max_recall_at_100_precision([True, False], [0.9, bad_score])


@pytest.mark.parametrize("bad_label", ["False", None, 2, "0"])
def test_average_precision_rejects_malformed_labels(bad_label):
    with pytest.raises(ValueError, match="labels must be bool"):
        average_precision([True, bad_label], [0.9, 0.1])


@pytest.mark.parametrize("bad_label", ["False", None, 2, "0"])
def test_max_recall_at_100_precision_rejects_malformed_labels(bad_label):
    with pytest.raises(ValueError, match="labels must be bool"):
        max_recall_at_100_precision([True, bad_label], [0.9, 0.1])
