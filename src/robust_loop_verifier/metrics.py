"""Candidate-level metrics for robust loop verification."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


def assign_failure_worst_scores(scores: Iterable[float | None]) -> List[float]:
    """Replace failed candidate scores with deterministic finite worst scores.

    Canonical scores are larger-is-better. Finite scores are preserved. ``None``
    values represent failures and are placed below every normal score, with
    distinct values that preserve input order under a stable descending sort.
    """

    fixed: List[float | None] = list(scores)
    normal_scores = [score for score in fixed if score is not None]
    for score in normal_scores:
        _validate_finite_score(score)

    failure_indices = [index for index, score in enumerate(fixed) if score is None]
    if not failure_indices:
        return [float(score) for score in fixed if score is not None]

    if not normal_scores:
        failure_scores = [-(rank + 1.0) for rank in range(len(failure_indices))]
    else:
        failure_scores = _finite_scores_below(min(normal_scores), len(failure_indices))

    for index, failure_score in zip(failure_indices, failure_scores):
        fixed[index] = failure_score

    return [float(score) for score in fixed if score is not None]


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute tie-safe average precision over larger-is-better scores."""

    labels, scores = _validate_labels_and_scores(labels, scores)
    total_positives = sum(labels)
    if total_positives == 0:
        return 0.0

    positives_seen = 0
    candidates_seen = 0
    previous_recall = 0.0
    result = 0.0
    for group in _score_threshold_groups(labels, scores):
        candidates_seen += len(group)
        positives_seen += sum(label for label, _ in group)
        recall = positives_seen / total_positives
        precision = positives_seen / candidates_seen
        result += (recall - previous_recall) * precision
        previous_recall = recall

    return result


def max_recall_at_100_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return maximum recall observed while threshold precision remains exactly 1.0."""

    labels, scores = _validate_labels_and_scores(labels, scores)
    total_positives = sum(labels)
    if total_positives == 0:
        return 0.0

    positives_seen = 0
    candidates_seen = 0
    best_recall = 0.0
    for group in _score_threshold_groups(labels, scores):
        candidates_seen += len(group)
        positives_seen += sum(label for label, _ in group)
        if positives_seen == candidates_seen:
            best_recall = positives_seen / total_positives

    return best_recall


def _finite_scores_below(upper_bound: float, count: int) -> List[float]:
    scores: List[float] = []
    current = float(upper_bound)
    for _ in range(count):
        current = math.nextafter(current, -math.inf)
        if not math.isfinite(current):
            raise ValueError("cannot assign finite failure score below minimum normal score")
        scores.append(current)
    return scores


def _validate_labels_and_scores(
    labels: Sequence[bool], scores: Sequence[float]
) -> Tuple[List[bool], List[float]]:
    labels_list = list(labels)
    scores_list = [float(score) for score in scores]
    if len(labels_list) != len(scores_list):
        raise ValueError("labels and scores must have the same length")
    for label in labels_list:
        if type(label) is not bool:
            raise ValueError("labels must be bool values")
    for score in scores_list:
        _validate_finite_score(score)
    return labels_list, scores_list


def _validate_finite_score(score: float) -> None:
    if not math.isfinite(score):
        raise ValueError("scores must be finite")


def _stable_score_order(labels: Sequence[bool], scores: Sequence[float]) -> List[Tuple[bool, float]]:
    indexed = enumerate(zip(labels, scores))
    return [item for _, item in sorted(indexed, key=lambda entry: (-entry[1][1], entry[0]))]


def _score_threshold_groups(
    labels: Sequence[bool], scores: Sequence[float]
) -> List[List[Tuple[bool, float]]]:
    ordered = _stable_score_order(labels, scores)
    groups: List[List[Tuple[bool, float]]] = []
    for label, score in ordered:
        if not groups or score != groups[-1][0][1]:
            groups.append([])
        groups[-1].append((label, score))
    return groups
