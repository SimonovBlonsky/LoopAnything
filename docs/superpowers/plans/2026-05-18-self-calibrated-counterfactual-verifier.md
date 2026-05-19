# Self-Calibrated Counterfactual Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quickly validate whether rank/percentile self-calibrated counterfactual evidence beats SALAD-only and deformation-only ROVER scores using existing `candidate_records.jsonl`.

**Architecture:** This is a lightweight post-hoc score-sweep implementation. It only extends `robust_loop_verifier.score_sweep` and its tests; it does not rerun DA3, does not rerun PGO, and does not change the runtime verifier pipeline. Existing batch sweep scripts will pick up the new methods automatically because they call `compute_score_sweep()`.

**Tech Stack:** Python 3.9+, existing `robust_loop_verifier` package, pytest, existing score-sweep CLI.

**Spec:** `LoopAnything/docs/superpowers/specs/2026-05-18-self-calibrated-counterfactual-verifier-design.md`

**Git Policy:** Do not commit during execution unless the user explicitly requests it. Each task ends with `git status --short`.

---

## Scope

This plan intentionally avoids a large implementation. The first validation should answer one question:

```text
Do weight-free rank/percentile evidence scores improve AP or MR@100P over
SALAD-only and ROVER deformation-only on existing candidate records?
```

In scope:

- Query-local rank-product scores.
- Sequence-level percentile-product scores.
- Sequence-level percentile-min score.
- Score-sweep JSON/Markdown output through the existing CLI.
- Batch score sweep over already-generated run roots.

Out of scope:

- New DA3 forward logic.
- New PGO logic.
- New candidate record fields.
- Online AsterSLAM integration.
- Causal calibration buffer.
- Neural training or learned calibration.

## File Structure

- Modify `LoopAnything/src/robust_loop_verifier/score_sweep.py`
  - Add helper functions for candidate signal extraction, per-query ranks,
    empirical percentiles, rank-product scores, percentile-product scores, and
    percentile-min scores.
  - Add the new score rows to `compute_score_sweep()`.
- Modify `LoopAnything/tests/robust_loop_verifier/test_score_sweep.py`
  - Add focused tests for query-local reranking, invalid measurement handling,
    percentile calibration, and CLI artifact output.
- Optionally modify `LoopAnything/docs/superpowers/specs/2026-05-18-self-calibrated-counterfactual-verifier-design.md`
  - Only if implementation reveals a formula ambiguity.

## Method Names

Use stable method names so batch tables are easy to compare:

```text
rank_product:res,def
rank_product:salad,def
rank_product:salad,res
rank_product:salad,res,def
percentile_product:res,def
percentile_product:salad,def
percentile_product:salad,res
percentile_product:salad,res,def
percentile_min:salad,res,def
```

Signal conventions:

```text
salad      larger is better: score_salad or salad_score
residual   smaller is better: log1p(pgo_error_after)
deform     smaller is better: trajectory_deformation_rmse
```

Invalid graph measurements:

```text
rank-product graph signal rank = K + 1 inside that query group
percentile graph signal = 0.0
final metric code still uses assign_failure_worst_scores() for None scores
```

---

## Task 1: Query-Local Rank Product

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/score_sweep.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_score_sweep.py`

- [ ] **Step 1: Add failing test for graph-only query-local rank product**

Append this test to `tests/robust_loop_verifier/test_score_sweep.py`:

```python
def test_rank_product_res_def_reranks_within_each_query():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    records = [
        {
            "query_idx": 10,
            "candidate_idx": 1,
            "label": False,
            "score_salad": 0.99,
            "trajectory_deformation_rmse": 2.0,
            "pgo_error_after": 6.0,
        },
        {
            "query_idx": 10,
            "candidate_idx": 2,
            "label": True,
            "score_salad": 0.70,
            "trajectory_deformation_rmse": 0.1,
            "pgo_error_after": 0.1,
        },
        {
            "query_idx": 11,
            "candidate_idx": 3,
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 1.5,
            "pgo_error_after": 4.0,
        },
        {
            "query_idx": 11,
            "candidate_idx": 4,
            "label": True,
            "score_salad": 0.60,
            "trajectory_deformation_rmse": 0.2,
            "pgo_error_after": 0.2,
        },
    ]

    result = compute_score_sweep(records, graph_weights=(1.0,), fusion_weights=(1.0,))
    rows = {row["name"]: row for row in result["scores"]}

    assert rows["SALAD score only"]["AP"] < 1.0
    assert rows["rank_product:res,def"]["AP"] == 1.0
    assert rows["rank_product:res,def"]["MR@100P"] == 1.0
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier/test_score_sweep.py::test_rank_product_res_def_reranks_within_each_query -q
```

Expected: FAIL with `KeyError: 'rank_product:res,def'`.

- [ ] **Step 3: Add rank-product helpers**

In `src/robust_loop_verifier/score_sweep.py`, add these helpers near the other
private score helpers:

```python
RANK_PRODUCT_METHODS = (
    ("rank_product:res,def", ("residual", "deformation")),
    ("rank_product:salad,def", ("salad", "deformation")),
    ("rank_product:salad,res", ("salad", "residual")),
    ("rank_product:salad,res,def", ("salad", "residual", "deformation")),
)


def _rank_product_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    signals = _counterfactual_signals(records)
    groups = _query_groups(records)
    scores: list[float | None] = [None for _ in records]
    for indices in groups:
        query_size = len(indices)
        ranks_by_signal = {
            signal_name: _query_signal_ranks(
                [signals[signal_name][index] for index in indices],
                larger_is_better=_signal_larger_is_better(signal_name),
                invalid_rank=query_size + 1,
            )
            for signal_name in signal_names
        }
        for local_offset, record_index in enumerate(indices):
            rank_sum = 0.0
            for signal_name in signal_names:
                rank_sum += math.log(float(ranks_by_signal[signal_name][local_offset]))
            scores[record_index] = -rank_sum
    return scores


def _query_groups(records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    groups_by_query: dict[int, list[int]] = {}
    fallback_query = -1
    for index, record in enumerate(records):
        query_idx = record.get("query_idx")
        if query_idx is None:
            query_idx = fallback_query
            fallback_query -= 1
        groups_by_query.setdefault(int(query_idx), []).append(index)
    return list(groups_by_query.values())


def _query_signal_ranks(
    values: Sequence[float | None],
    *,
    larger_is_better: bool,
    invalid_rank: int,
) -> list[int]:
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    valid.sort(key=lambda item: item[1], reverse=larger_is_better)
    ranks = [invalid_rank for _ in values]
    current_rank = 1
    previous_value: float | None = None
    for offset, (index, value) in enumerate(valid):
        if previous_value is None or value != previous_value:
            current_rank = offset + 1
            previous_value = value
        ranks[index] = current_rank
    return ranks
```

- [ ] **Step 4: Add shared signal extraction helpers**

Add these helpers below `_salad_scores()`:

```python
def _counterfactual_signals(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float | None]]:
    return {
        "salad": _salad_scores(records),
        "residual": [_negative(_log1p_or_none(record.get("pgo_error_after"))) for record in records],
        "deformation": [
            _negative(_finite_float_or_none(record.get("trajectory_deformation_rmse")))
            for record in records
        ],
    }


def _signal_larger_is_better(signal_name: str) -> bool:
    if signal_name in {"salad", "residual", "deformation"}:
        return True
    raise ValueError(f"unknown signal name: {signal_name}")
```

The helper stores residual/deformation as larger-is-better negative costs, so
all rank code can use descending rank for every current signal.

- [ ] **Step 5: Register rank-product rows**

Inside `compute_score_sweep()`, after the support-ensemble row block and before
manual graph/fusion sweeps, add:

```python
    for method_name, signal_names in RANK_PRODUCT_METHODS:
        score_rows.append(
            _metrics_row(
                method_name,
                labels,
                _rank_product_scores(records, signal_names),
            )
        )
```

- [ ] **Step 6: Run focused rank-product test**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier/test_score_sweep.py::test_rank_product_res_def_reranks_within_each_query -q
```

Expected: PASS.

- [ ] **Step 7: Check status**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected: only `score_sweep.py` and `test_score_sweep.py` are modified.

---

## Task 2: Sequence-Level Percentile Evidence

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/score_sweep.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_score_sweep.py`

- [ ] **Step 1: Add failing test for percentile product**

Append this test:

```python
def test_percentile_product_scores_global_candidate_distribution():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    records = [
        {
            "query_idx": 10,
            "candidate_idx": 1,
            "label": False,
            "score_salad": 0.99,
            "trajectory_deformation_rmse": 2.0,
            "pgo_error_after": 6.0,
        },
        {
            "query_idx": 10,
            "candidate_idx": 2,
            "label": True,
            "score_salad": 0.70,
            "trajectory_deformation_rmse": 0.1,
            "pgo_error_after": 0.1,
        },
        {
            "query_idx": 11,
            "candidate_idx": 3,
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 1.5,
            "pgo_error_after": 4.0,
        },
        {
            "query_idx": 11,
            "candidate_idx": 4,
            "label": True,
            "score_salad": 0.60,
            "trajectory_deformation_rmse": 0.2,
            "pgo_error_after": 0.2,
        },
    ]

    result = compute_score_sweep(records, graph_weights=(1.0,), fusion_weights=(1.0,))
    rows = {row["name"]: row for row in result["scores"]}

    assert rows["percentile_product:res,def"]["AP"] == 1.0
    assert rows["percentile_product:res,def"]["MR@100P"] == 1.0
    assert rows["percentile_min:salad,res,def"]["AP"] < 1.0
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier/test_score_sweep.py::test_percentile_product_scores_global_candidate_distribution -q
```

Expected: FAIL with `KeyError: 'percentile_product:res,def'`.

- [ ] **Step 3: Add percentile method constants**

In `score_sweep.py`, add below `RANK_PRODUCT_METHODS`:

```python
PERCENTILE_PRODUCT_METHODS = (
    ("percentile_product:res,def", ("residual", "deformation")),
    ("percentile_product:salad,def", ("salad", "deformation")),
    ("percentile_product:salad,res", ("salad", "residual")),
    ("percentile_product:salad,res,def", ("salad", "residual", "deformation")),
)

PERCENTILE_MIN_METHODS = (
    ("percentile_min:salad,res,def", ("salad", "residual", "deformation")),
)

PERCENTILE_EPS = 1e-6
```

- [ ] **Step 4: Add percentile helpers**

Add below `_rank_product_scores()`:

```python
def _percentile_product_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    percentiles = _percentile_features(records)
    scores: list[float | None] = []
    for index in range(len(records)):
        value = 0.0
        for signal_name in signal_names:
            percentile = percentiles[signal_name][index]
            value += math.log(PERCENTILE_EPS + percentile)
        scores.append(value)
    return scores


def _percentile_min_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    percentiles = _percentile_features(records)
    scores: list[float | None] = []
    for index in range(len(records)):
        scores.append(min(percentiles[signal_name][index] for signal_name in signal_names))
    return scores


def _percentile_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    signals = _counterfactual_signals(records)
    return {
        signal_name: _empirical_percentiles(values)
        for signal_name, values in signals.items()
    }


def _empirical_percentiles(values: Sequence[float | None]) -> list[float]:
    valid = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not valid:
        return [0.0 for _ in values]
    denominator = float(len(valid))
    percentiles: list[float] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            percentiles.append(0.0)
            continue
        count_leq = _count_less_equal(valid, float(value))
        percentiles.append(count_leq / denominator)
    return percentiles


def _count_less_equal(sorted_values: Sequence[float], value: float) -> int:
    left = 0
    right = len(sorted_values)
    while left < right:
        middle = (left + right) // 2
        if sorted_values[middle] <= value:
            left = middle + 1
        else:
            right = middle
    return left
```

Because residual and deformation are already stored as negative costs in
`_counterfactual_signals()`, a larger percentile still means better.

- [ ] **Step 5: Register percentile rows**

Inside `compute_score_sweep()`, after rank-product rows, add:

```python
    for method_name, signal_names in PERCENTILE_PRODUCT_METHODS:
        score_rows.append(
            _metrics_row(
                method_name,
                labels,
                _percentile_product_scores(records, signal_names),
            )
        )
    for method_name, signal_names in PERCENTILE_MIN_METHODS:
        score_rows.append(
            _metrics_row(
                method_name,
                labels,
                _percentile_min_scores(records, signal_names),
            )
        )
```

- [ ] **Step 6: Run focused percentile test**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier/test_score_sweep.py::test_percentile_product_scores_global_candidate_distribution -q
```

Expected: PASS.

- [ ] **Step 7: Run all score-sweep tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier/test_score_sweep.py -q
```

Expected: PASS.

- [ ] **Step 8: Check status**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected: only `score_sweep.py` and `test_score_sweep.py` are modified.

---

## Task 3: CLI And Batch Artifact Smoke Test

**Files:**
- Modify: `LoopAnything/tests/robust_loop_verifier/test_score_sweep.py`
- No production file should be needed if Tasks 1-2 are correct.

- [ ] **Step 1: Extend existing CLI artifact test**

In `test_sweep_scores_cli_writes_json_and_markdown`, after the existing
Markdown assertion, add:

```python
    assert "| rank_product:res,def |" in sweep_md.read_text(encoding="utf-8")
    assert "| percentile_product:res,def |" in sweep_md.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run CLI artifact test**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier/test_score_sweep.py::test_sweep_scores_cli_writes_json_and_markdown -q
```

Expected: PASS. If it fails because the Markdown output limit excludes the new
methods, change `write_score_sweep_markdown()` default `limit` from `40` to
`80` and rerun this test.

- [ ] **Step 3: Run full robust loop verifier unit tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest \
  tests/robust_loop_verifier -q
```

Expected: PASS.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 5: Check status**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected: only `score_sweep.py` and `test_score_sweep.py` are modified unless
`write_score_sweep_markdown()` limit was changed.

---

## Task 4: Existing Handheld Result Sweep

**Files:**
- No source changes expected.

- [ ] **Step 1: Run score sweep on the 4-sequence batch summary**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/batch_score_sweep.py \
  --batch-summary workspace/robust_loop_verifier_runs/FusionPortableV2/fusionportablev2_batch_20260518_195158/batch_summary.json \
  --output-root workspace/robust_loop_verifier_runs/FusionPortableV2/fusionportablev2_batch_20260518_195158/self_calibrated_score_sweep
```

Expected:

```text
batch_score_sweep_json=.../self_calibrated_score_sweep/batch_score_sweep.json
batch_score_sweep_average_csv=.../self_calibrated_score_sweep/batch_score_sweep_average.csv
batch_score_sweep_per_sequence_csv=.../self_calibrated_score_sweep/batch_score_sweep_per_sequence.csv
```

- [ ] **Step 2: Run score sweep on `handheld_escalator00`**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m robust_loop_verifier.cli sweep-scores \
  --candidate-records workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/handheld_escalator00_full_support_ensemble_20260518_172817_support_ensemble/candidate_records.jsonl \
  --output-root workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/handheld_escalator00_full_support_ensemble_20260518_172817_support_ensemble/self_calibrated_score_sweep
```

Expected:

```text
best_ap=...
best_mr=...
```

- [ ] **Step 3: Inspect top methods**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
sed -n '1,40p' \
  workspace/robust_loop_verifier_runs/FusionPortableV2/fusionportablev2_batch_20260518_195158/self_calibrated_score_sweep/batch_score_sweep_average.md
sed -n '1,40p' \
  workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/handheld_escalator00_full_support_ensemble_20260518_172817_support_ensemble/self_calibrated_score_sweep/score_sweep.md
```

Expected: the table includes the new `rank_product:*` and `percentile_*`
methods. Record whether any of them beat:

```text
SALAD score only
ROVER deformation only
DA3-ROVER++ support ensemble graph evidence
manual z_fusion diagnostic upper bound
```

- [ ] **Step 4: Summarize go/no-go**

Write a short conclusion in the final response:

```text
If self-calibrated methods beat ROVER deformation-only on average AP or MR@100P:
  proceed to full FusionPortableV2 batch and then runtime-oriented design.
If they do not:
  do not promote this method as the paper main line; revisit graph response
  features or support-ensemble diagnostics.
```

- [ ] **Step 5: Do not commit unless requested**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected: source/test changes are visible for user review; generated workspace
outputs are not committed.

