# Self-Calibrated Counterfactual Loop Verification

Status: design spec. This follows the mixed support-ensemble graph-evidence
result and defines the next DA3-ROVER++ research direction.

## Motivation

The support-ensemble graph-evidence experiment produced a mixed result. On the
four-sequence FusionPortableV2 handheld batch `20260518_195158`, it was weaker
than deformation-only ROVER-style scoring on average:

```text
SALAD score only:                              AP=0.620, MR@100P=0.252
ROVER deformation only:                       AP=0.893, MR@100P=0.400
DA3-ROVER++ support ensemble graph evidence:  AP=0.784, MR@100P=0.383
best diagnostic z_fusion upper bound:         AP=0.913, MR@100P=0.611
```

But on the full `handheld_escalator00` support-ensemble run, it strongly
outperformed deformation-only scoring:

```text
handheld_escalator00:
ROVER deformation only:                       AP=0.695, MR@100P=0.020
DA3-ROVER++ support ensemble graph evidence:  AP=0.906, MR@100P=0.159
```

Combining the five available handheld sequences gives:

```text
ROVER deformation only:                       AP=0.853, MR@100P=0.324
DA3-ROVER++ support ensemble graph evidence:  AP=0.808, MR@100P=0.338
```

This means support-level graph evidence should not be promoted as the sole
paper main method, but it remains valuable. It may provide diagnostic
information for escalator-like cases where deformation-only scoring is too
weak. The limitations are conceptual, not just
implementation-specific:

- A robust PGO backend can absorb false loop factors, so optimized graph
  evidence alone is not equivalent to loop correctness.
- Inflating uncertainty from support disagreement can protect bad loop factors
  by reducing their normalized residual.
- Supports near the same candidate are correlated perturbations, not
  independent loop evidence.
- DA3 support consistency is not strongly aligned with GT loop truth.

The useful signal from the experiments is broader: when DA3 loop factors are
tested counterfactually inside the graph, residual response, trajectory
deformation, and support-ensemble graph evidence can each help in different
failure modes. Fixed-weight `z_fusion` shows the signal is present, but it is
not a defensible main method because manual weights are easy to criticize.

The new main direction is therefore:

```text
DA3 loop factor + counterfactual PGO response + self-calibrated evidence
```

## Goal

Design a training-free loop verifier that is still ROVER-like, but improves on
deformation-only scoring by adding residual-aware graph response and
test-time calibration.

The method should:

- Use SALAD only as a top-k proposal generator and weak retrieval prior.
- Use DA3 to construct a metric loop factor for each retrieved candidate.
- Temporarily insert the DA3 loop factor into a prefix graph and observe the
  counterfactual PGO response.
- Score candidates using residual, deformation, and retrieval-prior evidence.
- Avoid supervised training, GT-dependent calibration, and hand-tuned linear
  fusion weights.
- Produce both offline AP/MR evaluation scores and an online-compatible
  top-k reranking path for AsterSLAM.

## Non-Goals

This spec does not revive learned loop policy features, labels, gates, or
preconditions. Those remain deferred and must not be used as priors.

This spec does not use support-ensemble graph evidence as the sole main method.
Multi-support may remain as loop-factor stabilization, an ablation, or an
additional diagnostic signal, but the main score is candidate-level
self-calibrated counterfactual evidence.

This spec does not add a neural classifier or learned uncertainty head.

## Candidate Pool

For each query keyframe `q`, SALAD retrieves a small historical candidate pool:

```text
C_q = {c_1, ..., c_K}
```

Default settings:

```text
main top_k = 10
ablation top_k = 5, 20
recent exclusion = existing retrieval recent filter
support selection = candidate-neighborhood support used by the current
                    single-support DA3 factor path
```

The default DA3 factor path should start from the current single-support
implementation because it is already stronger than support-ensemble graph
evidence in the ROVER deformation setting. Multi-support is allowed only as an
ablation until it proves useful.

## Counterfactual Measurements

For each candidate `(q, c)`:

1. Select a candidate-neighborhood support frame.
2. Run DA3 on the isolated triplet `[query, candidate, support]`.
3. Convert DA3 output to the correct `c2w` convention.
4. Align DA3 to metric scale using candidate-support odometry.
5. Extract the query-candidate loop factor `T_qc_da3`.
6. Insert `T_qc_da3` into the full-prefix graph ending at `q`.
7. Run PGO and record graph-response diagnostics.

The required candidate record fields are:

```text
salad_score
salad_rank
pgo_error_after
pgo_loop_chi2_after
pgo_odom_strain_chi2_after
trajectory_deformation_rmse
sim3_support_alignment_residual_m
sim3_direction_error_deg
da3_factor_valid
```

The primary graph-response signals are:

```text
residual_cost = log1p(pgo_error_after)
deformation_cost = trajectory_deformation_rmse
retrieval_cost = salad_rank
```

Optional diagnostics:

```text
loop_chi2_cost = log1p(pgo_loop_chi2_after)
odom_strain_cost = log1p(pgo_odom_strain_chi2_after)
sim3_alignment_cost = sim3_support_alignment_residual_m
```

## Query-Local Rank Evidence

Query-local rank evidence is the online-compatible reranking path.

For a fixed query `q`, rank all candidates in `C_q` independently by each
signal:

```text
r_salad(c) = rank of c by descending SALAD score
r_res(c)   = rank of c by ascending residual_cost
r_def(c)   = rank of c by ascending deformation_cost
```

Ranks start at one. Invalid DA3/PGO candidates receive rank `K + 1` for the
affected graph signals.

The weight-free query-local score is rank product:

```text
score_rank_product(c) =
  -log(r_salad(c)) - log(r_res(c)) - log(r_def(c))
```

Equivalent form:

```text
score_rank_product(c) = -log(r_salad(c) * r_res(c) * r_def(c))
```

This score has no hand-tuned weights. It asks whether a candidate is
simultaneously good under retrieval, residual response, and deformation
response.

For diagnostics, run the following ablations:

```text
rank_product(res, def)
rank_product(salad, def)
rank_product(salad, res)
rank_product(salad, res, def)
```

Expected behavior:

- `rank_product(res, def)` tests whether graph evidence alone can rerank.
- `rank_product(salad, res, def)` tests whether retrieval prior helps when
  graph evidence is ambiguous.
- If adding SALAD dominates the score, the method is not sufficiently
  graph-driven and should be reported carefully.

## Sequence-Level Percentile Evidence

Sequence-level percentile evidence is the offline AP/MR scoring path and the
online path once a sliding calibration buffer exists.

For each sequence, collect valid candidate measurements across all query top-k
sets:

```text
S_salad = {salad_score(c)}
S_res   = {-residual_cost(c)}
S_def   = {-deformation_cost(c)}
```

Convert each candidate to an empirical percentile:

```text
P_salad(c) = percentile_rank(salad_score(c), S_salad)
P_res(c)   = percentile_rank(-residual_cost(c), S_res)
P_def(c)   = percentile_rank(-deformation_cost(c), S_def)
```

`P=1` means the candidate is among the best candidates in the current
calibration set. Invalid measurements receive `P=0`.

The default weight-free percentile evidence is Fisher-style log evidence:

```text
score_percentile(c) =
  log(eps + P_salad(c)) +
  log(eps + P_res(c)) +
  log(eps + P_def(c))
```

with fixed numerical stability:

```text
eps = 1e-6
```

This is not a tunable fusion weight. It is an equal-evidence product in
percentile space.

Run these percentile ablations:

```text
percentile_product(res, def)
percentile_product(salad, def)
percentile_product(salad, res)
percentile_product(salad, res, def)
percentile_min(salad, res, def)
```

`percentile_min` is stricter:

```text
score_percentile_min(c) = min(P_salad(c), P_res(c), P_def(c))
```

It is useful to test whether all evidence types must agree.

## Calibration Scope

Offline experiments may use the whole evaluated sequence as the calibration
set, because AP/MR are computed after the full run.

For online AsterSLAM, the same formula should be implemented with a causal
calibration buffer:

```text
calibration set at query q = candidates processed before q
```

Minimum viable online policy:

```text
if buffer_size < N_min:
  use query-local rank product only
else:
  use causal percentile evidence
```

Suggested initial constants:

```text
N_min = 100 candidate records
buffer = all previous records in the current sequence
```

These constants control estimator stability, not fusion weighting. They should
be reported as runtime calibration settings.

## Why This Is Different From Manual z_fusion

Manual `z_fusion` uses weights such as:

```text
0.5 * z_salad - 4.0 * z_deformation - 0.5 * z_residual
```

This is useful as a diagnostic upper bound, but it is vulnerable to the
criticism that the weights were tuned on the test set.

Self-calibrated evidence avoids this by:

- Ranking or percentiling each signal inside the current candidate
  distribution.
- Combining evidence with equal rank/product rules.
- Using no dataset-specific learned parameters or supervised labels.
- Keeping SALAD, residual, and deformation units out of the final score.

The scientific claim is not that a linear fusion was tuned well. The claim is:

```text
false loops are exposed by counterfactual graph response, and the response can
be calibrated at test time without labels.
```

## Relationship To ROVER

ROVER-style reasoning asks whether adding a candidate loop factor causes
implausible trajectory deformation.

This method keeps the same counterfactual spirit but extends it:

```text
ROVER-like:        deformation after adding the factor
ours:              deformation + residual consistency + retrieval prior,
                   calibrated by test-time candidate distributions
DA3 contribution:  a 3D foundation model produces the loop factor from images
                   without a trained loop verifier
```

The method should be described as residual-aware and self-calibrated rather
than as a generic score fusion.

## Evaluation Plan

Main FusionPortableV2 table:

```text
SALAD only
ROVER deformation only
PGO residual only
manual z_fusion diagnostic upper bound
query-local rank_product(res, def)
query-local rank_product(salad, res, def)
sequence percentile_product(res, def)
sequence percentile_product(salad, res, def)
sequence percentile_min(salad, res, def)
```

Primary metrics:

```text
AP
MR@100P
```

Secondary diagnostics:

```text
top-1 query recall after reranking
top-k candidate recall before reranking
false-positive examples at highest scores
score distribution plots for positives and negatives
```

Expected success criterion:

```text
sequence percentile or query-local rank evidence should beat both
SALAD-only and ROVER deformation-only on average AP or MR@100P,
without using manual fusion weights.
```

If the self-calibrated method cannot beat ROVER deformation-only, the current
innovation path is not strong enough for the main paper method.

## Implementation Boundary

The first implementation should reuse existing candidate records and score
sweep infrastructure. It should not rerun DA3 or PGO until the post-hoc score
logic is validated.

Initial implementation tasks:

1. Add query-local rank evidence computation from `candidate_records.jsonl`.
2. Add sequence-level percentile evidence computation.
3. Extend batch score sweep outputs with the new non-weighted methods.
4. Run on existing handheld batch results.
5. If promising, rerun full FusionPortableV2 handheld/ugv/legged batches.

Only after post-hoc evidence is promising should runtime candidate reranking be
added to AsterSLAM.

## Reporting Rules

Do not present manual `z_fusion` as the main method. It is an analysis upper
bound.

Do not present support-ensemble graph evidence as a complete standalone main
method yet. It is a mixed ablation: valuable on `handheld_escalator00`, weaker
on average AP, and worth preserving as a diagnostic/component candidate.

Do not use learned loop-policy documents, labels, rules, preconditions, or hard
gates as priors.

Do not batch independent candidates or triplets into one DA3 forward, because
DA3 cross-view attention would contaminate geometric groups.
