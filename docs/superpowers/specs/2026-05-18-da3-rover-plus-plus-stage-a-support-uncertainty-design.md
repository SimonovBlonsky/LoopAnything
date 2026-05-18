# DA3-ROVER++ Stage A: Support-Ensemble Test-Time Uncertainty

Status: paused / mixed ablation result as of 2026-05-18. This spec documents
the support-ensemble test-time uncertainty idea and the experiments showing
that it is useful on some sequences but not yet robust enough as the sole main
DA3-ROVER++ score.

Current decision:

```text
Do not use support-ensemble graph evidence as the sole paper main method yet.
Keep it as experimental infrastructure, diagnostics, and an important ablation.
```

The active follow-up direction is:

```text
LoopAnything/docs/superpowers/specs/2026-05-18-self-calibrated-counterfactual-verifier-design.md
```

## 2026-05-18 Results

FusionPortableV2 handheld results are mixed. The four-sequence batch
`20260518_195158` suggested support-ensemble graph evidence was weaker than
deformation-only ROVER-style scoring on average:

```text
SALAD score only:                              AP=0.620, MR@100P=0.252
ROVER deformation only:                       AP=0.893, MR@100P=0.400
DA3-ROVER++ support ensemble graph evidence:  AP=0.784, MR@100P=0.383
best diagnostic z_fusion upper bound:         AP=0.913, MR@100P=0.611
```

However, the earlier full support-ensemble run on `handheld_escalator00` shows
that the idea is not valueless:

```text
handheld_escalator00:
SALAD score only:                              AP=0.834, MR@100P=0.087
ROVER deformation only:                       AP=0.695, MR@100P=0.020
DA3-ROVER++ support ensemble graph evidence:  AP=0.906, MR@100P=0.159
```

Combining `handheld_escalator00` with the four-sequence batch gives:

| sequence | SALAD AP/MR | ROVER AP/MR | support AP/MR | support - ROVER AP | support - ROVER MR |
| --- | ---: | ---: | ---: | ---: | ---: |
| handheld_escalator00 | 0.834/0.087 | 0.695/0.020 | 0.906/0.159 | +0.211 | +0.139 |
| handheld_escalator01 | 0.841/0.140 | 0.860/0.124 | 0.891/0.193 | +0.031 | +0.069 |
| handheld_grass00 | 0.082/0.000 | 0.765/0.300 | 0.395/0.300 | -0.370 | +0.000 |
| handheld_room00 | 0.603/0.396 | 0.956/0.434 | 0.883/0.472 | -0.073 | +0.038 |
| handheld_room01 | 0.955/0.473 | 0.989/0.743 | 0.966/0.568 | -0.024 | -0.176 |

| method | average AP | average MR@100P |
| --- | ---: | ---: |
| SALAD score only | 0.663 | 0.219 |
| SALAD + DA3-ROVER full-prefix trajectory score | 0.853 | 0.324 |
| DA3-ROVER++ support ensemble graph evidence | 0.808 | 0.338 |
| SALAD + DA3/Sim3 self-consistency score | 0.190 | 0.000 |

Interpretation:

- Support-ensemble graph evidence is valuable on escalator-style sequences and
  can greatly improve over deformation-only scoring there.
- It is not robust enough as a single global score because AP drops badly on
  `handheld_grass00` and moderately on room sequences.
- The original Stage A success criterion is therefore not fully satisfied, but
  the idea should remain as an ablation and potential diagnostic component.

Likely failure modes of using support-ensemble graph evidence alone are:

- Robust PGO can absorb false loop factors, so optimized graph evidence alone
  is not equivalent to loop correctness.
- Uncertainty inflation can protect bad factors by reducing normalized
  residuals.
- Candidate-neighborhood supports are correlated perturbations, not independent
  loop evidence.
- DA3 support consistency is not strongly aligned with GT loop truth.

The useful follow-up is to combine the support-ensemble insight with
counterfactual graph response: residual and trajectory deformation after
inserting a DA3 loop factor. That motivates the self-calibrated counterfactual
verifier spec above.

## Purpose

Stage A makes a single DA3-derived loop measurement reliable before designing
candidate-level reranking. Given a fixed retrieval pair `(query, candidate)`,
the method perturbs only the support frame, runs DA3 on isolated triplets, and
uses the resulting loop-factor variability as a test-time uncertainty estimate.

The target contribution is:

```text
training-free DA3 test-time uncertainty for SLAM loop measurements
```

DA3 does not natively output pose covariance. Its `Prediction.conf` field is a
depth confidence map, not a loop-factor uncertainty. Stage A therefore derives
uncertainty from support perturbation:

```text
same query-candidate pair + different candidate-neighborhood supports
-> multiple DA3 loop factors
-> support-level consistency
-> loop-factor mean and covariance
```

## Scope

Stage A includes:

- Multi-support DA3 loop measurement for a fixed `(query, candidate)` pair.
- Support-level quality diagnostics and robust support weighting.
- Robust SE(3) loop-factor aggregation.
- Diagonal DA3 loop-factor covariance estimated from support perturbation.
- Covariance-aware full-prefix PGO and graph-evidence scoring.
- Ablations that isolate whether support uncertainty improves AP and MR.

Stage A excludes:

- Candidate-level top-k reranking as the main method.
- Learned loop-policy labels, hard gates, or feature schemas.
- A neural verifier, classifier, logistic calibrator, or supervised uncertainty
  head.
- Cross-candidate or cross-triplet DA3 batching.

Candidate-level reranking is Stage B. Stage A only improves the measurement
quality for each already-retrieved candidate.

## DA3 Isolation Constraint

DA3's vision transformer performs cross-view attention over the images in a
forward pass. Therefore, independent candidates or independent triplets must not
be placed in the same DA3 batch.

Allowed Stage A forward:

```text
[query, candidate, support_i]
```

Forbidden forward:

```text
batch([query, candidate, support_1], [query, candidate, support_2])
batch([query, candidate_1, support_1], [query, candidate_2, support_2])
[query, candidate_1, support_1, query, candidate_2, support_2]
```

Each support perturbation must be an isolated DA3 geometric group. Parallelism
may only happen outside the DA3 forward, for example by scheduling independent
forwards across time, processes, or GPUs.

The default Stage A setting is:

```text
support_count = 4
DA3 forwards per candidate = 4 isolated triplets
retrieval_top_k = inherited from the current offline run, not reranked by Stage A
```

## Pipeline

For each candidate pair `(q, c)` from the existing retrieval output:

1. Select `support_count` candidate-neighborhood support frames.
2. For each support `s_i`, run DA3 on the isolated triplet `[q, c, s_i]`.
3. Convert DA3 extrinsics from native `w2c` to `c2w`.
4. Align the DA3 triplet to metric scale using the candidate-support odometry
   prior.
5. Extract one metric query-candidate loop factor `T_qc_i`.
6. Estimate support-level weights from alignment quality and cross-support
   consensus.
7. Aggregate `{T_qc_i}` into one robust mean loop factor `T_qc_mean`.
8. Estimate a diagonal covariance `Sigma_da3` from weighted SE(3) residuals.
9. Add `T_qc_mean` to full-prefix PGO using `Sigma_da3` as the loop noise.
10. Score the candidate with graph evidence.

The output for each candidate is:

```text
T_qc_mean
Sigma_da3
support weights
support residual diagnostics
graph evidence score
```

## Support Selection

Support selection keeps the current candidate-centered rule:

```text
valid supports are near the candidate keyframe index,
pass image and pose availability checks,
pass candidate-support minimum baseline,
and pass query-relative recent exclusion.
```

After filtering, supports are sorted by nearest candidate keyframe index. Stage
A takes the first `support_count` supports from this ordered list. This preserves
the previous finding that candidate-neighborhood supports tend to keep visual
overlap better than maximizing candidate-support baseline.

If fewer than two valid supports are available, Stage A falls back to the
single-support baseline and marks the covariance as floor-only.

## Support-Level Weight

For each support `s_i`, DA3 and Sim3 alignment produce:

```text
T_qc_i
support_alignment_residual_m_i
direction_error_deg_i
candidate_support_baseline_m_i
```

Stage A uses two training-free terms:

```text
w_i = w_align_i * w_consensus_i
```

The alignment term measures whether the support itself is a valid scale anchor:

```text
e_align_i =
  support_alignment_residual_m_i / max(candidate_support_baseline_m_i, eps)
  + lambda_dir * 2 * sin(direction_error_rad_i / 2)

w_align_i = Cauchy(e_align_i / c_align)
```

The consensus term measures whether this support agrees with the other supports:

```text
e_consensus_i = norm(Log(T_mean^-1 * T_qc_i), Sigma_floor^-1)
w_consensus_i = HuberWeight(e_consensus_i / c_consensus)
```

`T_mean` is initialized by an unweighted SE(3) mean and refined with robust
weights for a small fixed number of iterations.

These weights do not use GT labels, query-candidate odometry consistency, SALAD
score, learned-policy features, or learned thresholds.

## Loop-Factor Mean And Covariance

Stage A computes a weighted robust SE(3) mean:

```text
T_qc_mean = RobustMeanSE3({T_qc_i}, weights=w_i)
r_i = Log(T_qc_mean^-1 * T_qc_i)
```

The residual vector convention is:

```text
r_i = [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
```

Because the default support count is four, full empirical `6x6` covariance is
rank-deficient. The main method therefore uses a diagonal covariance with a
noise floor:

```text
sigma_rot2 =
  weighted_rms(r_i[0:3])^2 + sigma_rot_floor^2

sigma_trans2 =
  weighted_rms(r_i[3:6])^2 + sigma_trans_floor^2

Sigma_da3 = diag(
  sigma_rot2, sigma_rot2, sigma_rot2,
  sigma_trans2, sigma_trans2, sigma_trans2
)
```

Optional later ablations may evaluate shrinkage full covariance when the support
count is increased, but full covariance is not the Stage A main method.

## Graph Evidence Score

The main score is a larger-is-better negative graph-evidence objective:

```text
score_stage_a = -graph_evidence_NLL
```

The default graph evidence is:

```text
graph_evidence_NLL =
  0.5 * chi2_loop_after
  + 0.5 * chi2_odom_strain_after
  + 0.5 * logdet(Sigma_da3 / Sigma_floor)
```

Definitions:

- `chi2_loop_after`: whitened residual of the DA3 loop factor after PGO.
- `chi2_odom_strain_after`: whitened increase of odometry-chain residual after
  accepting the loop factor.
- `logdet(Sigma_da3 / Sigma_floor)`: uncertainty penalty that prevents an
  unstable DA3 measurement from passing only because its covariance is large.

All three terms have probabilistic measurement-model meaning. This replaces
manual z-score fusion weights and avoids using SALAD score in the main verifier
score.

## Calibration Policy

The main research setting allows dataset/platform-level calibration of
measurement-model parameters:

```text
sigma_rot_floor
sigma_trans_floor
covariance scale alpha
c_align
c_consensus
lambda_dir
support_count
support_window
```

This calibration is treated as SLAM system configuration, not learned verifier
training. It must not train a network, train a classifier, or fit a supervised
loop probability model.

To address generalization, Stage A should report:

- A calibrated main result.
- A fixed-parameter variant.
- A leave-one-sequence-out calibration variant when enough sequences are
  available.

## Required Ablations

Stage A is considered experimentally useful only if the ablations identify where
the gain comes from:

```text
A0: single nearest support + fixed loop noise
A1: multi-support uniform mean + fixed loop noise
A2: multi-support robust mean + fixed loop noise
A3: robust mean + Sigma_da3 covariance-aware PGO
A4: A3 + logdet uncertainty penalty
```

Expected interpretation:

- `A1 > A0`: support perturbation improves measurement stability.
- `A2 > A1`: robust support weighting handles bad supports.
- `A3 > A2`: DA3 test-time covariance improves the PGO measurement model.
- `A4 > A3`: uncertainty penalty prevents over-permissive large covariance.

## Diagnostics

Stage A artifacts should make the uncertainty interpretable:

- Per-support triplet visualizations.
- Per-support `T_qc_i`, alignment residual, direction error, and weight.
- SE(3) residual distribution around `T_qc_mean`.
- `Sigma_da3` trace, logdet, rotation sigma, and translation sigma.
- Graph residual before and after PGO.
- False-positive examples where support uncertainty is high.
- True-positive examples where support uncertainty is low.

The key diagnostic claim to validate is:

```text
DA3 support-perturbation uncertainty correlates with loop-factor reliability.
```

## Metrics

Stage A keeps the existing candidate-level evaluation:

```text
AP
MR@100P
```

It also logs uncertainty-specific diagnostics:

```text
correlation(Sigma_da3, loop-factor GT error)
true-loop vs false-loop uncertainty distributions
effective_support_count
support outlier rate
```

GT labels are used only for evaluation and diagnostics, never for support
weights, covariance estimation, PGO scoring, or verifier acceptance.

## Relationship To Stage B

Stage A produces one reliable measurement and one graph evidence score for each
already-retrieved candidate. Stage B will use these candidate-level scores to
rerank SALAD top-k candidates.

The causal order is:

```text
Stage A: make each DA3 loop measurement reliable
Stage B: use reliable measurements to rerank candidate loops
```

It is not:

```text
candidate reranking first, then retroactively pick supports to justify it
```

## Paper Positioning

Stage A should be presented as a training-free bridge from frozen 3D foundation
models to probabilistic SLAM measurements:

```text
Frozen DA3 predicts relative 3D geometry but not loop-factor covariance.
Support perturbation exposes test-time measurement uncertainty.
This uncertainty becomes a pose-graph noise model and graph-evidence verifier.
```

The method is suitable for a CoRL-style robotics-learning story because it
studies how to use a frozen foundation model inside a robust robot state
estimation system without training a task-specific verifier.
