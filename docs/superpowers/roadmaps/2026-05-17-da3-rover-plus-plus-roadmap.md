# DA3+ROVER++ Research Roadmap

Status: roadmap only. This is not an implementation plan.

Update on 2026-05-18: the support-ensemble graph-evidence direction produced a
mixed result. It did not beat deformation-only ROVER-style scoring on average
AP over the five available FusionPortableV2 handheld sequences, but it strongly
improved `handheld_escalator00` and slightly improved average MR@100P. It is
therefore retained as a valuable ablation and possible diagnostic component.
The active research direction is now
self-calibrated counterfactual candidate verification:

```text
LoopAnything/docs/superpowers/specs/2026-05-18-self-calibrated-counterfactual-verifier-design.md
```

The Stage 1 offline DA3-ROVER baseline is converged. Residual-aware sweeps over
FusionPortableV2 handheld sequences showed that graph response is the useful
signal:

```text
SALAD only: lower AP/MR than graph-aware scoring
ROVER deformation-only: useful but misses high-residual false loops
raw graph deformation + residual: strong gain without SALAD fusion
z_fusion: best diagnostic score but relies on manual weights
```

The next research phase should therefore not use `z_fusion` as the main method.
It should turn DA3 predictions into loop measurements and verify them with
counterfactual graph response plus test-time self-calibration.

## Guiding Constraints

- The method remains training-free: no neural verifier, no learned loop-policy
  prior, and no supervised probability head.
- Dataset/platform-level measurement-model calibration is allowed, following
  normal SLAM configuration practice.
- SALAD is a proposal generator. Its score is a baseline and diagnostic, not the
  main DA3-ROVER++ verifier score.
- DA3 does not output pose uncertainty. Any uncertainty or confidence proxy must
  be derived at test time from support perturbation, measurement consistency, or
  graph response.
- DA3 forwards must be isolated by geometric group. Independent candidates or
  independent triplets must not be put into the same DA3 batch, because DA3's
  transformer performs cross-view attention across images in a forward pass.

## Stage A: Support-Level Verification

Spec:

```text
LoopAnything/docs/superpowers/specs/2026-05-18-da3-rover-plus-plus-stage-a-support-uncertainty-design.md
```

Goal: make a single DA3 loop measurement reliable before candidate reranking.

Status: mixed result for the current graph-evidence score. The support-ensemble
implementation can stabilize loop factors and provide useful diagnostics,
especially on `handheld_escalator00`, but its graph-evidence score is not robust
enough to be the sole DA3-ROVER++ method.

For a fixed retrieval pair `(query, candidate)`:

```text
select support_count=4 candidate-neighborhood supports
run isolated DA3 triplets [query, candidate, support_i]
derive multiple loop factors T_qc_i
estimate robust mean T_qc_mean
estimate diagonal Sigma_da3 from support perturbation
use Sigma_da3 as the PGO loop noise
score by graph evidence NLL
```

Stage A's main contribution is:

```text
Support-Ensemble DA3 Test-Time Uncertainty
```

The paper story is that frozen DA3 predicts relative 3D geometry but no
loop-factor covariance. Support perturbation exposes test-time measurement
uncertainty, which can be converted into a probabilistic SLAM loop factor.

Required Stage A ablations:

```text
A0: single nearest support + fixed loop noise
A1: multi-support uniform mean + fixed loop noise
A2: multi-support robust mean + fixed loop noise
A3: robust mean + Sigma_da3 covariance-aware PGO
A4: A3 + logdet uncertainty penalty
```

Original Stage A success criterion:

```text
multi-support uncertainty improves AP/MR over single-support and
deformation-only ROVER-style scoring, while uncertainty diagnostics correlate
with loop-factor reliability.
```

Current result: partially satisfied. Support-ensemble graph evidence improves
`handheld_escalator00` substantially and improves five-sequence average MR@100P
slightly, but loses average AP to deformation-only scoring.

## Self-Calibrated Counterfactual Candidate Verification

Goal: rerank SALAD top-k candidates using DA3 loop factors and counterfactual
PGO response, without hand-tuned fusion weights.

Default starting point:

```text
retrieval_top_k = 10
support_count = 1 initially, with multi-support as an ablation
DA3 forwards per query = top_k * support_count isolated triplets
```

For each query:

```text
SALAD proposes top-k candidates
DA3 estimates one loop measurement per candidate
full-prefix counterfactual PGO measures residual and deformation response
rank/percentile evidence calibrates SALAD, residual, and deformation signals
candidate list is reranked without manual fusion weights
```

This direction keeps the ROVER-like counterfactual optimization idea, but adds
residual-aware graph response and test-time self-calibration. The main metric
remains candidate-level AP and MR@100P, with additional query-level top-1/top-k
recall diagnostics.

## Later Directions

These are deferred until self-calibrated counterfactual verification results are
clear:

- PGO response curves over multiple loop-noise strengths.
- Same-candidate multi-support DA3 group inference as a speed ablation.
- Online AsterSLAM integration with a lower-cost configuration such as
  `top_k=4, support_count=2`.
- KITTI, GEODE, and private dog dataset expansion after FusionPortableV2 is
  stable.

## Decision Rule

Do not promote a method to the paper main table only because it wins a post-hoc
score sweep. The main method should have a defensible measurement-model story:

```text
DA3 image-based loop factor
-> counterfactual PGO residual/deformation response
-> self-calibrated candidate evidence
-> loop decision
```

`z_fusion` remains a diagnostic upper-bound and ablation, not the main
DA3-ROVER++ contribution.
