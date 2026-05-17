# DA3+ROVER++ Research Roadmap

Status: roadmap only. This is not a spec or implementation plan.

## Immediate Experiment

Run residual-aware score sweeps over existing candidate records before changing
the verifier architecture. The first objective is to measure the empirical upper
bound of:

- `pgo_error_after`
- `trajectory_deformation_rmse`
- SALAD score
- residual/deformation/SALAD fusion

The current hypothesis is that graph residual is a primary false-loop signal.
The baseline deformation-only ROVER-like score is insufficient because GTSAM can
leave false loops as high-residual constraints without strongly deforming the
optimized trajectory.

## Candidate Innovation Directions

1. Residual-aware trajectory-prior verifier.

Use both graph residual and trajectory deformation as the interpretable loop
factor response. This is the first candidate for the main DA3+ROVER++ method if
the sweep consistently beats SALAD and deformation-only ROVER-like scoring.

2. PGO response curve.

Run the same loop factor under multiple loop noise settings and evaluate how
`pgo_error_after` and deformation change as loop strength increases. True loops
should remain explainable; false loops should either keep high residuals or
force unstable trajectory deformation.

3. DA3 multi-support factor consensus.

Generate multiple DA3-derived loop factors for the same query-candidate pair
using several candidate-neighborhood supports. True loops should produce
consistent factors and low graph residuals; false loops should be unstable
across supports or produce high residuals after PGO.

## Decision Rule

Do not lock the next method design until residual-aware score sweep results are
available on FusionPortableV2. If the simple residual/deformation/SALAD fusion
already gives strong AP and MR gains, use it as the next baseline and add PGO
response curves or multi-support consensus only where the failure analysis shows
remaining false positives.
