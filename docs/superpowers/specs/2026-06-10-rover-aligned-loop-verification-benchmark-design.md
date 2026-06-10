# ROVER-Aligned Loop Verification Benchmark Design

**Date:** 2026-06-10
**Status:** Approved design

## 1. Objective

Construct a lightweight loop-closure verification benchmark following the
evaluation organization of ROVER. The benchmark uses a fixed set of causal
DBoW2 retrieval candidates, binary manual labels, and common candidate-level
AP and maximum recall at 100% precision (`MR@100P`) evaluation.

The benchmark ground truth describes whether a query-candidate image pair is a
valid loop closure. It is method-independent. DA3, odometry, pose-graph
optimization, reference poses, distance thresholds, and verifier outputs must
not participate in candidate labeling.

## 2. Benchmark Sequences

The benchmark contains ten sequences.

### 2.1 FusionPortableV2

- `handheld/handheld_escalator00`
- `handheld/handheld_room00`
- `handheld/handheld_room01`
- `ugv/ugv_campus01`
- `ugv/ugv_parking01`

### 2.2 GEODE

- `Offroad/Offroad02_beta`
- `Offroad/Offroad05_beta`

### 2.3 NTU-VIRAL

- `NTU-VIRAL/eee_01`
- `NTU-VIRAL/eee_02`
- `NTU-VIRAL/nya_02`

The benchmark builder reads the existing VPR cache for each sequence,
including `keyframes.jsonl` and referenced image files. Existing
`positives.jsonl` files are not used.

The listed order is the canonical `sequence_order`. A benchmark version is
valid only when all ten configured sequences produce at least one selected
query and ten candidates per selected query. Missing, extra, or empty sequence
groups are fatal rather than silently omitted from the macro average.

## 3. DBoW2 Candidate Generation

### 3.1 Fixed Retrieval Configuration

Use the existing ORB-SLAM3 DBoW2 implementation and vocabulary:

```text
/home/chenguyuan/code/NeurIPS26/ORB_SLAM3/Vocabulary/ORBvoc.txt
```

Freeze the following ORB extraction parameters:

```text
nfeatures=1000
scale_factor=1.2
nlevels=8
ini_fast=20
min_fast=7
```

Each query retrieves the ten highest-scoring legal historical candidates.
Candidate ordering is descending DBoW2 score, with ascending keyframe index as
the deterministic tie-breaker.

The manifest also freezes the DBoW2 execution environment:

- ORB-SLAM3 repository path and Git commit;
- generated helper C++ source SHA256;
- compiled helper binary SHA256;
- resolved DBoW2 shared-library path and SHA256;
- vocabulary path and SHA256.

The helper is rebuilt whenever its generated source differs from the source
used by the existing binary. It emits scores with 17 significant digits so a
rebuild does not silently change ordering through text truncation.

### 3.2 Sequence-Adaptive Recent Exclusion

For sequence `s`, let `N_s` be the number of keyframes with existing images.
Compute:

```math
E_s = \operatorname{clip}\left(
  \operatorname{round}(0.08N_s), 5, 30
\right)
```

Here `round` means deterministic round-half-up for non-negative values:
`floor(0.08N_s + 0.5)`.

A candidate is legal only when:

```math
c < q - E_s
```

This is a single benchmark-wide rule. No sequence-specific manual adjustment
is allowed.

### 3.3 Eligible Queries

A keyframe is an eligible query only when:

- its image exists and produces a valid ORB descriptor;
- at least ten legal historical keyframes also produce valid descriptors;
- DBoW2 returns a complete top-10 candidate list.

The complete eligible-query list is produced before query sampling.

## 4. Query Sampling

Select at most 40 queries per sequence.

For a sequence with at least 40 eligible queries:

1. Sort eligible queries by keyframe index.
2. Divide their ordinal range into 40 equal normalized intervals.
3. Select the eligible query closest to each interval center.
4. Resolve any duplicated rounded selection deterministically by choosing the
   nearest unselected eligible query, preferring the smaller keyframe index.

If a sequence contains fewer than 40 eligible queries, select all of them.

Sampling must not use image content, existing positive labels, trajectory
information, dataset ground truth, retrieval scores, or any verification
method output.

Each selected query contributes exactly ten DBoW2 candidates. The expected
maximum benchmark size is 4,000 image pairs.

## 5. Frozen Pair Manifest

Write one immutable `benchmark_pairs.jsonl` containing:

```json
{
  "pair_id": "FusionPortableV2_handheld_room00_q000080_c000014",
  "dataset": "FusionPortableV2",
  "platform": "handheld",
  "sequence": "handheld_room00",
  "query_idx": 80,
  "candidate_idx": 14,
  "rank": 2,
  "dbow2_score": 0.63,
  "query_image": "images/000080.png",
  "candidate_image": "images/000014.png",
  "query_context": ["images/000079.png", "images/000080.png", "images/000081.png"],
  "candidate_context": ["images/000013.png", "images/000014.png", "images/000015.png"]
}
```

`pair_id` is unique and stable. Candidate generation is complete before
annotation begins. Regenerating candidates creates a new benchmark version
rather than silently changing an existing manifest.

An existing non-empty benchmark version directory is never overwritten. Any
candidate, parameter, cache, helper, or vocabulary change requires a new
benchmark version and output directory.

Each pair additionally freezes cache-relative previous/current/next image
paths for both query and candidate. Context is defined over the ordered list of
keyframes whose images exist: previous and next mean the nearest valid image
before and after the current keyframe, not raw index minus or plus one.
Boundary context is stored as `null`. The benchmark manifest records the
absolute cache root for every sequence so these relative paths remain
resolvable during annotation.

The benchmark manifest records:

- benchmark version;
- selected datasets and sequences;
- DBoW2 and ORB parameters;
- vocabulary path and SHA256;
- ORB-SLAM3 commit, helper source SHA256, and helper binary SHA256;
- resolved DBoW2 shared-library path and SHA256;
- recent-exclusion formula and per-sequence value;
- eligible and selected query counts;
- pair count per sequence;
- canonical ten-entry `sequence_order` and cache roots;
- pair-manifest SHA256;
- annotation display-order seed.

## 6. Binary Manual Annotation

### 6.1 Labels

Every pair receives exactly one explicit label:

- `positive`: query and candidate observe the same place with sufficient
  commonly visible static structure to form a valid loop closure.
- `negative`: the pair does not satisfy the positive definition.

The protocol has no `ambiguous`, `skip`, automatic default, or inferred label.
Unlabeled pairs are excluded from intermediate progress reports, and final
AP/MR evaluation is prohibited until all frozen pairs are labeled.

### 6.2 Annotation Display

For each pair, display:

- query previous, current, and next image;
- candidate previous, current, and next image;
- annotation progress.

The current query and candidate images are visually dominant. Missing boundary
context images are shown as unavailable rather than replaced by another frame.

Hide all information that could bias labels:

- DBoW2 score and retrieval rank;
- keyframe indices;
- trajectory, timestamp, distance, and orientation;
- existing automatic labels;
- DA3 pose, depth, confidence, or Sim3 output;
- PGO, deformation, residual, or verifier output;
- predictions or scores from any evaluated method.

Use a deterministic shuffled display order so candidates from the same query
are not necessarily annotated consecutively.

The annotation API exposes opaque image tokens rather than filenames or paths,
because filenames may reveal hidden keyframe indices.

### 6.3 Controls and Persistence

Provide only:

- `P`: assign `positive`;
- `N`: assign `negative`;
- `Backspace`: undo the previous annotation action.

Each action is immediately persisted as an append-only event. Undo appends a
correction event rather than editing history. On restart, the annotation tool
replays events and resumes at the first unlabeled pair in display order.

The annotation server is single-threaded. Every state response includes an
opaque pair token and state nonce. Label and undo requests must return the
current token and nonce; stale or duplicate requests are rejected, and the UI
disables annotation keys until the active request completes.

Every append-only event has a unique `event_id`. An undo event records the
`target_event_id` of the active label action it reverses. Replay resolves the
latest non-undone label for each pair without relying on adjacency in the
event log.

The finalized `annotations.jsonl` contains one resolved record per pair:

```json
{
  "pair_id": "FusionPortableV2_handheld_room00_q000080_c000014",
  "label": 1,
  "annotated_at": "2026-06-10T12:00:00+08:00",
  "annotation_version": 1
}
```

Before evaluation, validation must confirm:

- every manifest pair has exactly one resolved binary label;
- no annotation references an unknown pair;
- no pair is missing;
- the original pair-manifest hash still matches `manifest.json`.

Finalization writes a separate immutable `annotation_seal.json` containing the
benchmark version, pair-manifest SHA256, annotations SHA256, resolved pair
count, annotation version, and completion timestamp. Evaluation requires this
seal and verifies both hashes. The frozen candidate manifest is not mutated
after annotation. If `annotations.jsonl` or `annotation_seal.json` already
exists, finalization fails. Once sealed, label and undo endpoints are
read-only. Correcting a sealed dataset requires a new annotation version and
new output directory; sealed files are never rewritten in place.

## 7. Common Evaluation Protocol

Every evaluated method consumes the same frozen pair manifest and outputs one
row per pair:

```json
{"pair_id": "...", "score": 0.63, "status": "ok"}
```

Larger scores always mean greater loop-closure confidence. Methods with a
smaller-is-better native quantity negate that quantity before evaluation.

The main comparison may include:

- DBoW2;
- NetVLAD;
- SALAD;
- local-feature geometric verification;
- ROVER-like trajectory deformation;
- LoopAnything.

Method-specific internal inputs are allowed, but they do not affect benchmark
labels or candidate selection. In particular, DA3 is an internal component of
LoopAnything and is absent from benchmark construction.

Score adapters must consume `benchmark_pairs.jsonl` directly. They may cache
per-image descriptors or geometry, but they must score the listed query and
candidate rather than running their own retrieval, and they must not read
`positives.jsonl` or `annotations.jsonl`. Labels are loaded only by the common
evaluator after all method scores have been written.

Each score file has a sidecar manifest recording the candidate-manifest hash,
method configuration/model hashes, source commit, command, and score-file
hash. These provenance fields do not affect metric computation.

All methods must return one row for every frozen pair. A successful row has a
finite score. A processing failure is represented explicitly as
`{"score": null, "status": "failed"}` and remains in the evaluation set.
During evaluation, every failed row of one method receives the same tied score
strictly below that method's minimum finite score. If all rows fail, all receive
the same score. Methods may not remove failed or unfavorable samples, and
failed samples may not receive distinct fallback scores that induce an
arbitrary ranking.

For each sequence independently, compute:

- average precision (`AP`);
- maximum recall at 100% precision (`MR@100P`).

Thresholds are swept independently over each method's native score. No score
normalization across methods is required because AP and MR depend on ranking.
Tied scores are evaluated as one threshold group.

Report the arithmetic macro average of the ten per-sequence AP values and the
ten per-sequence MR values. Do not pool candidates across sequences before
computing the paper-facing average.

## 8. Paper Table

Table 1 follows the organization of ROVER Table I:

- rows group retrieval, geometric-verification, trajectory-verification, and
  proposed methods;
- each sequence contributes an `AP / MR` column pair;
- the final column reports the ten-sequence macro average;
- best and second-best values are highlighted independently for AP and MR,
  with deterministic tie handling;
- candidate and positive counts are reported in the caption or accompanying
  text.

The table evaluates loop-closure verification on fixed DBoW2 candidates. It
does not claim that all compared methods use the same internal sensing or
relative-pose estimator.

## 9. Output Layout

Use an ignored workspace directory:

```text
LoopAnything/workspace/rover_aligned_benchmark/
  benchmark_v1/
    manifest.json
    benchmark_pairs.jsonl
    annotation_events.jsonl
    annotations.jsonl
    annotation_seal.json
    annotation_progress.json
    scores/
      dbow2.jsonl
      dbow2.manifest.json
      netvlad.jsonl
      netvlad.manifest.json
      salad.jsonl
      salad.manifest.json
      geometric_verification.jsonl
      geometric_verification.manifest.json
      rover_like.jsonl
      rover_like.manifest.json
      loopanything.jsonl
      loopanything.manifest.json
    metrics/
      metrics_per_sequence.csv
      metrics_summary.json
      table1.md
```

Generated datasets, images, annotations, and method scores remain outside git.
Only source code, configuration, and documentation are committed.

## 10. Acceptance Criteria

The benchmark is ready for paper-facing evaluation when:

1. all ten sequence caches and image paths validate;
2. candidate generation is deterministic across repeated runs;
3. every selected query has exactly ten legal candidates;
4. exactly the canonical ten non-empty sequence groups are present;
5. the manifest records all parameters, cache roots, and executable hashes;
6. the benchmark builder refuses to overwrite a non-empty version directory;
7. every frozen pair has one explicit binary annotation;
8. `annotation_seal.json` validates the frozen pairs and resolved annotations;
9. the annotation UI hides all prohibited information and rejects stale input;
10. score validation detects missing, duplicate, unknown, invalid, and failed rows;
11. all failures for a method form one tied worst-score group;
12. AP and MR are computed per sequence and macro-averaged in canonical order;
13. the table reports candidate/positive counts and highlights best/second-best;
14. a rerun from the frozen manifest reproduces the same evaluation table.
