# ROVER-Aligned Loop Verification Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight ROVER-aligned benchmark that freezes causal DBoW2 top-10 image pairs, supports complete binary manual annotation, and evaluates arbitrary method scores with per-sequence AP/MR@100P and macro averages.

**Architecture:** Reuse the existing ORB-SLAM3 DBoW2 backend in `baseline_scripts`, while adding focused benchmark modules under `src/robust_loop_verifier`. One builder freezes the ten-sequence pair manifest, one local HTTP annotation tool persists binary labels, and one evaluator validates complete score files and writes paper-facing metrics. Benchmark construction never reads existing positive labels, DA3 outputs, trajectories, or verifier scores.

**Tech Stack:** Python 3.9+, standard-library HTTP server, NumPy, existing ORB-SLAM3 DBoW2 C++ helper, existing `robust_loop_verifier` JSONL and metrics utilities, pytest.

**Git Policy:** Do not commit during execution unless the user explicitly requests it. End each task with `git status --short`.

---

## File Structure

- Create `LoopAnything/baseline_scripts/__init__.py`
  - Makes the existing DBoW2 implementation importable by the benchmark builder.
- Modify `LoopAnything/baseline_scripts/orb_dbow2_retrieval_pipeline.py`
  - Rebuilds the helper when generated source changes, emits full-precision scores, and
    exposes helper fingerprints to the benchmark manifest.
- Create `LoopAnything/src/robust_loop_verifier/rover_benchmark.py`
  - Sequence configuration, adaptive recent exclusion, deterministic query sampling, pair records, SHA256 manifests, and benchmark validation.
- Create `LoopAnything/src/robust_loop_verifier/rover_annotation.py`
  - Append-only annotation events, undo replay, finalized labels, annotation seal, frozen
    image-context lookup, and annotation progress.
- Create `LoopAnything/src/robust_loop_verifier/rover_evaluation.py`
  - Annotation-seal/score validation, tied failure handling, per-sequence AP/MR, macro
    averages, and table generation.
- Create `LoopAnything/src/robust_loop_verifier/rover_pair_scoring.py`
  - Scores the frozen query-candidate pairs with descriptor and verifier backends without
    rerunning retrieval or reading labels.
- Modify `LoopAnything/src/robust_loop_verifier/pipeline.py`
  - Exposes candidate scoring for caller-provided frozen pairs while preserving the existing
    sequence-evaluation entrypoint.
- Modify `LoopAnything/src/robust_loop_verifier/score_sweep.py`
  - Exposes label-free named score computation from frozen candidate records.
- Modify `LoopAnything/src/robust_loop_verifier/metrics.py`
  - Make AP tie-safe by evaluating equal scores as one threshold group.
- Create `LoopAnything/configs/robust_loop_verifier/rover_aligned_benchmark.yaml`
  - Frozen ten-sequence cache list and benchmark parameters.
- Create `LoopAnything/robust_loop_verification_scripts/build_rover_aligned_benchmark.py`
  - CLI for DBoW2 candidate generation and immutable manifest creation.
- Create `LoopAnything/robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py`
  - Local annotation HTTP server with embedded HTML/CSS/JavaScript.
- Create `LoopAnything/robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py`
  - Generic `pair_id, score` evaluator and Table 1 Markdown exporter.
- Create `LoopAnything/robust_loop_verification_scripts/score_rover_aligned_benchmark.py`
  - CLI adapters for NetVLAD, SALAD, ROVER-like, and LoopAnything scores on frozen pairs.
- Create `LoopAnything/tests/robust_loop_verifier/test_rover_benchmark.py`
- Create `LoopAnything/tests/robust_loop_verifier/test_rover_annotation.py`
- Create `LoopAnything/tests/robust_loop_verifier/test_rover_evaluation.py`
- Create `LoopAnything/tests/robust_loop_verifier/test_rover_pair_scoring.py`
- Modify `LoopAnything/tests/robust_loop_verifier/test_metrics.py`
- Modify `LoopAnything/tests/baseline_scripts/test_orb_dbow2_retrieval_pipeline.py`

---

### Task 1: Make AP Tie-Safe

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/metrics.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_metrics.py`

- [ ] **Step 1: Add a failing AP tie test**

Append:

```python
def test_average_precision_treats_equal_scores_as_one_threshold_group():
    labels_a = [True, False, True]
    labels_b = [False, True, True]
    scores = [0.5, 0.5, 0.1]

    expected = 7.0 / 12.0
    assert math.isclose(average_precision(labels_a, scores), expected)
    assert math.isclose(average_precision(labels_b, scores), expected)
```

The expected value is:

```text
precision at score 0.5 = 1 / 2
recall increment = 1 / 2
precision at score 0.1 = 2 / 3
recall increment = 1 / 2
AP = (1/2)(1/2) + (2/3)(1/2) = 7/12
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_metrics.py::test_average_precision_treats_equal_scores_as_one_threshold_group \
  -q
```

Expected: FAIL because current AP breaks ties by input order.

- [ ] **Step 3: Replace rank-by-rank AP with threshold-group AP**

Replace `average_precision()` with:

```python
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
```

Remove or retain `_stable_score_order()` only according to its remaining uses;
do not change `max_recall_at_100_precision()`.

- [ ] **Step 4: Run the complete metrics tests**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_metrics.py -q
```

Expected: all tests pass. Update the old
`test_equal_scores_preserve_input_order_for_average_precision` expectation to
the tie-safe value rather than deleting tie coverage.

- [ ] **Step 5: Record the checkpoint**

Run:

```bash
git status --short
```

Expected: only the metrics source/test changes plus pre-existing documentation changes.

---

### Task 2: Freeze DBoW2 Candidates and Benchmark Manifest

**Files:**
- Create: `LoopAnything/baseline_scripts/__init__.py`
- Modify: `LoopAnything/baseline_scripts/orb_dbow2_retrieval_pipeline.py`
- Create: `LoopAnything/src/robust_loop_verifier/rover_benchmark.py`
- Create: `LoopAnything/configs/robust_loop_verifier/rover_aligned_benchmark.yaml`
- Create: `LoopAnything/robust_loop_verification_scripts/build_rover_aligned_benchmark.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_rover_benchmark.py`
- Modify: `LoopAnything/tests/baseline_scripts/test_orb_dbow2_retrieval_pipeline.py`

- [ ] **Step 1: Write failing tests for the frozen sampling rules**

Create `tests/robust_loop_verifier/test_rover_benchmark.py` with:

```python
from robust_loop_verifier.rover_benchmark import (
    BenchmarkPair,
    compute_recent_exclusion,
    sample_eligible_queries,
    validate_pair_manifest,
)


def test_recent_exclusion_uses_single_clipped_sequence_rule():
    assert compute_recent_exclusion(50) == 5
    assert compute_recent_exclusion(116) == 9
    assert compute_recent_exclusion(184) == 15
    assert compute_recent_exclusion(1000) == 30


def test_query_sampling_is_deterministic_and_covers_ordinal_range():
    eligible = list(range(10, 110))
    selected = sample_eligible_queries(eligible, limit=40)

    assert selected == [
        11, 13, 16, 18, 21, 23, 26, 28, 31, 33,
        36, 38, 41, 43, 46, 48, 51, 53, 56, 58,
        61, 63, 66, 68, 71, 73, 76, 78, 81, 83,
        86, 88, 91, 93, 96, 98, 101, 103, 106, 108,
    ]


def test_query_sampling_prefers_smaller_index_on_exact_tie():
    eligible = [10, 20, 30, 40, 50, 60, 70, 80]
    assert sample_eligible_queries(eligible, limit=4) == [10, 30, 50, 70]


def test_query_sampling_keeps_all_when_fewer_than_limit():
    assert sample_eligible_queries([7, 9, 12], limit=40) == [7, 9, 12]


def test_manifest_requires_exactly_ten_legal_candidates_per_query():
    pairs = [
        BenchmarkPair(
            pair_id=f"d_p_s_q000020_c{candidate:06d}",
            dataset="d",
            platform="p",
            sequence="s",
            query_idx=20,
            candidate_idx=candidate,
            rank=rank,
            dbow2_score=1.0 / rank,
            query_image="images/000020.png",
            candidate_image=f"images/{candidate:06d}.png",
            query_context=("images/000019.png", "images/000020.png", "images/000021.png"),
            candidate_context=(None, f"images/{candidate:06d}.png", None),
        )
        for rank, candidate in enumerate(range(10), start=1)
    ]

    validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})
```

Add failure cases for duplicate `pair_id`, duplicate rank, non-causal
candidate, illegal recent candidate, missing rank, and 9/11 candidates.

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_rover_benchmark.py -q
```

Expected: FAIL because `rover_benchmark` does not exist.

- [ ] **Step 3: Implement deterministic benchmark primitives**

Create `src/robust_loop_verifier/rover_benchmark.py` with these public APIs:

```python
@dataclass(frozen=True)
class BenchmarkSequence:
    dataset: str
    platform: str
    sequence: str
    cache: Path


@dataclass(frozen=True)
class BenchmarkPair:
    pair_id: str
    dataset: str
    platform: str
    sequence: str
    query_idx: int
    candidate_idx: int
    rank: int
    dbow2_score: float
    query_image: str
    candidate_image: str
    query_context: tuple[str | None, str, str | None]
    candidate_context: tuple[str | None, str, str | None]


def compute_recent_exclusion(keyframe_count: int) -> int:
    if keyframe_count <= 0:
        raise ValueError("keyframe_count must be positive")
    return min(30, max(5, int(math.floor(0.08 * keyframe_count + 0.5))))


def sample_eligible_queries(indices: Sequence[int], limit: int = 40) -> list[int]:
    ordered = sorted(set(int(index) for index in indices))
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(ordered) <= limit:
        return ordered
    positions = [
        (bin_index + 0.5) * len(ordered) / limit - 0.5
        for bin_index in range(limit)
    ]
    selected_positions = []
    used = set()
    for position in positions:
        candidates = sorted(
            range(len(ordered)),
            key=lambda idx: (abs(idx - position), ordered[idx]),
        )
        chosen = next(idx for idx in candidates if idx not in used)
        used.add(chosen)
        selected_positions.append(chosen)
    return sorted(ordered[idx] for idx in selected_positions)
```

Also implement:

```python
def read_sequence_keyframes(sequence: BenchmarkSequence) -> list[KeyframeImage]
def build_image_context(keyframes: Sequence[KeyframeImage], keyframe_idx: int) -> tuple[str | None, str, str | None]
def build_pairs_for_sequence(sequence, retrieval_records, selected_queries, recent_exclusion)
def validate_pair_manifest(pairs, recent_exclusion_by_sequence) -> None
def sha256_file(path: Path) -> str
def write_frozen_benchmark(output_root, pairs, manifest) -> None
```

`read_sequence_keyframes()` must ignore `positives.jsonl`, reject duplicate
indices, and retain only existing images. `build_image_context()` uses adjacent
entries in that valid-image list, not numeric index plus or minus one. The
manifest stores each sequence's absolute cache root; pair rows store only
cache-relative paths.

- [ ] **Step 4: Add the frozen ten-sequence config**

Create `configs/robust_loop_verifier/rover_aligned_benchmark.yaml`:

```yaml
benchmark_version: benchmark_v1
query_limit_per_sequence: 40
retrieval_top_k: 10
annotation_shuffle_seed: 20260610
orb_slam3_root: /home/chenguyuan/code/NeurIPS26/ORB_SLAM3
helper_build_dir: workspace/rover_aligned_benchmark/helper_build
netvlad_root: /home/chenguyuan/code/NeurIPS26/netvlad_image_retrieval
verifier_configs:
  FusionPortableV2: configs/robust_loop_verifier/fusionportablev2_handheld.yaml
  GEODE: configs/robust_loop_verifier/geode_offroad.yaml
  NTU-VIRAL: configs/robust_loop_verifier/ntu_viral.yaml
orb:
  nfeatures: 1000
  scale_factor: 1.2
  nlevels: 8
  ini_fast: 20
  min_fast: 7
vocabulary: /home/chenguyuan/code/NeurIPS26/ORB_SLAM3/Vocabulary/ORBvoc.txt
sequences:
  - dataset: FusionPortableV2
    platform: handheld
    sequence: handheld_escalator00
    cache: /data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_escalator00
  - dataset: FusionPortableV2
    platform: handheld
    sequence: handheld_room00
    cache: /data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_room00
  - dataset: FusionPortableV2
    platform: handheld
    sequence: handheld_room01
    cache: /data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_room01
  - dataset: FusionPortableV2
    platform: ugv
    sequence: ugv_campus01
    cache: /data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/ugv/ugv_campus01
  - dataset: FusionPortableV2
    platform: ugv
    sequence: ugv_parking01
    cache: /data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/ugv/ugv_parking01
  - dataset: GEODE
    platform: Offroad
    sequence: Offroad02_beta
    cache: /data/datasets/GEODE/robust_loop_verifier_cache/GEODE/Offroad/Offroad02_beta
  - dataset: GEODE
    platform: Offroad
    sequence: Offroad05_beta
    cache: /data/datasets/GEODE/robust_loop_verifier_cache/GEODE/Offroad/Offroad05_beta
  - dataset: NTU-VIRAL
    platform: NTU-VIRAL
    sequence: eee_01
    cache: /data/datasets/NTU-VIRAL/robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/eee_01
  - dataset: NTU-VIRAL
    platform: NTU-VIRAL
    sequence: eee_02
    cache: /data/datasets/NTU-VIRAL/robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/eee_02
  - dataset: NTU-VIRAL
    platform: NTU-VIRAL
    sequence: nya_02
    cache: /data/datasets/NTU-VIRAL/robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/nya_02
```

- [ ] **Step 5: Implement the builder CLI**

Create `baseline_scripts/__init__.py`, then create
`robust_loop_verification_scripts/build_rover_aligned_benchmark.py`.

The script must:

1. load the YAML config;
2. construct `OrbDbow2RetrievalBackend` from
   `baseline_scripts.orb_dbow2_retrieval_pipeline`;
3. read all valid images for each sequence;
4. compute `E_s`;
5. retrieve DBoW2 top-10 for all keyframes;
6. retain queries with exactly ten candidates;
7. call `sample_eligible_queries(..., limit=40)`;
8. write only selected-query pairs;
9. validate the pair manifest;
10. write `benchmark_pairs.jsonl`, `scores/dbow2.jsonl`,
    `scores/dbow2.manifest.json`, and `manifest.json`;
11. record the canonical ten-entry sequence order, cache roots, vocabulary
    hash, ORB-SLAM3 Git commit, helper source hash, helper binary hash, and the
    resolved DBoW2 shared-library path/hash;
12. refuse to write when the output directory already contains any file.

Use this CLI:

```python
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--rebuild-dbow2-helper", action="store_true")
```

The builder validates exactly ten configured sequences and rejects any
sequence with no selected query. It generates the helper source in
`helper_build_dir`, rebuilds when the source hash changes or the binary is
missing, adds `<iomanip>`, and emits each DBoW2 score with
`std::setprecision(17)`. `--rebuild-dbow2-helper` forces the same deterministic
rebuild path; it does not permit replacing an existing benchmark version.
After compilation, resolve `libDBoW2.so` from `ldd <helper_binary>`, require the
resolved path to exist, and record its absolute path and SHA256.

The DBoW2 score file rows are:

```json
{"pair_id": "...", "score": 0.63, "status": "ok"}
```

Add tests that reject a nine- or eleven-sequence config, an empty sequence
group, an existing non-empty output root, and a stale helper binary whose
recorded source hash differs. Verify the manifest contains exactly ten cache
roots and the vocabulary, helper source, helper binary, DBoW2 shared library,
and ORB-SLAM3 commit fingerprints. In
`tests/baseline_scripts/test_orb_dbow2_retrieval_pipeline.py`, add a regression
test that creates an existing binary, changes generated helper source, and
asserts `_compile_helper()` is called. Also assert `_helper_source()` contains
`#include <iomanip>` and `std::setprecision(17)`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=src:. pytest \
  tests/robust_loop_verifier/test_rover_benchmark.py \
  tests/baseline_scripts/test_orb_dbow2_retrieval_pipeline.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 7: Record the checkpoint**

Run:

```bash
git status --short
```

---

### Task 3: Implement Binary Annotation Service

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/rover_annotation.py`
- Create: `LoopAnything/robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_rover_annotation.py`

- [ ] **Step 1: Write failing event-replay tests**

Create `tests/robust_loop_verifier/test_rover_annotation.py`:

```python
from robust_loop_verifier.rover_annotation import (
    AnnotationEvent,
    append_event,
    finalize_annotations,
    replay_events,
)


def test_replay_supports_positive_negative_and_undo(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))
    append_event(events, AnnotationEvent("e3", "undo", "pair-b", None, "e2", "t3"))

    assert replay_events(events) == {"pair-a": 1}


def test_finalize_requires_complete_binary_labels(tmp_path):
    pairs = [{"pair_id": "pair-a"}, {"pair_id": "pair-b"}]
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(ValueError, match="unlabeled"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=tmp_path / "annotations.jsonl",
            manifest_path=tmp_path / "manifest.json",
            seal_path=tmp_path / "annotation_seal.json",
        )
```

Add tests for invalid labels, unknown pair IDs, duplicate manifest pair IDs,
duplicate event IDs, restart replay, undo targeting an
unknown/already-undone event, seal hash verification, refusal to finalize or
append after sealing, valid-image context at sequence boundaries, opaque image
tokens, and stale state nonce rejection.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_rover_annotation.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement append-only annotation state**

Create `src/robust_loop_verifier/rover_annotation.py` with:

```python
@dataclass(frozen=True)
class AnnotationEvent:
    event_id: str
    action: Literal["label", "undo"]
    pair_id: str
    label: int | None
    target_event_id: str | None
    annotated_at: str


def append_event(path: Path, event: AnnotationEvent) -> None:
    if event.action == "label" and event.label not in (0, 1):
        raise ValueError("label events require label 0 or 1")
    if event.action == "label" and event.target_event_id is not None:
        raise ValueError("label events cannot target another event")
    if event.action == "undo" and (
        event.label is not None or event.target_event_id is None
    ):
        raise ValueError("undo events require label=None and target_event_id")
    existing_ids = {
        str(row["event_id"]) for row in read_jsonl(path)
    } if path.exists() else set()
    if event.event_id in existing_ids:
        raise ValueError(f"duplicate event_id: {event.event_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replay_events(path: Path) -> dict[str, int]:
    labels_by_event = {}
    undone = set()
    seen_event_ids = set()
    for row in read_jsonl(path) if path.exists() else []:
        if row["event_id"] in seen_event_ids:
            raise ValueError(f"duplicate event_id: {row['event_id']}")
        seen_event_ids.add(row["event_id"])
        if row["action"] == "label":
            labels_by_event[row["event_id"]] = row
        elif row["action"] == "undo":
            target = row["target_event_id"]
            if target not in labels_by_event or target in undone:
                raise ValueError(f"invalid undo target: {target}")
            if row["pair_id"] != labels_by_event[target]["pair_id"]:
                raise ValueError("undo pair_id does not match target event")
            undone.add(target)
    resolved = {}
    for event_id, row in labels_by_event.items():
        if event_id not in undone:
            resolved[row["pair_id"]] = int(row["label"])
    return resolved
```

Also implement:

```python
def deterministic_display_order(pair_ids, seed) -> list[str]
def opaque_image_tokens(pair, token_secret) -> dict[str, str | None]
def annotation_progress(pair_ids, resolved) -> dict[str, int]
def replay_resolved_label_events(path) -> dict[str, AnnotationEvent]
def finalize_annotations(pairs, event_path, output_path, manifest_path, seal_path) -> None
def verify_annotation_seal(manifest_path, pairs_path, annotations_path, seal_path) -> None
```

`finalize_annotations()` must require 100% coverage and write one resolved row
per manifest pair in manifest order, preserving `annotated_at` from the active
label event. It then writes `annotation_seal.json`
containing the pair-manifest hash, annotation hash, count, version, and
completion timestamp. It never edits `manifest.json`. It fails before writing
if `annotations.jsonl` or `annotation_seal.json` already exists. Once the seal
exists, the server returns `409 Conflict` for label, undo, and finalize
requests; corrections require a new annotation-version output directory.

- [ ] **Step 4: Implement the local annotation server**

Create `robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py`
using the single-threaded `HTTPServer`.

Required endpoints:

```text
GET  /                  embedded annotation page
GET  /api/state         pair_token, state_nonce, six opaque image URLs, progress
GET  /image/<token>     validated image file
POST /api/label         {"pair_token": "...", "state_nonce": "...", "label": 0|1}
POST /api/undo          {"pair_token": "...", "state_nonce": "..."}
POST /api/finalize      write annotations.jsonl and annotation_seal.json at 100%
```

The page must bind:

```javascript
if (event.key.toLowerCase() === "p") submitLabel(1);
if (event.key.toLowerCase() === "n") submitLabel(0);
if (event.key === "Backspace") undoLast();
```

Display the current query/candidate prominently and adjacent frames smaller.
Do not send rank, score, indices, trajectory, DA3, or verifier fields in
`/api/state`. Image URLs use deterministic opaque tokens rather than cache
filenames. Disable all keyboard actions while a request is active. Reject a
label or undo if its token/nonce does not match the current server state, then
reload `/api/state` rather than appending an event.

CLI:

```python
parser.add_argument("benchmark_root", type=Path)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--open", action="store_true")
```

- [ ] **Step 5: Run annotation tests and syntax validation**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_rover_annotation.py -q
python -m py_compile \
  src/robust_loop_verifier/rover_annotation.py \
  robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py
```

Expected: tests pass and compilation exits 0.

- [ ] **Step 6: Record the checkpoint**

Run:

```bash
git status --short
```

---

### Task 4: Implement Common Score Evaluation and Table Export

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/rover_evaluation.py`
- Create: `LoopAnything/robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_rover_evaluation.py`

- [ ] **Step 1: Write failing score-contract tests**

Create `tests/robust_loop_verifier/test_rover_evaluation.py`:

```python
from robust_loop_verifier.rover_evaluation import evaluate_score_file, validate_scores


def test_score_validation_requires_exact_pair_coverage():
    pairs = [{"pair_id": "a"}, {"pair_id": "b"}]

    with pytest.raises(ValueError, match="missing"):
        validate_scores(pairs, [{"pair_id": "a", "score": 0.9, "status": "ok"}])
    with pytest.raises(ValueError, match="unknown"):
        validate_scores(
            pairs,
            [
                {"pair_id": "a", "score": 0.9, "status": "ok"},
                {"pair_id": "b", "score": 0.8, "status": "ok"},
                {"pair_id": "x", "score": 0.1, "status": "ok"},
            ],
        )


def test_failed_rows_share_one_tied_worst_score():
    pairs = [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "c"}]
    rows = [
        {"pair_id": "a", "score": 0.5, "status": "ok"},
        {"pair_id": "b", "score": None, "status": "failed"},
        {"pair_id": "c", "score": None, "status": "failed"},
    ]

    scores = validate_scores(pairs, rows)

    assert scores["b"] == scores["c"]
    assert scores["b"] < scores["a"]


def test_evaluation_is_per_sequence_then_macro_averaged(tmp_path):
    pairs = [
        {"pair_id": "a", "dataset": "d", "platform": "p", "sequence": "s1"},
        {"pair_id": "b", "dataset": "d", "platform": "p", "sequence": "s1"},
        {"pair_id": "c", "dataset": "d", "platform": "p", "sequence": "s2"},
        {"pair_id": "d", "dataset": "d", "platform": "p", "sequence": "s2"},
    ]
    labels = {"a": 1, "b": 0, "c": 0, "d": 1}
    scores = {"a": 0.9, "b": 0.1, "c": 0.9, "d": 0.1}

    result = evaluate_score_file(
        "method",
        pairs,
        labels,
        scores,
        sequence_order=["d/p/s1", "d/p/s2"],
    )

    assert result["sequences"]["d/p/s1"]["AP"] == 1.0
    assert result["sequences"]["d/p/s2"]["AP"] == 0.5
    assert result["macro_average"]["AP"] == 0.75
```

Add tests for duplicate score rows, invalid `status`, null successful scores,
finite failed scores, incomplete annotations, a broken annotation seal,
missing or mismatched method score manifests,
zero-positive sequences, tied scores, missing/extra/empty sequence groups,
canonical method/sequence ordering, candidate and positive counts, and
best/second-best highlighting independently for AP and MR.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_rover_evaluation.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement strict score validation and macro evaluation**

Create `src/robust_loop_verifier/rover_evaluation.py` with:

```python
def validate_scores(
    pairs: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    expected = {str(pair["pair_id"]) for pair in pairs}
    scores = {}
    for row in score_rows:
        pair_id = str(row["pair_id"])
        if pair_id in scores:
            raise ValueError(f"duplicate score for pair_id={pair_id}")
        status = str(row.get("status", "ok"))
        if status == "ok":
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite score for pair_id={pair_id}")
            scores[pair_id] = score
        elif status == "failed" and row.get("score") is None:
            scores[pair_id] = None
        else:
            raise ValueError(f"invalid score row for pair_id={pair_id}")
    missing = sorted(expected - scores.keys())
    unknown = sorted(scores.keys() - expected)
    if missing:
        raise ValueError(f"missing scores: {missing[:5]}")
    if unknown:
        raise ValueError(f"unknown scores: {unknown[:5]}")
    finite = [score for score in scores.values() if score is not None]
    if not finite:
        return {pair_id: 0.0 for pair_id in scores}
    if any(score is None for score in scores.values()):
        scores = {
            pair_id: None if score is None else math.atan(score)
            for pair_id, score in scores.items()
        }
        worst = -math.pi / 2.0
    else:
        worst = 0.0
    return {
        pair_id: worst if score is None else score
        for pair_id, score in scores.items()
    }
```

`atan` is applied only when failures exist. It is strictly monotonic, preserves
all successful rankings and ties, and guarantees a finite common failure score
below every transformed successful score, including the minimum-float edge
case.

Implement:

```python
def read_complete_annotations(pairs, manifest_path, annotations_path, seal_path) -> dict[str, int]
def validate_sequence_groups(pairs, sequence_order) -> None
def evaluate_score_file(method_name, pairs, labels, scores, sequence_order) -> dict[str, object]
def evaluate_methods(benchmark_root, method_files) -> dict[str, object]
def verify_score_manifest(pair_manifest_path, score_path, score_manifest_path) -> None
def write_metrics_outputs(output_dir, results) -> None
def render_table1_markdown(results, sequence_order, method_order) -> str
```

Sequence keys are `dataset/platform/sequence`. Compute AP and MR independently
per sequence using `robust_loop_verifier.metrics`, then average the ten
sequence values arithmetically. `evaluate_methods()` reads `sequence_order`
from `manifest.json`, requires exactly ten non-empty groups with no extras, and
verifies `annotation_seal.json` before loading any method score. The Markdown
table reports candidate/positive counts and marks best and second-best AP/MR
with deterministic tie handling.

- [ ] **Step 4: Implement the evaluation CLI**

Create `robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py`.

CLI:

```text
evaluate_rover_aligned_benchmark.py BENCHMARK_ROOT
  --method "DBoW2=scores/dbow2.jsonl"
  --method "NetVLAD=scores/netvlad.jsonl"
  --method "SALAD=scores/salad.jsonl"
  --method "ROVER-like=scores/rover_like.jsonl"
  --method "LoopAnything=scores/loopanything.jsonl"
  --output-dir BENCHMARK_ROOT/metrics
```

Parse each `NAME=PATH` exactly once. Write:

```text
metrics/metrics_per_sequence.csv
metrics/metrics_summary.json
metrics/table1.md
```

For each score path, require the sibling `<stem>.manifest.json` and verify its
candidate-manifest and score-file hashes. Fail before writing outputs if the
annotation seal, sequence groups, score provenance, or any method score file
is incomplete or inconsistent.

- [ ] **Step 5: Run evaluator and metrics tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_metrics.py \
  tests/robust_loop_verifier/test_rover_evaluation.py \
  -q
python -m py_compile \
  src/robust_loop_verifier/rover_evaluation.py \
  robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py
```

Expected: all tests pass and compilation exits 0.

- [ ] **Step 6: Record the checkpoint**

Run:

```bash
git status --short
```

---

### Task 5: Score the Frozen Pairs Without Reretrieval

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/rover_pair_scoring.py`
- Modify: `LoopAnything/src/robust_loop_verifier/pipeline.py`
- Modify: `LoopAnything/src/robust_loop_verifier/score_sweep.py`
- Create: `LoopAnything/robust_loop_verification_scripts/score_rover_aligned_benchmark.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_rover_pair_scoring.py`

- [ ] **Step 1: Write failing fixed-pair scorer tests**

Create `tests/robust_loop_verifier/test_rover_pair_scoring.py` with:

```python
def test_descriptor_scorer_scores_manifest_pairs_without_retrieval():
    pairs = [
        {"pair_id": "p1", "query_idx": 2, "candidate_idx": 0},
        {"pair_id": "p2", "query_idx": 2, "candidate_idx": 1},
    ]
    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2],
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]]),
    )

    rows = score_descriptor_pairs(pairs, descriptors)

    assert rows == [
        {"pair_id": "p1", "score": 0.8, "status": "ok"},
        {"pair_id": "p2", "score": 0.6, "status": "ok"},
    ]


def test_verifier_scorer_preserves_pair_ids_and_explicit_failures():
    pairs = [
        {"pair_id": "p1", "query_idx": 10, "candidate_idx": 2, "rank": 1},
        {"pair_id": "p2", "query_idx": 10, "candidate_idx": 3, "rank": 2},
    ]
    records = [
        {"pair_id": "p1", "score_rover": -0.1, "pgo_converged": True},
        {"pair_id": "p2", "score_rover": None, "pgo_converged": False},
    ]

    rows = candidate_records_to_score_rows(records, "score_rover")

    assert rows == [
        {"pair_id": "p1", "score": -0.1, "status": "ok"},
        {"pair_id": "p2", "score": None, "status": "failed"},
    ]
```

Add tests that:

- reject score output with reordered, missing, duplicate, or unknown pair IDs;
- compute SALAD/NetVLAD cosine scores only for listed pairs;
- group verifier work by the frozen query and rank without calling
  `retrieve_historical_topk`;
- never open `positives.jsonl`, `annotations.jsonl`, or
  `annotation_seal.json`;
- compute `ROVER deformation only` and
  `query_gate_graph:def=0.5,res=0.25,margin=0` from the same candidate records.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src:. pytest \
  tests/robust_loop_verifier/test_rover_pair_scoring.py \
  -q
```

Expected: FAIL because `rover_pair_scoring` does not exist.

- [ ] **Step 3: Implement descriptor and score-row primitives**

Create `src/robust_loop_verifier/rover_pair_scoring.py` with:

```python
def score_descriptor_pairs(
    pairs: Sequence[Mapping[str, object]],
    descriptors: DescriptorSet,
) -> list[dict[str, object]]:
    row_by_idx = {
        int(keyframe_idx): row
        for row, keyframe_idx in enumerate(descriptors.keyframe_indices)
    }
    output = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        query_row = row_by_idx.get(int(pair["query_idx"]))
        candidate_row = row_by_idx.get(int(pair["candidate_idx"]))
        if query_row is None or candidate_row is None:
            output.append({"pair_id": pair_id, "score": None, "status": "failed"})
            continue
        score = float(
            np.dot(
                descriptors.descriptors[query_row],
                descriptors.descriptors[candidate_row],
            )
        )
        output.append({"pair_id": pair_id, "score": score, "status": "ok"})
    return output


def candidate_records_to_score_rows(records, score_field) -> list[dict[str, object]]:
    rows = []
    for record in records:
        score = record.get(score_field)
        valid = score is not None and math.isfinite(float(score))
        rows.append(
            {
                "pair_id": str(record["pair_id"]),
                "score": float(score) if valid else None,
                "status": "ok" if valid else "failed",
            }
        )
    return rows
```

Also implement:

```python
def load_frozen_pairs_by_sequence(benchmark_root) -> dict[str, list[dict[str, object]]]
def compute_sequence_descriptors(pairs, cache_root, backend) -> DescriptorSet
def score_netvlad_pairs(benchmark_root, backend) -> list[dict[str, object]]
def score_salad_pairs(benchmark_root, backend) -> list[dict[str, object]]
def score_verifier_pairs(benchmark_root, config, da3_backend) -> list[dict[str, object]]
def validate_score_row_order(pairs, rows) -> None
```

Descriptor computation may deduplicate image inference within a sequence, but
scoring iterates `benchmark_pairs.jsonl` in manifest order. This module must
not import retrieval functions or label readers.

- [ ] **Step 4: Expose label-free verifier scoring**

In `pipeline.py`, extract the existing candidate-scoring body into:

```python
def score_frozen_query_candidates(
    *,
    config,
    query_idx,
    frozen_candidates,
    image_by_idx,
    odom_by_idx,
    cache_order,
    da3_runner,
    timing=None,
) -> list[dict[str, Any]]:
```

`frozen_candidates` contains the manifest `pair_id`, candidate index, rank,
and the SALAD cosine score computed for that fixed pair. The function performs
support selection, DA3/Sim3, and PGO exactly as the existing pipeline, but
does not call retrieval and does not accept or write a label. Add `pair_id` to
every returned candidate record. Keep `run_cached_sequence()` behavior
unchanged by adapting its retrieved candidates to this public function.

In `score_sweep.py`, expose:

```python
def compute_named_scores(
    records: Sequence[Mapping[str, Any]],
    method_name: str,
) -> list[float | None]:
```

Support at minimum:

```text
ROVER deformation only
query_gate_graph:def=0.5,res=0.25,margin=0
```

This function computes scores only; it must not read labels or metrics.

- [ ] **Step 5: Implement the frozen-pair scoring CLI**

Create `robust_loop_verification_scripts/score_rover_aligned_benchmark.py`:

```text
score_rover_aligned_benchmark.py BENCHMARK_ROOT
  --method netvlad|salad|verifier
  --backend real
  --device cuda
```

Behavior:

- `netvlad` writes `scores/netvlad.jsonl`;
- `salad` writes `scores/salad.jsonl`;
- `verifier` runs each frozen pair once, writes an ignored
  `candidate_records.jsonl`, then writes `scores/rover_like.jsonl` from
  `ROVER deformation only` and `scores/loopanything.jsonl` from
  `query_gate_graph:def=0.5,res=0.25,margin=0`;
- descriptor/model roots and dataset-specific verifier configs come from the
  frozen benchmark config recorded in `manifest.json`;
- all outputs preserve manifest order and contain exactly one explicit
  `ok`/`failed` row per pair;
- no mode reads `positives.jsonl`, `annotations.jsonl`, or reruns retrieval.
- each mode writes `scores/<method>.manifest.json` with the command, source
  commit, model/config paths and SHA256 values, candidate-manifest SHA256, and
  output score-file SHA256.

Refuse to overwrite an existing score file unless its SHA256 and content are
identical. A changed method configuration requires a new score filename or
benchmark experiment directory.

- [ ] **Step 6: Run scorer tests**

Run:

```bash
PYTHONPATH=src:. pytest \
  tests/robust_loop_verifier/test_rover_pair_scoring.py \
  tests/robust_loop_verifier/test_pipeline.py \
  tests/robust_loop_verifier/test_score_sweep.py \
  tests/baseline_scripts/test_netvlad_retrieval_pipeline.py \
  -q
python -m py_compile \
  src/robust_loop_verifier/rover_pair_scoring.py \
  robust_loop_verification_scripts/score_rover_aligned_benchmark.py
```

Expected: all tests pass and compilation exits 0.

- [ ] **Step 7: Record the checkpoint**

Run:

```bash
git status --short
```

---

### Task 6: Real-Data Candidate Smoke and Software Acceptance

**Files:**
- Modify only if required by observed defects:
  - `LoopAnything/src/robust_loop_verifier/rover_benchmark.py`
  - `LoopAnything/robust_loop_verification_scripts/build_rover_aligned_benchmark.py`
  - focused tests for any defect fixed

- [ ] **Step 1: Run the complete benchmark-focused test set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src:. pytest \
  tests/robust_loop_verifier/test_metrics.py \
  tests/robust_loop_verifier/test_rover_benchmark.py \
  tests/robust_loop_verifier/test_rover_annotation.py \
  tests/robust_loop_verifier/test_rover_evaluation.py \
  tests/robust_loop_verifier/test_rover_pair_scoring.py \
  tests/baseline_scripts/test_orb_dbow2_retrieval_pipeline.py \
  tests/baseline_scripts/test_netvlad_retrieval_pipeline.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Build the real ten-sequence benchmark**

Run:

```bash
PYTHONPATH=src:. /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/build_rover_aligned_benchmark.py \
  --config configs/robust_loop_verifier/rover_aligned_benchmark.yaml \
  --output-root workspace/rover_aligned_benchmark/benchmark_v1
```

Expected:

- ten sequence summaries;
- exactly ten non-empty sequence groups in canonical order;
- each selected query has ten candidates;
- no existing `positives.jsonl` is read;
- total pair count is at most 4,000;
- `manifest.json`, `benchmark_pairs.jsonl`, `scores/dbow2.jsonl`, and
  `scores/dbow2.manifest.json` exist;
- the manifest contains cache roots and all DBoW2 source, binary, library, and
  repository fingerprints.

- [ ] **Step 3: Verify determinism without overwriting**

Build to a temporary second root:

```bash
PYTHONPATH=src:. /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/build_rover_aligned_benchmark.py \
  --config configs/robust_loop_verifier/rover_aligned_benchmark.yaml \
  --output-root /tmp/rover_aligned_benchmark_repeat
sha256sum \
  workspace/rover_aligned_benchmark/benchmark_v1/benchmark_pairs.jsonl \
  /tmp/rover_aligned_benchmark_repeat/benchmark_pairs.jsonl
```

Expected: identical SHA256 values.

- [ ] **Step 4: Verify immutable output protection**

Run the Step 2 command again against the same output root.

Expected: non-zero exit before writing, with an error that the benchmark
version directory is non-empty.

- [ ] **Step 5: Launch the annotation UI for manual acceptance**

Run:

```bash
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py \
  workspace/rover_aligned_benchmark/benchmark_v1 \
  --open
```

Manually verify:

- six image panels load;
- current query/candidate dominate the layout;
- no index, rank, score, trajectory, DA3, or verifier output is visible;
- browser image URLs and API state do not expose filenames or keyframe indices;
- `P`, `N`, and `Backspace` work;
- rapid repeated key presses append only one event for the active state;
- restart resumes at the first unlabeled pair;
- finalize is rejected before 100% completion.

- [ ] **Step 6: Run repository regression tests**

Run:

```bash
PYTHONPATH=src:. pytest tests/robust_loop_verifier tests/baseline_scripts -q
```

Expected: all tests pass.

- [ ] **Step 7: Inspect final changes**

Run:

```bash
git status --short
git diff --check
```

Expected: no whitespace errors and no generated benchmark artifacts tracked by git.

---

## Operational Completion After Software Acceptance

The six implementation tasks are complete when the focused tests, immutable
real-data build, and annotation UI acceptance pass. Paper benchmark completion
is a separate human-in-the-loop operation:

1. generate scores on the frozen pairs before opening the annotation tool:

```bash
PYTHONPATH=src:. /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/score_rover_aligned_benchmark.py \
  workspace/rover_aligned_benchmark/benchmark_v1 --method netvlad --device cuda
PYTHONPATH=src:. /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/score_rover_aligned_benchmark.py \
  workspace/rover_aligned_benchmark/benchmark_v1 --method salad --device cuda
PYTHONPATH=src:. /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/score_rover_aligned_benchmark.py \
  workspace/rover_aligned_benchmark/benchmark_v1 \
  --method verifier --backend real --device cuda
```

2. verify that every score file has one row per frozen pair and uses explicit
   `status: failed` rows for failures;
3. annotate every frozen pair with `P` or `N`;
4. call `/api/finalize` and verify `annotations.jsonl` plus
   `annotation_seal.json`;
5. run:

```bash
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py \
  workspace/rover_aligned_benchmark/benchmark_v1 \
  --method "DBoW2=scores/dbow2.jsonl" \
  --method "NetVLAD=scores/netvlad.jsonl" \
  --method "SALAD=scores/salad.jsonl" \
  --method "ROVER-like=scores/rover_like.jsonl" \
  --method "LoopAnything=scores/loopanything.jsonl" \
  --output-dir workspace/rover_aligned_benchmark/benchmark_v1/metrics
```

The paper-facing benchmark is complete only when this command verifies the
seal, consumes all ten sequence groups and all frozen pairs, and writes
`metrics_per_sequence.csv`, `metrics_summary.json`, and `table1.md`.
