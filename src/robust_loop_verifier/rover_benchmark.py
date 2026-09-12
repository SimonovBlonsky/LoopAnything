from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .io import read_jsonl, write_json, write_jsonl
from .retrieval import RetrievalRecord


@dataclass(frozen=True)
class BenchmarkSequence:
    dataset: str
    platform: str
    sequence: str
    cache: Path


@dataclass(frozen=True)
class KeyframeImage:
    keyframe_idx: int
    image_path: Path
    relative_image_path: str


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
        (bin_index + 0.5) * len(ordered) / limit - 0.5 for bin_index in range(limit)
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


def read_sequence_keyframes(sequence: BenchmarkSequence) -> list[KeyframeImage]:
    cache = Path(sequence.cache)
    seen_indices: set[int] = set()
    keyframes: list[KeyframeImage] = []
    for row in read_jsonl(cache / "keyframes.jsonl"):
        keyframe_idx = int(row["idx"])
        if keyframe_idx in seen_indices:
            raise ValueError(
                f"duplicate keyframe index in {sequence.dataset}/{sequence.platform}/"
                f"{sequence.sequence}: {keyframe_idx}"
            )
        seen_indices.add(keyframe_idx)
        image_path = row.get("image_path")
        if image_path is None:
            continue
        relative_image_path = str(image_path)
        absolute_image_path = cache / relative_image_path
        if not absolute_image_path.is_file():
            continue
        keyframes.append(
            KeyframeImage(
                keyframe_idx=keyframe_idx,
                image_path=absolute_image_path,
                relative_image_path=relative_image_path,
            )
        )
    if not keyframes:
        raise ValueError(f"No keyframes with existing images found in {cache}")
    return sorted(keyframes, key=lambda keyframe: keyframe.keyframe_idx)


def build_image_context(
    keyframes: Sequence[KeyframeImage],
    keyframe_idx: int,
) -> tuple[str | None, str, str | None]:
    for position, keyframe in enumerate(keyframes):
        if int(keyframe.keyframe_idx) != int(keyframe_idx):
            continue
        previous_path = keyframes[position - 1].relative_image_path if position > 0 else None
        next_path = (
            keyframes[position + 1].relative_image_path
            if position + 1 < len(keyframes)
            else None
        )
        return previous_path, keyframe.relative_image_path, next_path
    raise KeyError(f"keyframe index not found: {keyframe_idx}")


def build_pairs_for_sequence(
    sequence: BenchmarkSequence,
    retrieval_records: Mapping[int, RetrievalRecord],
    selected_queries: Sequence[int],
    recent_exclusion: int,
) -> list[BenchmarkPair]:
    keyframes = read_sequence_keyframes(sequence)
    keyframes_by_idx = {keyframe.keyframe_idx: keyframe for keyframe in keyframes}
    pairs: list[BenchmarkPair] = []
    for query_idx in selected_queries:
        retrieval = retrieval_records.get(int(query_idx))
        if retrieval is None:
            continue
        query_keyframe = keyframes_by_idx[int(query_idx)]
        for candidate in sorted(retrieval.candidates, key=lambda item: item.rank):
            candidate_idx = int(candidate.candidate_idx)
            if candidate_idx >= int(query_idx) - int(recent_exclusion):
                continue
            candidate_keyframe = keyframes_by_idx.get(candidate_idx)
            if candidate_keyframe is None:
                continue
            pairs.append(
                BenchmarkPair(
                    pair_id=(
                        f"{sequence.dataset}_{sequence.platform}_{sequence.sequence}_"
                        f"q{int(query_idx):06d}_c{candidate_idx:06d}"
                    ),
                    dataset=sequence.dataset,
                    platform=sequence.platform,
                    sequence=sequence.sequence,
                    query_idx=int(query_idx),
                    candidate_idx=candidate_idx,
                    rank=int(candidate.rank),
                    dbow2_score=float(candidate.score),
                    query_image=query_keyframe.relative_image_path,
                    candidate_image=candidate_keyframe.relative_image_path,
                    query_context=build_image_context(keyframes, int(query_idx)),
                    candidate_context=build_image_context(keyframes, candidate_idx),
                )
            )
    return pairs


def validate_pair_manifest(
    pairs: Sequence[BenchmarkPair],
    recent_exclusion_by_sequence: Mapping[tuple[str, str, str], int],
) -> None:
    if not pairs:
        raise ValueError("empty pair manifest is not allowed")

    seen_pair_ids: set[str] = set()
    pairs_by_query: dict[tuple[str, str, str, int], list[BenchmarkPair]] = {}
    for pair in pairs:
        if pair.pair_id in seen_pair_ids:
            raise ValueError(f"duplicate pair_id: {pair.pair_id}")
        seen_pair_ids.add(pair.pair_id)
        sequence_key = (pair.dataset, pair.platform, pair.sequence)
        if sequence_key not in recent_exclusion_by_sequence:
            raise ValueError(f"missing recent exclusion for sequence: {sequence_key}")
        if pair.candidate_idx >= pair.query_idx:
            raise ValueError(f"non-causal candidate in pair {pair.pair_id}")
        recent_exclusion = int(recent_exclusion_by_sequence[sequence_key])
        if pair.candidate_idx >= pair.query_idx - recent_exclusion:
            raise ValueError(f"recent candidate violates exclusion in pair {pair.pair_id}")
        pairs_by_query.setdefault((*sequence_key, pair.query_idx), []).append(pair)

    for query_key, query_pairs in pairs_by_query.items():
        if len(query_pairs) != 10:
            raise ValueError(f"query {query_key} must have exactly 10 candidates")
        ranks = [int(pair.rank) for pair in query_pairs]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"duplicate rank for query {query_key}")
        if set(ranks) != set(range(1, 11)):
            raise ValueError(f"query {query_key} must have ranks 1..10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen_benchmark(
    output_root: Path,
    pairs: Sequence[BenchmarkPair],
    manifest: Mapping[str, object],
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "benchmark_pairs.jsonl", (asdict(pair) for pair in pairs))
    write_json(output_root / "manifest.json", manifest)
