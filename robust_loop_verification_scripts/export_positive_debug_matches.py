#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SEQUENCE_CACHE = Path(
    "/data/datasets/FusionPortable/robust_loop_verifier_cache/"
    "FusionPortableV2/handheld/handheld_escalator00"
)


def export_positive_debug_matches(
    sequence_cache: Path = DEFAULT_SEQUENCE_CACHE,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    sequence_cache = Path(sequence_cache)
    if output_dir is None:
        output_dir = sequence_cache / "positive_debug"
    else:
        output_dir = Path(output_dir)

    keyframes = _read_keyframes(sequence_cache / "keyframes.jsonl", sequence_cache)
    positives = list(_read_jsonl(sequence_cache / "positives.jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    skipped = []
    for row in positives:
        query_idx = int(row["query_idx"])
        positive_indices = [int(idx) for idx in row.get("positive_indices", [])]
        if not positive_indices:
            continue

        match_dir = output_dir / _match_dir_name(query_idx, positive_indices)
        match_dir.mkdir(parents=True, exist_ok=True)
        missing_images = []

        query_image = keyframes.get(query_idx)
        if query_image is None or not query_image.is_file():
            missing_images.append({"role": "query", "idx": query_idx})
        else:
            shutil.copy2(query_image, match_dir / f"query_{query_idx:06d}{query_image.suffix}")

        for candidate_idx in positive_indices:
            candidate_image = keyframes.get(candidate_idx)
            if candidate_image is None or not candidate_image.is_file():
                missing_images.append({"role": "candidate", "idx": candidate_idx})
                continue
            shutil.copy2(
                candidate_image,
                match_dir / f"candidate_{candidate_idx:06d}{candidate_image.suffix}",
            )

        meta = {
            "query_idx": query_idx,
            "positive_indices": positive_indices,
            "missing_images": missing_images,
            "source_sequence_cache": str(sequence_cache),
        }
        (match_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        item = {
            "query_idx": query_idx,
            "positive_indices": positive_indices,
            "match_dir": str(match_dir),
            "missing_image_count": len(missing_images),
        }
        if missing_images:
            skipped.append(item)
        exported.append(item)

    summary = {
        "sequence_cache": str(sequence_cache),
        "output_dir": str(output_dir),
        "exported_match_count": len(exported),
        "exported_positive_pair_count": sum(len(item["positive_indices"]) for item in exported),
        "matches_with_missing_images": skipped,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_keyframes(path: Path, sequence_cache: Path) -> dict[int, Path | None]:
    keyframes: dict[int, Path | None] = {}
    for row in _read_jsonl(path):
        idx = int(row["idx"])
        image_path = row.get("image_path")
        if image_path is None:
            keyframes[idx] = None
            continue
        path_value = Path(str(image_path))
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError(f"Unsafe image_path for keyframe {idx}: {image_path!r}")
        keyframes[idx] = sequence_cache / path_value
    return keyframes


def _read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL rows must be objects: {path}")
            yield row


def _match_dir_name(query_idx: int, candidate_indices: list[int]) -> str:
    candidates = "-".join(f"{candidate_idx:06d}" for candidate_idx in candidate_indices)
    name = f"q{query_idx:06d}__c{candidates}"
    if len(name) <= 120:
        return name

    digest = hashlib.sha1(candidates.encode("ascii")).hexdigest()[:10]
    preview = "-".join(f"{candidate_idx:06d}" for candidate_idx in candidate_indices[:8])
    return f"q{query_idx:06d}__n{len(candidate_indices):03d}__c{preview}__h{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export query/candidate images for each positive label row in a sequence cache."
        )
    )
    parser.add_argument(
        "sequence_cache",
        nargs="?",
        type=Path,
        default=DEFAULT_SEQUENCE_CACHE,
        help="Preprocessed robust loop verifier sequence cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sequence_cache>/positive_debug.",
    )
    args = parser.parse_args()

    summary = export_positive_debug_matches(args.sequence_cache, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
