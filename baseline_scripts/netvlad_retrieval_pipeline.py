#!/usr/bin/env python3
"""Run NetVLAD-only retrieval on prebuilt robust-loop-verifier VPR caches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOPANYTHING_ROOT.parent
SRC_ROOT = LOOPANYTHING_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_loop_verifier.io import read_json, read_jsonl  # noqa: E402
from robust_loop_verifier.metrics import average_precision, max_recall_at_100_precision  # noqa: E402
from robust_loop_verifier.retrieval import (  # noqa: E402
    DescriptorSet,
    retrieve_historical_topk,
)


DEFAULT_CACHE_ROOTS = {
    "fusionportablev2": Path(
        "/data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2"
    ),
    "geode": Path("/data/datasets/GEODE/robust_loop_verifier_cache/GEODE"),
    "ntu_viral": Path("/data/datasets/NTU-VIRAL/robust_loop_verifier_cache/NTU-VIRAL"),
}

DEFAULT_SEQUENCES_BY_PLATFORM = {
    "handheld": [
        "handheld_escalator00",
        "handheld_escalator01",
        "handheld_grass00",
        "handheld_room00",
        "handheld_room01",
    ],
    "ugv": [
        "ugv_campus01",
        "ugv_parking00",
        "ugv_parking01",
        "ugv_parking02",
        "ugv_parking03",
    ],
}
DEFAULT_SEQUENCES_BY_DATASET = {
    "fusionportablev2": DEFAULT_SEQUENCES_BY_PLATFORM,
    "geode": {"Offroad": ["Offroad02_beta"]},
    "ntu_viral": {
        "NTU-VIRAL": ["eee_01", "eee_02", "nya_01", "nya_02", "nya_03"],
    },
}

DATASET_CHOICES = tuple(DEFAULT_CACHE_ROOTS)
PLATFORM_CHOICES = tuple(
    dict.fromkeys(
        platform
        for sequences_by_platform in DEFAULT_SEQUENCES_BY_DATASET.values()
        for platform in sequences_by_platform
    )
)
METHOD_NAME = "NetVLAD score only"


@dataclass(frozen=True)
class SequenceSpec:
    platform: str
    sequence: str
    sequence_cache: Path


@dataclass(frozen=True)
class SequenceResult:
    platform: str
    sequence: str
    sequence_cache: str
    metrics: dict[str, float]
    query_count: int
    descriptor_count: int
    candidate_count: int
    positive_candidate_count: int
    recent_exclusion_keyframes: int


class NetVladDescriptorBackend:
    """Local VGG16-NetVLAD-Pitts30K descriptor backend."""

    def __init__(self, netvlad_root: Path, device: str):
        self.netvlad_root = Path(netvlad_root)
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            import torch

            if not self.netvlad_root.is_dir():
                raise FileNotFoundError(f"NetVLAD repository was not found: {self.netvlad_root}")
            added_path = str(self.netvlad_root) not in sys.path
            if added_path:
                sys.path.insert(0, str(self.netvlad_root))
            try:
                from netvlad import NetVLAD
            finally:
                if added_path and str(self.netvlad_root) in sys.path:
                    sys.path.remove(str(self.netvlad_root))
            self._model = NetVLAD(NetVLAD.default_conf).eval().to(self.device)
            torch.set_grad_enabled(False)
        return self._model

    def compute(self, image_paths: Sequence[str], keyframe_indices: Sequence[int]) -> DescriptorSet:
        if len(image_paths) != len(keyframe_indices):
            raise ValueError("image_paths and keyframe_indices must have matching lengths")
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        import torch

        model = self._load_model()
        descriptors = []
        with torch.inference_mode():
            for image_path in image_paths:
                image = _load_image_for_netvlad(Path(image_path), self.device)
                descriptor = model({"image": image})["global_descriptor"]
                descriptors.append(descriptor.detach().cpu().numpy().squeeze(0))

        descriptor_matrix = np.asarray(descriptors, dtype=np.float64)
        return DescriptorSet(
            keyframe_indices=[int(index) for index in keyframe_indices],
            descriptors=_l2_normalize_rows(descriptor_matrix),
        )


def resolve_sequence_specs(
    dataset: str,
    cache_root: Path,
    platforms: Sequence[str],
    strict: bool,
    sequences: Sequence[str] | None = None,
) -> list[SequenceSpec]:
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset: {dataset}")

    specs: list[SequenceSpec] = []
    missing: list[Path] = []
    sequences_by_platform = DEFAULT_SEQUENCES_BY_DATASET[dataset]
    for platform in platforms:
        if platform not in sequences_by_platform:
            raise ValueError(f"Unsupported platform for {dataset}: {platform}")
        sequence_names = list(sequences) if sequences else sequences_by_platform[platform]
        for sequence in sequence_names:
            sequence_cache = Path(cache_root) / platform / sequence
            required_files = [
                sequence_cache / "manifest.json",
                sequence_cache / "keyframes.jsonl",
                sequence_cache / "positives.jsonl",
            ]
            if all(path.is_file() for path in required_files):
                specs.append(
                    SequenceSpec(
                        platform=platform,
                        sequence=sequence,
                        sequence_cache=sequence_cache,
                    )
                )
            else:
                missing.append(sequence_cache)

    if missing and strict:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing sequence caches:\n{missing_text}")
    for path in missing:
        print(f"[skip] missing sequence cache: {path}", file=sys.stderr)
    return specs


def evaluate_sequence(
    spec: SequenceSpec,
    descriptor_backend,
    top_k: int,
    recent_exclusion_keyframes: int | None,
) -> SequenceResult:
    keyframes = _read_keyframes(spec.sequence_cache)
    positives_by_query = _read_positives(spec.sequence_cache / "positives.jsonl")
    manifest = read_json(spec.sequence_cache / "manifest.json")
    recent_exclusion = (
        int(recent_exclusion_keyframes)
        if recent_exclusion_keyframes is not None
        else int(manifest.get("recent_exclusion_keyframes", 30))
    )

    image_paths = [str(path) for _, path in keyframes]
    keyframe_indices = [idx for idx, _ in keyframes]
    descriptors = descriptor_backend.compute(image_paths, keyframe_indices)

    labels: list[bool] = []
    scores: list[float] = []
    for query_idx in descriptors.keyframe_indices:
        retrieval = retrieve_historical_topk(
            query_idx=int(query_idx),
            descriptors=descriptors,
            top_k=top_k,
            recent_exclusion_keyframes=recent_exclusion,
        )
        positive_indices = positives_by_query.get(int(query_idx), set())
        for candidate in retrieval.candidates:
            labels.append(int(candidate.candidate_idx) in positive_indices)
            scores.append(float(candidate.score))

    metrics = {
        "AP": average_precision(labels, scores) if labels else 0.0,
        "MR@100P": max_recall_at_100_precision(labels, scores) if labels else 0.0,
    }
    return SequenceResult(
        platform=spec.platform,
        sequence=spec.sequence,
        sequence_cache=str(spec.sequence_cache),
        metrics=metrics,
        query_count=len(keyframe_indices),
        descriptor_count=len(descriptors.keyframe_indices),
        candidate_count=len(labels),
        positive_candidate_count=sum(labels),
        recent_exclusion_keyframes=recent_exclusion,
    )


def summarize_results(results: Sequence[SequenceResult]) -> dict[str, object]:
    if not results:
        return {
            "method": METHOD_NAME,
            "sequence_count": 0,
            "average": {"AP": 0.0, "MR@100P": 0.0},
            "sequences": [],
        }
    return {
        "method": METHOD_NAME,
        "sequence_count": len(results),
        "average": {
            "AP": sum(result.metrics["AP"] for result in results) / len(results),
            "MR@100P": sum(result.metrics["MR@100P"] for result in results) / len(results),
        },
        "sequences": [asdict(result) for result in results],
    }


def write_outputs(output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sequence_rows = list(summary["sequences"])
    with (output_dir / "metrics_per_sequence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "platform",
                "sequence",
                "method",
                "AP",
                "MR@100P",
                "candidate_count",
                "positive_candidate_count",
            ]
        )
        for row in sequence_rows:
            writer.writerow(
                [
                    row["platform"],
                    row["sequence"],
                    METHOD_NAME,
                    row["metrics"]["AP"],
                    row["metrics"]["MR@100P"],
                    row["candidate_count"],
                    row["positive_candidate_count"],
                ]
            )
    _write_markdown(output_dir / "metrics.md", summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="fusionportablev2")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--platform",
        action="append",
        choices=PLATFORM_CHOICES,
        default=None,
        help="Platform to evaluate. Repeat to evaluate multiple platforms. Defaults to handheld+ugv.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Sequence name override for every selected platform. Repeat for multiple sequences.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recent-exclusion-keyframes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--netvlad-root",
        type=Path,
        default=WORKSPACE_ROOT / "netvlad_image_retrieval",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=LOOPANYTHING_ROOT / "workspace" / "baseline_runs" / "netvlad_retrieval",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def evaluate_sequence_specs(sequence_specs, evaluate, *, keep_going: bool):
    results = []
    failures = []
    for spec in sequence_specs:
        print(f"[run] {spec.platform}/{spec.sequence}")
        try:
            results.append(evaluate(spec))
        except Exception as error:
            if not keep_going:
                raise
            failure = {
                "platform": spec.platform,
                "sequence": spec.sequence,
                "error": str(error),
            }
            failures.append(failure)
            print(f"[error] {spec.platform}/{spec.sequence}: {error}", file=sys.stderr)
    return results, failures


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if args.recent_exclusion_keyframes is not None and args.recent_exclusion_keyframes < 0:
        raise ValueError("--recent-exclusion-keyframes must be non-negative")

    cache_root = Path(args.cache_root or DEFAULT_CACHE_ROOTS[args.dataset])
    platforms = list(args.platform or DEFAULT_SEQUENCES_BY_DATASET[args.dataset])
    sequence_specs = resolve_sequence_specs(
        dataset=args.dataset,
        cache_root=cache_root,
        platforms=platforms,
        strict=args.strict,
        sequences=args.sequence,
    )
    if not sequence_specs:
        raise FileNotFoundError(f"No valid sequence caches found under {cache_root}")

    descriptor_backend = NetVladDescriptorBackend(args.netvlad_root, args.device)
    results, failures = evaluate_sequence_specs(
        sequence_specs,
        lambda spec: evaluate_sequence(
            spec,
            descriptor_backend=descriptor_backend,
            top_k=args.top_k,
            recent_exclusion_keyframes=args.recent_exclusion_keyframes,
        ),
        keep_going=args.keep_going,
    )

    summary = summarize_results(results)
    summary["failures"] = failures
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / args.dataset / run_name
    write_outputs(output_dir, summary)
    _print_summary(summary, output_dir)
    return 1 if failures else 0


def _read_keyframes(sequence_cache: Path) -> list[tuple[int, Path]]:
    rows = []
    for row in read_jsonl(sequence_cache / "keyframes.jsonl"):
        image_path = row.get("image_path")
        if image_path is None:
            continue
        path = sequence_cache / str(image_path)
        if not path.is_file():
            continue
        rows.append((int(row["idx"]), path))
    if not rows:
        raise ValueError(f"No keyframes with existing images found in {sequence_cache}")
    return rows


def _read_positives(path: Path) -> dict[int, set[int]]:
    positives = {}
    for row in read_jsonl(path):
        positives[int(row["query_idx"])] = {
            int(candidate_idx) for candidate_idx in row.get("positive_indices", [])
        }
    return positives


def _load_image_for_netvlad(image_path: Path, device: str):
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    bgr = rgb[:, :, ::-1].copy()
    tensor = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).to(device)
    min_value = tensor.amin()
    max_value = tensor.amax()
    denom = max_value - min_value
    if float(denom.detach().cpu()) <= 0.0:
        return torch.zeros_like(tensor)
    return (tensor - min_value) / denom


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Descriptor matrix must have shape (N, D)")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Descriptor norms must be finite and positive")
    return matrix / norms[:, None]


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "| platform | sequence | method | AP | MR@100P |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary["sequences"]:
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} |".format(
                row["platform"],
                row["sequence"],
                METHOD_NAME,
                row["metrics"]["AP"],
                row["metrics"]["MR@100P"],
            )
        )
    average = summary["average"]
    lines.extend(
        [
            "| average | {} sequences | {} | {:.4f} | {:.4f} |".format(
                summary["sequence_count"],
                METHOD_NAME,
                average["AP"],
                average["MR@100P"],
            )
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, object], output_dir: Path) -> None:
    average = summary["average"]
    print(f"output_dir: {output_dir}")
    print(
        "average {}: AP={:.6f} MR@100P={:.6f}".format(
            METHOD_NAME,
            average["AP"],
            average["MR@100P"],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
