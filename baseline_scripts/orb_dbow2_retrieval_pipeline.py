#!/usr/bin/env python3
"""Run ORB-SLAM3 ORB DBoW2 retrieval on prebuilt VPR caches."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOPANYTHING_ROOT.parent
SRC_ROOT = LOOPANYTHING_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_loop_verifier.io import read_json, read_jsonl  # noqa: E402
from robust_loop_verifier.metrics import average_precision, max_recall_at_100_precision  # noqa: E402
from robust_loop_verifier.retrieval import RetrievalCandidate, RetrievalRecord  # noqa: E402


DEFAULT_CACHE_ROOTS = {
    "fusionportablev2": Path(
        "/data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2"
    ),
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

DATASET_CHOICES = tuple(DEFAULT_CACHE_ROOTS)
PLATFORM_CHOICES = tuple(DEFAULT_SEQUENCES_BY_PLATFORM)
METHOD_NAME = "ORB DBoW2 score only"


@dataclass(frozen=True)
class KeyframeImage:
    keyframe_idx: int
    image_path: Path


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


class RetrievalBackend(Protocol):
    def retrieve(
        self,
        keyframes: Sequence[KeyframeImage],
        top_k: int,
        recent_exclusion_keyframes: int,
    ) -> dict[int, RetrievalRecord]:
        ...


class OrbDbow2RetrievalBackend:
    """Runtime-compiled ORB-SLAM3 ORB DBoW2 retrieval backend."""

    def __init__(
        self,
        orb_slam3_root: Path,
        vocabulary_path: Path,
        helper_build_dir: Path,
        rebuild_helper: bool,
        nfeatures: int,
        scale_factor: float,
        nlevels: int,
        ini_fast: int,
        min_fast: int,
    ):
        self.orb_slam3_root = Path(orb_slam3_root)
        self.vocabulary_path = Path(vocabulary_path)
        self.helper_build_dir = Path(helper_build_dir)
        self.rebuild_helper = bool(rebuild_helper)
        self.nfeatures = int(nfeatures)
        self.scale_factor = float(scale_factor)
        self.nlevels = int(nlevels)
        self.ini_fast = int(ini_fast)
        self.min_fast = int(min_fast)

    def retrieve(
        self,
        keyframes: Sequence[KeyframeImage],
        top_k: int,
        recent_exclusion_keyframes: int,
    ) -> dict[int, RetrievalRecord]:
        helper_bin = self.ensure_helper_built()
        with tempfile.TemporaryDirectory(prefix="orb-dbow2-retrieval-") as tmpdir:
            image_list = Path(tmpdir) / "images.tsv"
            _write_image_list(image_list, keyframes)
            command = [
                str(helper_bin),
                str(self.vocabulary_path),
                str(image_list),
                str(int(top_k)),
                str(int(recent_exclusion_keyframes)),
                str(self.nfeatures),
                str(self.scale_factor),
                str(self.nlevels),
                str(self.ini_fast),
                str(self.min_fast),
            ]
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        return parse_helper_output(completed.stdout)

    def ensure_helper_built(self) -> Path:
        if not self.vocabulary_path.is_file():
            raise FileNotFoundError(f"ORB vocabulary was not found: {self.vocabulary_path}")
        if not self.orb_slam3_root.is_dir():
            raise FileNotFoundError(f"ORB_SLAM3 root was not found: {self.orb_slam3_root}")

        self.helper_build_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.helper_build_dir / "orb_dbow2_retrieval_helper.cpp"
        binary_path = self.helper_build_dir / "orb_dbow2_retrieval_helper"
        source_text = _helper_source()
        if self.rebuild_helper or not source_path.is_file() or source_path.read_text() != source_text:
            source_path.write_text(source_text, encoding="utf-8")
        if self.rebuild_helper or not binary_path.is_file():
            self._compile_helper(source_path, binary_path)
        return binary_path

    def _compile_helper(self, source_path: Path, binary_path: Path) -> None:
        opencv_flags = _opencv_pkg_config_flags()
        dbow2_lib_dir = self.orb_slam3_root / "Thirdparty" / "DBoW2" / "lib"
        command = [
            "g++",
            "-std=c++11",
            "-O3",
            str(source_path),
            str(self.orb_slam3_root / "src" / "ORBextractor.cc"),
            "-I",
            str(self.orb_slam3_root),
            "-I",
            str(self.orb_slam3_root / "include"),
            "-I",
            str(self.orb_slam3_root / "Thirdparty" / "DBoW2"),
            "-L",
            str(dbow2_lib_dir),
            "-lDBoW2",
            f"-Wl,-rpath,{dbow2_lib_dir}",
            *opencv_flags,
            "-o",
            str(binary_path),
        ]
        subprocess.run(command, check=True)


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
    for platform in platforms:
        if platform not in DEFAULT_SEQUENCES_BY_PLATFORM:
            raise ValueError(f"Unsupported platform: {platform}")
        sequence_names = list(sequences) if sequences else DEFAULT_SEQUENCES_BY_PLATFORM[platform]
        for sequence in sequence_names:
            sequence_cache = Path(cache_root) / platform / sequence
            required_files = [
                sequence_cache / "manifest.json",
                sequence_cache / "keyframes.jsonl",
                sequence_cache / "positives.jsonl",
            ]
            if all(path.is_file() for path in required_files):
                specs.append(SequenceSpec(platform, sequence, sequence_cache))
            else:
                missing.append(sequence_cache)

    if missing and strict:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing sequence caches:\n{missing_text}")
    for path in missing:
        print(f"[skip] missing sequence cache: {path}", file=sys.stderr)
    return specs


def parse_helper_output(stdout: str) -> dict[int, RetrievalRecord]:
    candidates_by_query: dict[int, list[RetrievalCandidate]] = {}
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split("\t")
        if len(fields) != 4:
            raise ValueError(
                f"helper output row {line_number} must have 4 tab-separated fields"
            )
        try:
            query_idx = int(fields[0])
            candidate_idx = int(fields[1])
            rank = int(fields[2])
            score = float(fields[3])
        except ValueError as exc:
            raise ValueError(f"helper output row {line_number} contains invalid values") from exc
        candidates_by_query.setdefault(query_idx, []).append(
            RetrievalCandidate(query_idx, candidate_idx, rank, score)
        )
    return {
        query_idx: RetrievalRecord(
            query_idx=query_idx,
            candidates=sorted(candidates, key=lambda candidate: candidate.rank),
        )
        for query_idx, candidates in candidates_by_query.items()
    }


def evaluate_sequence(
    spec: SequenceSpec,
    retrieval_backend: RetrievalBackend,
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

    retrieval_records = retrieval_backend.retrieve(
        keyframes,
        top_k=top_k,
        recent_exclusion_keyframes=recent_exclusion,
    )
    labels: list[bool] = []
    scores: list[float] = []
    for retrieval in retrieval_records.values():
        positive_indices = positives_by_query.get(int(retrieval.query_idx), set())
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
        query_count=len(keyframes),
        descriptor_count=len(keyframes),
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
        for row in summary["sequences"]:
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
    parser.add_argument("--orb-slam3-root", type=Path, default=WORKSPACE_ROOT / "ORB_SLAM3")
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=WORKSPACE_ROOT / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt",
    )
    parser.add_argument(
        "--helper-build-dir",
        type=Path,
        default=LOOPANYTHING_ROOT
        / "workspace"
        / "baseline_runs"
        / "orb_dbow2_retrieval"
        / "helper_build",
    )
    parser.add_argument("--rebuild-helper", action="store_true")
    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument("--nlevels", type=int, default=8)
    parser.add_argument("--ini-fast", type=int, default=20)
    parser.add_argument("--min-fast", type=int, default=7)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=LOOPANYTHING_ROOT / "workspace" / "baseline_runs" / "orb_dbow2_retrieval",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if args.recent_exclusion_keyframes is not None and args.recent_exclusion_keyframes < 0:
        raise ValueError("--recent-exclusion-keyframes must be non-negative")

    cache_root = Path(args.cache_root or DEFAULT_CACHE_ROOTS[args.dataset])
    platforms = list(args.platform or PLATFORM_CHOICES)
    sequence_specs = resolve_sequence_specs(
        dataset=args.dataset,
        cache_root=cache_root,
        platforms=platforms,
        strict=args.strict,
        sequences=args.sequence,
    )
    if not sequence_specs:
        raise FileNotFoundError(f"No valid sequence caches found under {cache_root}")

    retrieval_backend = OrbDbow2RetrievalBackend(
        orb_slam3_root=args.orb_slam3_root,
        vocabulary_path=args.vocabulary_path,
        helper_build_dir=args.helper_build_dir,
        rebuild_helper=args.rebuild_helper,
        nfeatures=args.nfeatures,
        scale_factor=args.scale_factor,
        nlevels=args.nlevels,
        ini_fast=args.ini_fast,
        min_fast=args.min_fast,
    )
    results = []
    for spec in sequence_specs:
        print(f"[run] {spec.platform}/{spec.sequence}")
        results.append(
            evaluate_sequence(
                spec,
                retrieval_backend=retrieval_backend,
                top_k=args.top_k,
                recent_exclusion_keyframes=args.recent_exclusion_keyframes,
            )
        )

    summary = summarize_results(results)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / args.dataset / run_name
    write_outputs(output_dir, summary)
    _print_summary(summary, output_dir)
    return 0


def _read_keyframes(sequence_cache: Path) -> list[KeyframeImage]:
    rows: list[KeyframeImage] = []
    for row in read_jsonl(sequence_cache / "keyframes.jsonl"):
        image_path = row.get("image_path")
        if image_path is None:
            continue
        path = sequence_cache / str(image_path)
        if not path.is_file():
            continue
        rows.append(KeyframeImage(keyframe_idx=int(row["idx"]), image_path=path))
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


def _write_image_list(path: Path, keyframes: Sequence[KeyframeImage]) -> None:
    lines = [f"{item.keyframe_idx}\t{item.image_path}\n" for item in keyframes]
    path.write_text("".join(lines), encoding="utf-8")


def _opencv_pkg_config_flags() -> list[str]:
    for package_name in ("opencv4", "opencv"):
        try:
            output = subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", package_name],
                text=True,
            )
            return shlex.split(output)
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError("OpenCV pkg-config package was not found: tried opencv4 and opencv")


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
    lines.append(
        "| average | {} sequences | {} | {:.4f} | {:.4f} |".format(
            summary["sequence_count"],
            METHOD_NAME,
            average["AP"],
            average["MR@100P"],
        )
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


def _helper_source() -> str:
    return r'''
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ORBextractor.h"
#include "ORBVocabulary.h"

namespace {

struct Entry {
  int keyframe_idx = 0;
  std::string image_path;
  bool valid = false;
  DBoW2::BowVector bow;
};

std::vector<Entry> ReadImageList(const std::string& path) {
  std::ifstream input(path.c_str());
  if (!input.is_open()) {
    throw std::runtime_error("cannot open image list: " + path);
  }
  std::vector<Entry> entries;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      throw std::runtime_error("image list rows must contain a tab separator");
    }
    Entry entry;
    entry.keyframe_idx = std::atoi(line.substr(0, tab).c_str());
    entry.image_path = line.substr(tab + 1);
    entries.push_back(entry);
  }
  return entries;
}

std::vector<cv::Mat> DescriptorRows(const cv::Mat& descriptors) {
  std::vector<cv::Mat> rows;
  rows.reserve(descriptors.rows);
  for (int row = 0; row < descriptors.rows; ++row) {
    rows.push_back(descriptors.row(row));
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "usage: helper VOCAB IMAGE_LIST TOP_K RECENT NFEATURES SCALE NLEVELS INI_FAST MIN_FAST\n";
    return 64;
  }

  const std::string vocabulary_path = argv[1];
  const std::string image_list_path = argv[2];
  const int top_k = std::atoi(argv[3]);
  const int recent_exclusion = std::atoi(argv[4]);
  const int nfeatures = std::atoi(argv[5]);
  const float scale_factor = std::atof(argv[6]);
  const int nlevels = std::atoi(argv[7]);
  const int ini_fast = std::atoi(argv[8]);
  const int min_fast = std::atoi(argv[9]);

  try {
    ORB_SLAM3::ORBVocabulary vocabulary;
    if (!vocabulary.loadFromTextFile(vocabulary_path)) {
      std::cerr << "failed to load ORB vocabulary: " << vocabulary_path << "\n";
      return 2;
    }

    ORB_SLAM3::ORBextractor extractor(nfeatures, scale_factor, nlevels, ini_fast, min_fast);
    std::vector<Entry> entries = ReadImageList(image_list_path);
    for (Entry& entry : entries) {
      cv::Mat image = cv::imread(entry.image_path, cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
        continue;
      }
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      std::vector<int> lapping_area;
      lapping_area.push_back(0);
      lapping_area.push_back(-1);
      extractor(image, cv::Mat(), keypoints, descriptors, lapping_area);
      if (descriptors.empty()) {
        continue;
      }
      std::vector<cv::Mat> descriptor_rows = DescriptorRows(descriptors);
      vocabulary.transform(descriptor_rows, entry.bow);
      entry.valid = true;
    }

    for (const Entry& query : entries) {
      if (!query.valid) {
        continue;
      }
      const int exclusion_threshold = query.keyframe_idx - recent_exclusion;
      std::vector<std::pair<double, int> > scored;
      for (const Entry& candidate : entries) {
        if (!candidate.valid) {
          continue;
        }
        if (candidate.keyframe_idx == query.keyframe_idx) {
          continue;
        }
        if (candidate.keyframe_idx >= exclusion_threshold) {
          continue;
        }
        const double score = vocabulary.score(query.bow, candidate.bow);
        scored.push_back(std::make_pair(score, candidate.keyframe_idx));
      }

      std::sort(scored.begin(), scored.end(),
                [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
                  if (lhs.first != rhs.first) {
                    return lhs.first > rhs.first;
                  }
                  return lhs.second < rhs.second;
                });

      const int count = std::min(top_k, static_cast<int>(scored.size()));
      for (int i = 0; i < count; ++i) {
        std::cout << query.keyframe_idx << '\t'
                  << scored[i].second << '\t'
                  << (i + 1) << '\t'
                  << scored[i].first << '\n';
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }

  return 0;
}
'''


if __name__ == "__main__":
    raise SystemExit(main())
