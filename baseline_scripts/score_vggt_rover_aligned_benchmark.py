#!/usr/bin/env python3
"""Score frozen ROVER-aligned benchmark pairs with VGGT track verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import cv2
import numpy as np


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOPANYTHING_ROOT.parent
DEFAULT_VGGT_ROOT = WORKSPACE_ROOT / "vggt"


@dataclass(frozen=True)
class RansacConfig:
    method: str = "usac_magsac"
    threshold_px: float = 1.0
    confidence: float = 0.999
    max_iters: int = 10000
    min_matches: int = 8


@dataclass(frozen=True)
class VGGTConfig:
    model_path: str = "facebook/VGGT-1B"
    preprocess_mode: str = "pad"
    grid_size: int = 32
    border_px: int = 8
    visibility_threshold: float = 0.2
    confidence_threshold: float = 0.2
    max_query_points: int = 2048
    dtype: str = "auto"


@dataclass(frozen=True)
class VGGTMatchResult:
    mkpts0: np.ndarray
    mkpts1: np.ndarray
    num_query_points: int
    num_visible_tracks: int
    mean_track_confidence: float
    mean_visibility: float


class VGGTTrackMatcher:
    def __init__(
        self,
        *,
        vggt_root: Path,
        device: str,
        vggt_config: VGGTConfig,
    ) -> None:
        self.vggt_root = Path(vggt_root)
        self.device = str(device)
        self.vggt_config = vggt_config
        self._model = None
        self._torch = None
        self._load_and_preprocess_images = None

    def prepare(self) -> None:
        self._load_model()
        self._load_image_loader()

    def match(self, query_image: Path, candidate_image: Path) -> VGGTMatchResult:
        torch = self._load_torch()
        model = self._load_model()
        load_images = self._load_image_loader()
        images = load_images(
            [str(query_image), str(candidate_image)],
            mode=str(self.vggt_config.preprocess_mode),
        ).to(self.device)
        if images.ndim != 4 or images.shape[0] != 2:
            raise ValueError(f"expected VGGT images with shape (2,C,H,W), got {tuple(images.shape)}")
        height = int(images.shape[-2])
        width = int(images.shape[-1])
        query_points = _make_grid_points(
            width=width,
            height=height,
            grid_size=int(self.vggt_config.grid_size),
            border_px=int(self.vggt_config.border_px),
            max_points=int(self.vggt_config.max_query_points),
        )
        if len(query_points) == 0:
            return VGGTMatchResult(
                mkpts0=np.zeros((0, 2), dtype=np.float32),
                mkpts1=np.zeros((0, 2), dtype=np.float32),
                num_query_points=0,
                num_visible_tracks=0,
                mean_track_confidence=0.0,
                mean_visibility=0.0,
            )
        query_points_t = torch.from_numpy(query_points).float().to(self.device)
        dtype = _resolve_autocast_dtype(torch, self.device, str(self.vggt_config.dtype))
        with torch.inference_mode():
            with _autocast_context(torch, self.device, dtype):
                predictions = model(images, query_points=query_points_t)

        tracks = predictions["track"].detach().float().cpu().numpy()
        visibility = predictions["vis"].detach().float().cpu().numpy()
        confidence = predictions["conf"].detach().float().cpu().numpy()
        if tracks.ndim != 4 or tracks.shape[0] != 1 or tracks.shape[1] != 2 or tracks.shape[-1] != 2:
            raise ValueError(f"unexpected VGGT track shape: {tracks.shape}")

        q_points = tracks[0, 0]
        c_points = tracks[0, 1]
        c_vis = visibility[0, 1]
        c_conf = confidence[0, 1]
        valid = (
            np.isfinite(q_points).all(axis=1)
            & np.isfinite(c_points).all(axis=1)
            & (c_vis >= float(self.vggt_config.visibility_threshold))
            & (c_conf >= float(self.vggt_config.confidence_threshold))
            & (c_points[:, 0] >= 0)
            & (c_points[:, 0] < width)
            & (c_points[:, 1] >= 0)
            & (c_points[:, 1] < height)
        )
        return VGGTMatchResult(
            mkpts0=np.asarray(q_points[valid], dtype=np.float32),
            mkpts1=np.asarray(c_points[valid], dtype=np.float32),
            num_query_points=int(len(query_points)),
            num_visible_tracks=int(np.count_nonzero(valid)),
            mean_track_confidence=float(np.mean(c_conf[valid])) if np.any(valid) else 0.0,
            mean_visibility=float(np.mean(c_vis[valid])) if np.any(valid) else 0.0,
        )

    def _load_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
            torch.set_grad_enabled(False)
        return self._torch

    def _load_image_loader(self):
        if self._load_and_preprocess_images is not None:
            return self._load_and_preprocess_images
        self._ensure_import_path()
        from vggt.utils.load_fn import load_and_preprocess_images

        self._load_and_preprocess_images = load_and_preprocess_images
        return load_and_preprocess_images

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.vggt_root.is_dir():
            raise FileNotFoundError(f"VGGT repository not found: {self.vggt_root}")
        self._ensure_import_path()
        torch = self._load_torch()
        from vggt.models.vggt import VGGT

        model = VGGT.from_pretrained(str(self.vggt_config.model_path)).to(self.device)
        model.eval()
        self._model = model
        return model

    def _ensure_import_path(self) -> None:
        root = str(self.vggt_root)
        if root not in sys.path:
            sys.path.insert(0, root)


def score_benchmark_pairs(
    benchmark_root: Path,
    matcher,
    *,
    ransac_config: RansacConfig,
    ransac_summary_fn: Optional[
        Callable[[np.ndarray, np.ndarray, RansacConfig], Mapping[str, object]]
    ] = None,
) -> list[dict[str, object]]:
    benchmark_root = Path(benchmark_root)
    pairs = list(_read_jsonl(benchmark_root / "benchmark_pairs.jsonl"))
    cache_by_sequence = _cache_by_sequence(benchmark_root)
    summarize = compute_ransac_summary if ransac_summary_fn is None else ransac_summary_fn

    rows: list[dict[str, object]] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        try:
            cache_root = cache_by_sequence[_sequence_key(pair)]
            query_image = _pair_image_path(cache_root, pair, "query_image")
            candidate_image = _pair_image_path(cache_root, pair, "candidate_image")
            match = matcher.match(query_image, candidate_image)
            if len(match.mkpts0) < int(ransac_config.min_matches):
                summary = {
                    "num_matches": int(len(match.mkpts0)),
                    "num_inliers": 0,
                    "inlier_ratio": 0.0,
                }
            else:
                summary = summarize(match.mkpts0, match.mkpts1, ransac_config)
            num_matches = int(summary["num_matches"])
            num_inliers = int(summary["num_inliers"])
            inlier_ratio = float(summary["inlier_ratio"])
            rows.append(
                {
                    "pair_id": pair_id,
                    "score": float(num_inliers),
                    "status": "ok",
                    "num_matches": num_matches,
                    "num_inliers": num_inliers,
                    "inlier_ratio": inlier_ratio,
                    "num_query_points": int(match.num_query_points),
                    "num_visible_tracks": int(match.num_visible_tracks),
                    "mean_track_confidence": float(match.mean_track_confidence),
                    "mean_visibility": float(match.mean_visibility),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pair_id": pair_id,
                    "score": None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    _validate_score_row_order(pairs, rows)
    return rows


def compute_ransac_summary(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    config: RansacConfig,
) -> dict[str, object]:
    points0 = np.asarray(mkpts0, dtype=np.float32)
    points1 = np.asarray(mkpts1, dtype=np.float32)
    if points0.ndim != 2 or points0.shape[1] != 2:
        raise ValueError("mkpts0 must have shape (N, 2)")
    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError("mkpts1 must have shape (N, 2)")
    if len(points0) != len(points1):
        raise ValueError("mkpts0 and mkpts1 must have matching lengths")

    num_matches = int(len(points0))
    if num_matches < int(config.min_matches):
        return {"num_matches": num_matches, "num_inliers": 0, "inlier_ratio": 0.0}

    method = _ransac_method(config.method)
    _, mask = cv2.findFundamentalMat(
        points0,
        points1,
        method,
        float(config.threshold_px),
        float(config.confidence),
        int(config.max_iters),
    )
    if mask is None:
        return {"num_matches": num_matches, "num_inliers": 0, "inlier_ratio": 0.0}
    inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
    if len(inlier_mask) != num_matches:
        return {"num_matches": num_matches, "num_inliers": 0, "inlier_ratio": 0.0}
    num_inliers = int(np.count_nonzero(inlier_mask))
    return {
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": float(num_inliers / num_matches) if num_matches else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--model-path", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="pad")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--border-px", type=int, default=8)
    parser.add_argument("--visibility-threshold", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--max-query-points", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--ransac-method",
        choices=("usac_magsac", "ransac", "lmeds"),
        default="usac_magsac",
    )
    parser.add_argument("--ransac-threshold-px", type=float, default=1.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iters", type=int, default=10000)
    parser.add_argument("--min-matches", type=int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(effective_argv)
    benchmark_root = Path(args.benchmark_root)
    vggt_config = VGGTConfig(
        model_path=str(args.model_path),
        preprocess_mode=str(args.preprocess_mode),
        grid_size=int(args.grid_size),
        border_px=int(args.border_px),
        visibility_threshold=float(args.visibility_threshold),
        confidence_threshold=float(args.confidence_threshold),
        max_query_points=int(args.max_query_points),
        dtype=str(args.dtype),
    )
    ransac_config = RansacConfig(
        method=str(args.ransac_method),
        threshold_px=float(args.ransac_threshold_px),
        confidence=float(args.ransac_confidence),
        max_iters=int(args.ransac_max_iters),
        min_matches=int(args.min_matches),
    )
    matcher = VGGTTrackMatcher(
        vggt_root=Path(args.vggt_root),
        device=str(args.device),
        vggt_config=vggt_config,
    )
    matcher.prepare()
    rows = score_benchmark_pairs(
        benchmark_root,
        matcher,
        ransac_config=ransac_config,
    )
    _write_score_outputs(
        benchmark_root,
        rows,
        command=[str(Path(__file__).resolve()), *effective_argv],
        manifest_extra={
            "backend": "VGGT-track",
            "device": str(args.device),
            "vggt_root": str(Path(args.vggt_root).resolve()),
            "vggt_root_git_commit": _optional_git_commit(Path(args.vggt_root)),
            "model_path": str(args.model_path),
            "preprocess_mode": str(args.preprocess_mode),
            "target_size": 518,
            "grid_size": int(args.grid_size),
            "border_px": int(args.border_px),
            "visibility_threshold": float(args.visibility_threshold),
            "confidence_threshold": float(args.confidence_threshold),
            "max_query_points": int(args.max_query_points),
            "dtype": str(args.dtype),
            "ransac_method": str(args.ransac_method),
            "ransac_threshold_px": float(args.ransac_threshold_px),
            "ransac_confidence": float(args.ransac_confidence),
            "ransac_max_iters": int(args.ransac_max_iters),
            "min_matches": int(args.min_matches),
        },
    )
    return 0


def _make_grid_points(
    *,
    width: int,
    height: int,
    grid_size: int,
    border_px: int,
    max_points: int,
) -> np.ndarray:
    if width <= 2 * border_px or height <= 2 * border_px:
        return np.zeros((0, 2), dtype=np.float32)
    x_count = max(2, int(round((width - 2 * border_px) / float(grid_size))) + 1)
    y_count = max(2, int(round((height - 2 * border_px) / float(grid_size))) + 1)
    xs = np.linspace(border_px, width - border_px - 1, x_count, dtype=np.float32)
    ys = np.linspace(border_px, height - border_px - 1, y_count, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)]).astype(np.float32)
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
        points = points[indices]
    return points


def _resolve_autocast_dtype(torch, device: str, dtype_name: str):
    if not str(device).startswith("cuda") or dtype_name == "float32":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    capability_major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if capability_major >= 8 else torch.float16


def _autocast_context(torch, device: str, dtype):
    if not str(device).startswith("cuda") or dtype is None:
        from contextlib import nullcontext

        return nullcontext()
    return torch.cuda.amp.autocast(dtype=dtype)


def _ransac_method(name: str) -> int:
    normalized = str(name).lower()
    if normalized == "usac_magsac" and hasattr(cv2, "USAC_MAGSAC"):
        return int(cv2.USAC_MAGSAC)
    if normalized == "usac_magsac":
        return int(cv2.RANSAC)
    if normalized == "ransac":
        return int(cv2.RANSAC)
    if normalized == "lmeds":
        return int(cv2.LMEDS)
    raise ValueError(f"unsupported RANSAC method: {name}")


def _cache_by_sequence(benchmark_root: Path) -> dict[str, Path]:
    manifest = _read_json(Path(benchmark_root) / "manifest.json")
    rows = manifest.get("sequences")
    if not isinstance(rows, list):
        raise ValueError("manifest sequences must be a list")
    output: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest sequence rows must be mappings")
        output[_sequence_key(row)] = Path(str(row["cache"]))
    return output


def _pair_image_path(cache_root: Path, pair: Mapping[str, object], field: str) -> Path:
    raw_value = pair.get(field)
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"pair {pair.get('pair_id')} missing {field}")
    path = Path(raw_value)
    return path if path.is_absolute() else Path(cache_root) / path


def _sequence_key(row: Mapping[str, object]) -> str:
    return f"{row['dataset']}/{row['platform']}/{row['sequence']}"


def _validate_score_row_order(
    pairs: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> None:
    expected = [str(pair["pair_id"]) for pair in pairs]
    observed = [str(row["pair_id"]) for row in rows]
    if observed != expected:
        raise ValueError("VGGT score rows must preserve benchmark pair order")


def _write_score_outputs(
    benchmark_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    command: Sequence[str],
    manifest_extra: Mapping[str, object],
) -> None:
    benchmark_root = Path(benchmark_root)
    score_path = benchmark_root / "scores" / "vggt.jsonl"
    score_content = _jsonl_content(rows)
    manifest = {
        "method": "VGGT track RANSAC inlier count",
        "evaluation_name": "VGGT-track",
        "score_file": "scores/vggt.jsonl",
        "pair_manifest_sha256": _sha256_file(benchmark_root / "benchmark_pairs.jsonl"),
        "score_file_sha256": hashlib.sha256(score_content.encode("utf-8")).hexdigest(),
        "source_commit": _source_commit(),
        "command": list(command),
    }
    manifest.update(manifest_extra)
    _publish_output_set(
        {
            score_path: score_content,
            score_path.with_suffix(".manifest.json"): _json_content(manifest),
        }
    )


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _jsonl_content(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(_sanitize_json_value(dict(row)), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _json_content(payload: Mapping[str, object]) -> str:
    return json.dumps(_sanitize_json_value(dict(payload)), indent=2, sort_keys=True) + "\n"


def _sanitize_json_value(value):
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Mapping):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    return value


def _publish_output_set(payloads: Mapping[Path, str]) -> None:
    normalized = {Path(path): content for path, content in payloads.items()}
    missing: list[tuple[Path, str]] = []
    for path, content in sorted(normalized.items(), key=lambda item: item[0].name):
        if not path.exists():
            missing.append((path, content))
            continue
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
    if not missing:
        return

    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for path, content in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temp_path = Path(temp_name)
            staged[path] = temp_path
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for path, temp_path in staged.items():
            try:
                os.link(temp_path, path)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite changed output: {path}") from exc
            published.append(path)
        for temp_path in staged.values():
            temp_path.unlink()
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(LOOPANYTHING_ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _optional_git_commit(path: Path) -> Optional[str]:
    if not Path(path).exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
