#!/usr/bin/env python3
"""Score frozen ROVER-aligned benchmark pairs with MAST3R geometric verification."""

from __future__ import annotations

import argparse
import contextlib
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
DEFAULT_MAST3R_ROOT = WORKSPACE_ROOT / "mast3r"
DEFAULT_MAST3R_CKPT = (
    DEFAULT_MAST3R_ROOT
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)


@dataclass(frozen=True)
class RansacConfig:
    method: str = "usac_magsac"
    threshold_px: float = 1.0
    confidence: float = 0.999
    max_iters: int = 10000
    min_matches: int = 8


@dataclass(frozen=True)
class MAST3RConfig:
    image_size: int = 512
    subsample: int = 8
    border_px: int = 3
    batch_size: int = 1
    block_size: int = 2**13


@dataclass(frozen=True)
class MAST3RMatchResult:
    mkpts0: np.ndarray
    mkpts1: np.ndarray
    num_reciprocal_matches: int
    mean_match_confidence: float


class MAST3RPairMatcher:
    def __init__(
        self,
        *,
        mast3r_root: Path,
        checkpoint_path: Path,
        device: str,
        mast3r_config: MAST3RConfig,
    ) -> None:
        self.mast3r_root = Path(mast3r_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.mast3r_config = mast3r_config
        self._model = None

    def match(self, query_image: Path, candidate_image: Path) -> MAST3RMatchResult:
        import torch

        if not self.mast3r_root.is_dir():
            raise FileNotFoundError(f"MAST3R repository not found: {self.mast3r_root}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"MAST3R checkpoint not found: {self.checkpoint_path}")

        self._ensure_import_paths()
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.fast_nn import fast_reciprocal_NNs

        model = self._load_model()
        images = load_images(
            [str(query_image), str(candidate_image)],
            size=int(self.mast3r_config.image_size),
            verbose=False,
        )
        with torch.no_grad():
            output = inference(
                [tuple(images)],
                model,
                self.device,
                batch_size=int(self.mast3r_config.batch_size),
                verbose=False,
            )

        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]
        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()
        matches0, matches1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=int(self.mast3r_config.subsample),
            device=self.device,
            dist="dot",
            block_size=int(self.mast3r_config.block_size),
        )
        matches0 = np.asarray(matches0, dtype=np.float32)
        matches1 = np.asarray(matches1, dtype=np.float32)
        matches0, matches1 = _filter_border_matches(
            matches0,
            matches1,
            _true_shape_hw(view1),
            _true_shape_hw(view2),
            border_px=int(self.mast3r_config.border_px),
        )
        mean_confidence = _mean_match_confidence(pred1, pred2, matches0, matches1)
        return MAST3RMatchResult(
            mkpts0=matches0,
            mkpts1=matches1,
            num_reciprocal_matches=int(len(matches0)),
            mean_match_confidence=mean_confidence,
        )

    def _ensure_import_paths(self) -> None:
        mast3r_path = str(self.mast3r_root)
        dust3r_path = str(self.mast3r_root / "dust3r")
        for path in (dust3r_path, mast3r_path):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._ensure_import_paths()
        import torch
        from mast3r.model import AsymmetricMASt3R

        with patch_torch_load_for_trusted_mast3r_checkpoint(torch):
            model = AsymmetricMASt3R.from_pretrained(str(self.checkpoint_path)).to(self.device)
        model.eval()
        torch.set_grad_enabled(False)
        self._model = model
        return model


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
                    "num_reciprocal_matches": int(match.num_reciprocal_matches),
                    "mean_match_confidence": float(match.mean_match_confidence),
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


def trusted_torch_load_wrapper(load_fn):
    def wrapped(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return load_fn(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def patch_torch_load_for_trusted_mast3r_checkpoint(torch_module):
    original_load = torch_module.load
    torch_module.load = trusted_torch_load_wrapper(original_load)
    try:
        yield
    finally:
        torch_module.load = original_load


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--mast3r-root", type=Path, default=DEFAULT_MAST3R_ROOT)
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_MAST3R_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--border-px", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2**13)
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
    mast3r_config = MAST3RConfig(
        image_size=int(args.image_size),
        subsample=int(args.subsample),
        border_px=int(args.border_px),
        batch_size=int(args.batch_size),
        block_size=int(args.block_size),
    )
    ransac_config = RansacConfig(
        method=str(args.ransac_method),
        threshold_px=float(args.ransac_threshold_px),
        confidence=float(args.ransac_confidence),
        max_iters=int(args.ransac_max_iters),
        min_matches=int(args.min_matches),
    )
    matcher = MAST3RPairMatcher(
        mast3r_root=Path(args.mast3r_root),
        checkpoint_path=Path(args.ckpt_path),
        device=str(args.device),
        mast3r_config=mast3r_config,
    )
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
            "backend": "MAST3R",
            "device": str(args.device),
            "mast3r_root": str(Path(args.mast3r_root).resolve()),
            "mast3r_root_git_commit": _optional_git_commit(Path(args.mast3r_root)),
            "mast3r_checkpoint": str(Path(args.ckpt_path).resolve()),
            "mast3r_checkpoint_sha256": _optional_file_sha256(Path(args.ckpt_path)),
            "image_size": int(args.image_size),
            "subsample": int(args.subsample),
            "border_px": int(args.border_px),
            "batch_size": int(args.batch_size),
            "block_size": int(args.block_size),
            "ransac_method": str(args.ransac_method),
            "ransac_threshold_px": float(args.ransac_threshold_px),
            "ransac_confidence": float(args.ransac_confidence),
            "ransac_max_iters": int(args.ransac_max_iters),
            "min_matches": int(args.min_matches),
        },
    )
    return 0


def _filter_border_matches(
    matches0: np.ndarray,
    matches1: np.ndarray,
    shape0_hw: tuple[int, int],
    shape1_hw: tuple[int, int],
    *,
    border_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(matches0) == 0 or len(matches1) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    height0, width0 = shape0_hw
    height1, width1 = shape1_hw
    valid0 = (
        (matches0[:, 0] >= border_px)
        & (matches0[:, 0] < width0 - border_px)
        & (matches0[:, 1] >= border_px)
        & (matches0[:, 1] < height0 - border_px)
    )
    valid1 = (
        (matches1[:, 0] >= border_px)
        & (matches1[:, 0] < width1 - border_px)
        & (matches1[:, 1] >= border_px)
        & (matches1[:, 1] < height1 - border_px)
    )
    valid = valid0 & valid1
    return matches0[valid].astype(np.float32), matches1[valid].astype(np.float32)


def _true_shape_hw(view: Mapping[str, object]) -> tuple[int, int]:
    raw_shape = view["true_shape"]
    if hasattr(raw_shape, "detach"):
        raw_shape = raw_shape.detach().cpu().numpy()
    shape = np.asarray(raw_shape)
    if shape.ndim == 2:
        shape = shape[0]
    if shape.shape[0] != 2:
        raise ValueError(f"unexpected MAST3R true_shape: {shape}")
    return int(shape[0]), int(shape[1])


def _mean_match_confidence(
    pred1: Mapping[str, object],
    pred2: Mapping[str, object],
    matches0: np.ndarray,
    matches1: np.ndarray,
) -> float:
    if len(matches0) == 0:
        return 0.0
    if "desc_conf" not in pred1 or "desc_conf" not in pred2:
        return 0.0
    conf1 = _tensor_to_numpy(pred1["desc_conf"].squeeze(0))
    conf2 = _tensor_to_numpy(pred2["desc_conf"].squeeze(0))
    xy0 = np.rint(matches0).astype(np.int64)
    xy1 = np.rint(matches1).astype(np.int64)
    valid = (
        (xy0[:, 0] >= 0)
        & (xy0[:, 0] < conf1.shape[1])
        & (xy0[:, 1] >= 0)
        & (xy0[:, 1] < conf1.shape[0])
        & (xy1[:, 0] >= 0)
        & (xy1[:, 0] < conf2.shape[1])
        & (xy1[:, 1] >= 0)
        & (xy1[:, 1] < conf2.shape[0])
    )
    if not np.any(valid):
        return 0.0
    values = np.sqrt(conf1[xy0[valid, 1], xy0[valid, 0]] * conf2[xy1[valid, 1], xy1[valid, 0]])
    return float(np.mean(values))


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


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
        raise ValueError("MAST3R score rows must preserve benchmark pair order")


def _write_score_outputs(
    benchmark_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    command: Sequence[str],
    manifest_extra: Mapping[str, object],
) -> None:
    benchmark_root = Path(benchmark_root)
    score_path = benchmark_root / "scores" / "mast3r.jsonl"
    score_content = _jsonl_content(rows)
    manifest = {
        "method": "MAST3R reciprocal descriptor matches with RANSAC inlier count",
        "evaluation_name": "MAST3R",
        "score_file": "scores/mast3r.jsonl",
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


def _optional_file_sha256(path: Path) -> Optional[str]:
    return _sha256_file(path) if Path(path).is_file() else None


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
