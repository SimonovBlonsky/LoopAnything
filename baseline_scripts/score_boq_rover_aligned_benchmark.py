#!/usr/bin/env python3
"""Score frozen ROVER-aligned benchmark pairs with BoQ-DinoV2 retrieval."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOPANYTHING_ROOT.parent
DEFAULT_BOQ_ROOT = WORKSPACE_ROOT / "Bag-of-Queries"


@dataclass(frozen=True)
class BoQConfig:
    backbone_name: str = "dinov2"
    output_dim: int = 12288
    image_height: int = 322
    image_width: int = 322


@dataclass(frozen=True)
class BoQDescriptorResult:
    descriptor: np.ndarray
    resized_height: int
    resized_width: int


class BoQDescriptorBackend(Protocol):
    def describe(self, image_path: Path) -> BoQDescriptorResult:
        ...


class BoQImageDescriptor:
    def __init__(
        self,
        *,
        boq_root: Path,
        checkpoint_path: Optional[Path],
        device: str,
        config: BoQConfig,
    ) -> None:
        self.boq_root = Path(boq_root)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.device = str(device)
        self.config = config
        self._model = None
        self._transform = None
        self._torch = None
        self._Image = None

    def prepare(self) -> None:
        if not self.boq_root.is_dir():
            raise FileNotFoundError(f"BoQ repository not found: {self.boq_root}")
        if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"BoQ checkpoint not found: {self.checkpoint_path}")
        self._ensure_import_paths()
        self._load_components()

    def describe(self, image_path: Path) -> BoQDescriptorResult:
        self.prepare()
        torch = self._torch
        Image = self._Image
        assert torch is not None
        assert Image is not None
        assert self._model is not None
        assert self._transform is not None

        with torch.no_grad():
            pil_img = Image.open(image_path).convert("RGB")
            tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
            output = self._model(tensor)
            descriptor = output[0] if isinstance(output, tuple) else output
            descriptor = descriptor.detach().float().cpu().numpy().reshape(-1)
        return BoQDescriptorResult(
            descriptor=descriptor.astype(np.float32, copy=False),
            resized_height=int(self.config.image_height),
            resized_width=int(self.config.image_width),
        )

    def _load_components(self) -> None:
        if self._model is not None and self._transform is not None:
            return
        import torch
        import torchvision.transforms as T
        from PIL import Image

        self._torch = torch
        self._Image = Image
        torch.set_grad_enabled(False)
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (int(self.config.image_height), int(self.config.image_width)),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._model = self._load_model(torch).eval().to(self.device)

    def _load_model(self, torch):
        if self.checkpoint_path is None:
            hubconf = _load_boq_hubconf(self.boq_root)
            return hubconf.get_trained_boq(
                backbone_name=str(self.config.backbone_name),
                output_dim=int(self.config.output_dim),
            )
        model = _build_local_boq_model(
            self.boq_root,
            backbone_name=str(self.config.backbone_name),
            output_dim=int(self.config.output_dim),
        )
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        model.load_state_dict(state_dict)
        return model

    def _ensure_import_paths(self) -> None:
        for path in (self.boq_root / "src", self.boq_root):
            raw = str(path)
            if raw not in sys.path:
                sys.path.insert(0, raw)


def score_benchmark_pairs(
    benchmark_root: Path,
    descriptor_backend: BoQDescriptorBackend,
) -> list[dict[str, object]]:
    benchmark_root = Path(benchmark_root)
    pairs = list(_read_jsonl(benchmark_root / "benchmark_pairs.jsonl"))
    cache_by_sequence = _cache_by_sequence(benchmark_root)
    resolved_pairs: list[tuple[Mapping[str, object], Path, Path]] = []
    unique_images: dict[Path, None] = {}

    for pair in pairs:
        cache_root = cache_by_sequence[_sequence_key(pair)]
        query_image = _pair_image_path(cache_root, pair, "query_image")
        candidate_image = _pair_image_path(cache_root, pair, "candidate_image")
        resolved_pairs.append((pair, query_image, candidate_image))
        unique_images[query_image] = None
        unique_images[candidate_image] = None

    descriptors: dict[Path, BoQDescriptorResult] = {}
    for image_path in unique_images:
        descriptors[image_path] = descriptor_backend.describe(image_path)

    rows: list[dict[str, object]] = []
    for pair, query_image, candidate_image in resolved_pairs:
        pair_id = str(pair["pair_id"])
        try:
            query = descriptors[query_image]
            candidate = descriptors[candidate_image]
            score = _cosine_dot(query.descriptor, candidate.descriptor)
            rows.append(
                {
                    "pair_id": pair_id,
                    "score": score,
                    "status": "ok",
                    "resized_height_query": int(query.resized_height),
                    "resized_width_query": int(query.resized_width),
                    "resized_height_candidate": int(candidate.resized_height),
                    "resized_width_candidate": int(candidate.resized_width),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--boq-root", type=Path, default=DEFAULT_BOQ_ROOT)
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Optional local BoQ checkpoint. If omitted, official torch.hub weights are used.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-name", choices=("dinov2", "resnet50"), default="dinov2")
    parser.add_argument("--output-dim", type=int, default=12288)
    parser.add_argument("--image-height", type=int, default=322)
    parser.add_argument("--image-width", type=int, default=322)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(effective_argv)
    config = BoQConfig(
        backbone_name=str(args.backbone_name),
        output_dim=int(args.output_dim),
        image_height=int(args.image_height),
        image_width=int(args.image_width),
    )
    descriptor = BoQImageDescriptor(
        boq_root=Path(args.boq_root),
        checkpoint_path=Path(args.ckpt_path) if args.ckpt_path is not None else None,
        device=str(args.device),
        config=config,
    )
    descriptor.prepare()
    rows = score_benchmark_pairs(Path(args.benchmark_root), descriptor)
    score_name = f"boq_{config.backbone_name}"
    _write_score_outputs(
        Path(args.benchmark_root),
        rows,
        score_name=score_name,
        command=[str(Path(__file__).resolve()), *effective_argv],
        manifest_extra={
            "backend": "BoQ",
            "evaluation_name": f"BoQ-{config.backbone_name}",
            "device": str(args.device),
            "boq_root": str(Path(args.boq_root).resolve()),
            "boq_root_git_commit": _optional_git_commit(Path(args.boq_root)),
            "checkpoint_path": str(Path(args.ckpt_path).resolve())
            if args.ckpt_path is not None
            else None,
            "checkpoint_sha256": _optional_file_sha256(Path(args.ckpt_path))
            if args.ckpt_path is not None
            else None,
            "backbone_name": str(config.backbone_name),
            "output_dim": int(config.output_dim),
            "image_height": int(config.image_height),
            "image_width": int(config.image_width),
            "preprocessing": "ToTensor; bicubic resize; ImageNet normalization",
            "score_definition": "dot product of L2-normalized BoQ global descriptors",
        },
    )
    return 0


def _build_local_boq_model(boq_root: Path, *, backbone_name: str, output_dim: int):
    _ensure_boq_import_paths(boq_root)
    hubconf = _load_boq_hubconf(boq_root)
    if backbone_name not in hubconf.AVAILABLE_BACKBONES:
        raise ValueError(f"unsupported BoQ backbone: {backbone_name}")
    if int(output_dim) not in hubconf.AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(f"unsupported BoQ output_dim {output_dim} for {backbone_name}")

    from backbones import DinoV2, ResNet
    from boq import BoQ

    if "dinov2" in backbone_name:
        backbone = DinoV2()
        aggregator = BoQ(
            in_channels=backbone.out_channels,
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=int(output_dim) // 384,
        )
    elif "resnet" in backbone_name:
        backbone = ResNet(backbone_name=backbone_name, crop_last_block=True)
        aggregator = BoQ(
            in_channels=backbone.out_channels,
            proj_channels=512,
            num_queries=64,
            num_layers=2,
            row_dim=int(output_dim) // 512,
        )
    else:
        raise ValueError(f"unsupported BoQ backbone: {backbone_name}")
    return hubconf.VPRModel(backbone=backbone, aggregator=aggregator)


def _load_boq_hubconf(boq_root: Path):
    _ensure_boq_import_paths(boq_root)
    module_path = Path(boq_root) / "hubconf.py"
    spec = importlib.util.spec_from_file_location("_loopanything_boq_hubconf", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load BoQ hubconf: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_boq_import_paths(boq_root: Path) -> None:
    for path in (Path(boq_root) / "src", Path(boq_root)):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)


def _cosine_dot(query: np.ndarray, candidate: np.ndarray) -> float:
    query_vec = np.asarray(query, dtype=np.float32).reshape(-1)
    candidate_vec = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if query_vec.shape != candidate_vec.shape:
        raise ValueError(
            f"descriptor shapes must match, got {query_vec.shape} and {candidate_vec.shape}"
        )
    query_norm = float(np.linalg.norm(query_vec))
    candidate_norm = float(np.linalg.norm(candidate_vec))
    if query_norm <= 0.0 or candidate_norm <= 0.0:
        return 0.0
    score = float(np.dot(query_vec / query_norm, candidate_vec / candidate_norm))
    return score if math.isfinite(score) else 0.0


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
        raise ValueError("BoQ score rows must preserve benchmark pair order")


def _write_score_outputs(
    benchmark_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    score_name: str,
    command: Sequence[str],
    manifest_extra: Mapping[str, object],
) -> None:
    benchmark_root = Path(benchmark_root)
    score_path = benchmark_root / "scores" / f"{score_name}.jsonl"
    score_content = _jsonl_content(rows)
    manifest = {
        "method": "BoQ global descriptor similarity",
        "score_file": f"scores/{score_name}.jsonl",
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
