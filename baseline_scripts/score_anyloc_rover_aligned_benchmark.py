#!/usr/bin/env python3
"""Score frozen ROVER-aligned benchmark pairs with AnyLoc-VLAD-DINOv2 retrieval."""

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
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOPANYTHING_ROOT.parent
DEFAULT_ANYLOC_ROOT = WORKSPACE_ROOT / "AnyLoc"


@dataclass(frozen=True)
class AnyLocConfig:
    model_type: str = "dinov2_vitg14"
    desc_layer: int = 31
    desc_facet: str = "value"
    num_clusters: int = 32
    domain: str = "urban"
    max_img_size: int = 1024


@dataclass(frozen=True)
class AnyLocDescriptorResult:
    descriptor: np.ndarray
    num_patches: int
    resized_height: int
    resized_width: int


class AnyLocDescriptorBackend(Protocol):
    def describe(self, image_path: Path) -> AnyLocDescriptorResult:
        ...


class AnyLocImageDescriptor:
    def __init__(
        self,
        *,
        anyloc_root: Path,
        cache_root: Path,
        device: str,
        config: AnyLocConfig,
    ) -> None:
        self.anyloc_root = Path(anyloc_root)
        self.cache_root = Path(cache_root)
        self.device = str(device)
        self.config = config
        self._extractor = None
        self._vlad = None
        self._base_tf = None
        self._torch = None
        self._tvf = None
        self._T = None
        self._Image = None

    def prepare(self) -> None:
        if not self.anyloc_root.is_dir():
            raise FileNotFoundError(f"AnyLoc repository not found: {self.anyloc_root}")
        vocabulary_path = self.vocabulary_path
        if not vocabulary_path.is_file():
            raise FileNotFoundError(f"AnyLoc vocabulary not found: {vocabulary_path}")
        self._ensure_import_path()
        self._load_components()

    @property
    def vocabulary_path(self) -> Path:
        ext_specifier = (
            Path(str(self.config.model_type))
            / f"l{int(self.config.desc_layer)}_{self.config.desc_facet}_c{int(self.config.num_clusters)}"
            / str(self.config.domain)
            / "c_centers.pt"
        )
        return self.cache_root / "vocabulary" / ext_specifier

    def describe(self, image_path: Path) -> AnyLocDescriptorResult:
        self.prepare()
        torch = self._torch
        tvf = self._tvf
        T = self._T
        Image = self._Image
        assert torch is not None
        assert tvf is not None
        assert T is not None
        assert Image is not None
        assert self._extractor is not None
        assert self._vlad is not None

        with torch.no_grad():
            pil_img = Image.open(image_path).convert("RGB")
            img_pt = self._base_tf(pil_img).to(self.device)
            if max(img_pt.shape[-2:]) > int(self.config.max_img_size):
                _, height, width = img_pt.shape
                if height == max(img_pt.shape[-2:]):
                    width = int(width * int(self.config.max_img_size) / height)
                    height = int(self.config.max_img_size)
                else:
                    height = int(height * int(self.config.max_img_size) / width)
                    width = int(self.config.max_img_size)
                img_pt = T.resize(
                    img_pt,
                    (height, width),
                    interpolation=T.InterpolationMode.BICUBIC,
                )
            _, height, width = img_pt.shape
            height_new = (height // 14) * 14
            width_new = (width // 14) * 14
            if height_new <= 0 or width_new <= 0:
                raise ValueError(f"image too small after preprocessing: {image_path}")
            img_pt = tvf.CenterCrop((height_new, width_new))(img_pt)[None, ...]
            patch_descriptors = self._extractor(img_pt)
            vlad = self._vlad.generate(patch_descriptors.cpu().squeeze())
            descriptor = vlad.detach().cpu().numpy().astype(np.float32, copy=False)
        return AnyLocDescriptorResult(
            descriptor=descriptor,
            num_patches=int(patch_descriptors.shape[1]),
            resized_height=int(height_new),
            resized_width=int(width_new),
        )

    def _load_components(self) -> None:
        if self._extractor is not None and self._vlad is not None:
            return
        import torch
        from PIL import Image
        from torchvision import transforms as tvf
        from torchvision.transforms import functional as T

        self._torch = torch
        self._tvf = tvf
        self._T = T
        self._Image = Image
        torch.set_grad_enabled(False)

        self._ensure_import_path()
        from demo.utilities import DinoV2ExtractFeatures, VLAD

        self._base_tf = tvf.Compose(
            [
                tvf.ToTensor(),
                tvf.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._extractor = DinoV2ExtractFeatures(
            str(self.config.model_type),
            int(self.config.desc_layer),
            str(self.config.desc_facet),
            device=self.device,
        )
        self._vlad = VLAD(
            int(self.config.num_clusters),
            desc_dim=None,
            cache_dir=str(self.vocabulary_path.parent),
        )
        self._vlad.fit(None)

    def _ensure_import_path(self) -> None:
        root = str(self.anyloc_root)
        if root not in sys.path:
            sys.path.insert(0, root)


def score_benchmark_pairs(
    benchmark_root: Path,
    descriptor_backend: AnyLocDescriptorBackend,
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

    descriptors: dict[Path, AnyLocDescriptorResult] = {}
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
                    "num_patches_query": int(query.num_patches),
                    "num_patches_candidate": int(candidate.num_patches),
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
    parser.add_argument("--anyloc-root", type=Path, default=DEFAULT_ANYLOC_ROOT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", default="dinov2_vitg14")
    parser.add_argument("--desc-layer", type=int, default=31)
    parser.add_argument("--desc-facet", default="value")
    parser.add_argument("--num-clusters", type=int, default=32)
    parser.add_argument("--domain", choices=("urban", "indoor", "aerial"), default="urban")
    parser.add_argument("--max-img-size", type=int, default=1024)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(effective_argv)
    anyloc_root = Path(args.anyloc_root)
    cache_root = Path(args.cache_root) if args.cache_root is not None else anyloc_root / "cache"
    config = AnyLocConfig(
        model_type=str(args.model_type),
        desc_layer=int(args.desc_layer),
        desc_facet=str(args.desc_facet),
        num_clusters=int(args.num_clusters),
        domain=str(args.domain),
        max_img_size=int(args.max_img_size),
    )
    descriptor = AnyLocImageDescriptor(
        anyloc_root=anyloc_root,
        cache_root=cache_root,
        device=str(args.device),
        config=config,
    )
    descriptor.prepare()
    rows = score_benchmark_pairs(Path(args.benchmark_root), descriptor)
    _write_score_outputs(
        Path(args.benchmark_root),
        rows,
        command=[str(Path(__file__).resolve()), *effective_argv],
        manifest_extra={
            "backend": "AnyLoc-VLAD-DINOv2",
            "device": str(args.device),
            "anyloc_root": str(anyloc_root.resolve()),
            "anyloc_root_git_commit": _optional_git_commit(anyloc_root),
            "cache_root": str(cache_root.resolve()),
            "vocabulary_path": str(descriptor.vocabulary_path.resolve()),
            "vocabulary_sha256": _optional_file_sha256(descriptor.vocabulary_path),
            "model_type": str(config.model_type),
            "desc_layer": int(config.desc_layer),
            "desc_facet": str(config.desc_facet),
            "num_clusters": int(config.num_clusters),
            "domain": str(config.domain),
            "max_img_size": int(config.max_img_size),
            "preprocessing": "ImageNet normalization; resize max edge if needed; center-crop to multiples of 14",
            "score_definition": "dot product of L2-normalized AnyLoc VLAD descriptors",
        },
    )
    return 0


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
        raise ValueError("AnyLoc score rows must preserve benchmark pair order")


def _write_score_outputs(
    benchmark_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    command: Sequence[str],
    manifest_extra: Mapping[str, object],
) -> None:
    benchmark_root = Path(benchmark_root)
    score_path = benchmark_root / "scores" / "anyloc.jsonl"
    score_content = _jsonl_content(rows)
    manifest = {
        "method": "AnyLoc-VLAD-DINOv2 global descriptor similarity",
        "evaluation_name": "AnyLoc-VLAD-DINOv2",
        "score_file": "scores/anyloc.jsonl",
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
