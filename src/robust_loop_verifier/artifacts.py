"""Artifact writers for offline robust loop verifier runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw


def write_json(path: Path, data: Mapping[str, object]) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_metrics_markdown(path: Path, metrics: Mapping[str, Mapping[str, float]]) -> None:
    """Write retrieval metrics as a compact Markdown table."""

    for method in metrics:
        if "|" in method or "\n" in method or "\r" in method:
            raise ValueError(f"method name breaks Markdown table: {method!r}")

    lines = [
        "| method | AP | MR@100P |",
        "| --- | ---: | ---: |",
    ]
    for method, values in metrics.items():
        lines.append(f"| {method} | {values['AP']:.4f} | {values['MR@100P']:.4f} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_triplet_visual_record(
    root: Path,
    record_name: str,
    query_image: Path,
    candidate_image: Path,
    support_image: Path,
) -> None:
    """Copy triplet image components and write a labeled side-by-side preview."""

    record_dir = _safe_record_dir(root, record_name)
    inputs = [
        ("query_image", Path(query_image)),
        ("candidate_image", Path(candidate_image)),
        ("support_image", Path(support_image)),
    ]
    for label, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not an existing regular file: {path}")

    panels = []
    try:
        for label, path in [
            ("query", Path(query_image)),
            ("candidate", Path(candidate_image)),
            ("support", Path(support_image)),
        ]:
            with Image.open(path) as image:
                panel = image.convert("RGB")
                panel.load()
                panels.append((label, panel))

        record_dir.mkdir(parents=True, exist_ok=False)
        outputs = [
            ("query", Path(query_image), record_dir / "query.png"),
            ("candidate", Path(candidate_image), record_dir / "candidate.png"),
            ("support", Path(support_image), record_dir / "support.png"),
        ]
        for _, src, dst in outputs:
            shutil.copyfile(src, dst)

        _write_triplet(record_dir / "triplet.png", panels)
    finally:
        for _, image in panels:
            image.close()


def _safe_record_dir(root: Path, record_name: str) -> Path:
    record_path = Path(record_name)
    if (
        record_path.is_absolute()
        or not record_name
        or record_path == Path(".")
        or ".." in record_path.parts
    ):
        raise ValueError(f"unsafe record_name: {record_name!r}")
    return Path(root) / record_path


def _write_triplet(path: Path, panels: list[tuple[str, Image.Image]]) -> None:
    label_height = 18
    gap = 4
    width = sum(image.width for _, image in panels) + gap * (len(panels) - 1)
    height = label_height + max(image.height for _, image in panels)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, image in panels:
        draw.text((x + 2, 2), label, fill=(0, 0, 0))
        canvas.paste(image, (x, label_height))
        x += image.width + gap

    canvas.save(path)
