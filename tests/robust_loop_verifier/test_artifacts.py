import json
from pathlib import Path

import pytest
from PIL import Image

from robust_loop_verifier.artifacts import (
    write_json,
    write_metrics_markdown,
    write_triplet_visual_record,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=color).save(path)


def test_write_json_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "metrics.json"

    write_json(out, {"b": 2, "a": {"score": 0.5}})

    assert json.loads(out.read_text(encoding="utf-8")) == {"a": {"score": 0.5}, "b": 2}
    assert out.read_text(encoding="utf-8").startswith('{\n  "a"')


def test_write_metrics_markdown(tmp_path: Path) -> None:
    out = tmp_path / "metrics.md"

    write_metrics_markdown(out, {"salad": {"AP": 0.5, "MR@100P": 0.25}})

    text = out.read_text(encoding="utf-8")
    assert "| method | AP | MR@100P |" in text
    assert "| salad | 0.5000 | 0.2500 |" in text


@pytest.mark.parametrize("method", ["bad|method", "bad\nmethod", "bad\rmethod"])
def test_write_metrics_markdown_rejects_table_breaking_method_names(
    tmp_path: Path, method: str
) -> None:
    with pytest.raises(ValueError, match="method"):
        write_metrics_markdown(
            tmp_path / "metrics.md",
            {method: {"AP": 0.5, "MR@100P": 0.25}},
        )


def test_write_triplet_visual_record(tmp_path: Path) -> None:
    q, c, s = tmp_path / "q.png", tmp_path / "c.png", tmp_path / "s.png"
    _image(q, (255, 0, 0))
    _image(c, (0, 255, 0))
    _image(s, (0, 0, 255))
    out_dir = tmp_path / "visual"

    write_triplet_visual_record(out_dir, "q000010_c000003_s000002", q, c, s)

    record_dir = out_dir / "q000010_c000003_s000002"
    assert (record_dir / "query.png").is_file()
    assert (record_dir / "candidate.png").is_file()
    assert (record_dir / "support.png").is_file()
    assert (record_dir / "triplet.png").is_file()
    with Image.open(record_dir / "triplet.png") as triplet:
        assert triplet.size == (44, 26)


def test_write_triplet_visual_record_validates_inputs_before_creating_record_dir(
    tmp_path: Path,
) -> None:
    q = tmp_path / "q.png"
    c = tmp_path / "missing_c.png"
    s = tmp_path / "missing_s.png"
    _image(q, (255, 0, 0))
    out_dir = tmp_path / "visual"

    with pytest.raises(FileNotFoundError, match="candidate_image"):
        write_triplet_visual_record(out_dir, "missing_inputs", q, c, s)

    assert not (out_dir / "missing_inputs").exists()


def test_write_triplet_visual_record_decodes_inputs_before_creating_record_dir(
    tmp_path: Path,
) -> None:
    q = tmp_path / "q.png"
    c = tmp_path / "not_image.bin"
    s = tmp_path / "s.png"
    _image(q, (255, 0, 0))
    c.write_bytes(b"not an image")
    _image(s, (0, 0, 255))
    out_dir = tmp_path / "visual"

    with pytest.raises(Exception):
        write_triplet_visual_record(out_dir, "invalid_image", q, c, s)

    assert not (out_dir / "invalid_image").exists()


@pytest.mark.parametrize("record_name", ["../escape", "/absolute", "."])
def test_write_triplet_visual_record_rejects_unsafe_record_name(
    tmp_path: Path, record_name: str
) -> None:
    q, c, s = tmp_path / "q.png", tmp_path / "c.png", tmp_path / "s.png"
    _image(q, (255, 0, 0))
    _image(c, (0, 255, 0))
    _image(s, (0, 0, 255))

    with pytest.raises(ValueError, match="record_name"):
        write_triplet_visual_record(tmp_path / "visual", record_name, q, c, s)
