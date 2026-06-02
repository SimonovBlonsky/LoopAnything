import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root / "robust_loop_verification_scripts" / "export_positive_debug_matches.py"
    )
    spec = importlib.util.spec_from_file_location("export_positive_debug_matches", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_positive_debug_matches_copies_query_and_all_positive_candidates(tmp_path):
    module = _load_script_module()
    cache_dir = tmp_path / "cache"
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True)
    for idx in range(4):
        Image.new("RGB", (8, 8), color=(idx, 0, 0)).save(image_dir / f"{idx:06d}.png")

    with (cache_dir / "keyframes.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(4):
            handle.write(json.dumps({"idx": idx, "image_path": f"images/{idx:06d}.png"}))
            handle.write("\n")
    (cache_dir / "positives.jsonl").write_text(
        '{"query_idx":0,"positive_indices":[]}\n'
        '{"query_idx":3,"positive_indices":[1,2]}\n',
        encoding="utf-8",
    )

    summary = module.export_positive_debug_matches(cache_dir)

    assert summary["exported_match_count"] == 1
    match_dir = cache_dir / "positive_debug" / "q000003__c000001-000002"
    assert (match_dir / "query_000003.png").is_file()
    assert (match_dir / "candidate_000001.png").is_file()
    assert (match_dir / "candidate_000002.png").is_file()
    meta = json.loads((match_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["query_idx"] == 3
    assert meta["positive_indices"] == [1, 2]
    assert (cache_dir / "positive_debug" / "summary.json").is_file()


def test_export_positive_debug_matches_uses_bounded_directory_names(tmp_path):
    module = _load_script_module()
    cache_dir = tmp_path / "cache"
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_dir / "000100.png")
    candidate_indices = list(range(60))
    for idx in candidate_indices:
        Image.new("RGB", (8, 8), color=(idx, 0, 0)).save(image_dir / f"{idx:06d}.png")

    rows = [{"idx": 100, "image_path": "images/000100.png"}] + [
        {"idx": idx, "image_path": f"images/{idx:06d}.png"} for idx in candidate_indices
    ]
    with (cache_dir / "keyframes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    (cache_dir / "positives.jsonl").write_text(
        json.dumps({"query_idx": 100, "positive_indices": candidate_indices}) + "\n",
        encoding="utf-8",
    )

    module.export_positive_debug_matches(cache_dir)

    match_dirs = list((cache_dir / "positive_debug").glob("q000100__n060__c*"))
    assert len(match_dirs) == 1
    match_dir = match_dirs[0]
    assert len(match_dir.name) < 128
    assert "n060" in match_dir.name
    meta = json.loads((match_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["positive_indices"] == candidate_indices
