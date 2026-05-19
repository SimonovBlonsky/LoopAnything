from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "baseline_scripts" / "netvlad_retrieval_pipeline.py"
    spec = importlib.util.spec_from_file_location("netvlad_retrieval_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_sequence_cache(root: Path, platform: str, sequence: str) -> Path:
    sequence_cache = root / platform / sequence
    (sequence_cache / "images").mkdir(parents=True)
    keyframes = []
    for idx in range(5):
        image_path = f"images/{idx:06d}.png"
        (sequence_cache / image_path).write_bytes(b"fake-image")
        keyframes.append({"idx": idx, "image_path": image_path})
    _write_jsonl(sequence_cache / "keyframes.jsonl", keyframes)
    _write_jsonl(
        sequence_cache / "positives.jsonl",
        [
            {"query_idx": 0, "positive_indices": []},
            {"query_idx": 1, "positive_indices": []},
            {"query_idx": 2, "positive_indices": []},
            {"query_idx": 3, "positive_indices": [0]},
            {"query_idx": 4, "positive_indices": [1]},
        ],
    )
    (sequence_cache / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "FusionPortableV2",
                "platform": platform,
                "sequence_name": sequence,
                "recent_exclusion_keyframes": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sequence_cache


def test_fusionportablev2_defaults_include_requested_platform_sequences(tmp_path):
    module = _load_module()

    assert module.DEFAULT_SEQUENCES_BY_PLATFORM["handheld"] == [
        "handheld_escalator00",
        "handheld_escalator01",
        "handheld_grass00",
        "handheld_room00",
        "handheld_room01",
    ]
    assert module.DEFAULT_SEQUENCES_BY_PLATFORM["ugv"] == [
        "ugv_campus01",
        "ugv_parking00",
        "ugv_parking01",
        "ugv_parking02",
        "ugv_parking03",
    ]

    existing = _write_sequence_cache(tmp_path, "handheld", "handheld_escalator00")
    specs = module.resolve_sequence_specs(
        dataset="fusionportablev2",
        cache_root=tmp_path,
        platforms=["handheld"],
        strict=False,
    )

    assert [spec.sequence_cache for spec in specs] == [existing]
    assert specs[0].platform == "handheld"
    assert specs[0].sequence == "handheld_escalator00"


def test_evaluate_sequence_uses_causal_netvlad_topk_and_main_metrics(tmp_path):
    module = _load_module()
    sequence_cache = _write_sequence_cache(tmp_path, "handheld", "handheld_escalator00")

    class FakeBackend:
        def compute(self, image_paths, keyframe_indices):
            descriptors_by_idx = {
                0: [1.0, 0.0],
                1: [0.0, 1.0],
                2: [0.0, 1.0],
                3: [1.0, 0.0],
                4: [0.0, 1.0],
            }
            return module.DescriptorSet(
                keyframe_indices=[int(idx) for idx in keyframe_indices],
                descriptors=np.array(
                    [descriptors_by_idx[int(idx)] for idx in keyframe_indices],
                    dtype=np.float64,
                ),
            )

    result = module.evaluate_sequence(
        module.SequenceSpec(
            platform="handheld",
            sequence="handheld_escalator00",
            sequence_cache=sequence_cache,
        ),
        descriptor_backend=FakeBackend(),
        top_k=1,
        recent_exclusion_keyframes=None,
    )

    assert result.metrics == {"AP": 1.0, "MR@100P": 1.0}
    assert result.candidate_count == 3
    assert result.positive_candidate_count == 2
    assert result.query_count == 5
