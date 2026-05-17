import json
from pathlib import Path

from PIL import Image


def _config(tmp_path):
    from robust_loop_verifier.schema import RobustLoopVerifierConfig

    return RobustLoopVerifierConfig.from_mapping(
        {
            "dataset_name": "FusionPortableV2",
            "platform": "handheld",
            "input_root": str(tmp_path / "dataset"),
            "output_root": str(tmp_path / "cache"),
            "gt_root": str(tmp_path / "gt"),
            "positive_radius_m": 0.5,
            "recent_exclusion_keyframes": 1,
            "retrieval_top_k_main": 2,
            "retrieval_top_k_ablations": [1],
            "support_window": 4,
            "support_count": 1,
            "min_support_baseline_m": 0.3,
            "pgo_noise": {
                "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
                "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            },
            "da3": {
                "process_res": 504,
                "ref_view_strategy": "first",
            },
        }
    )


def test_preprocessed_cache_manifest_keyframes_and_positives_contract(tmp_path: Path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
    from robust_loop_verifier.io import read_json, read_jsonl

    raw_dir = tmp_path / "dataset" / "raw"
    image_dir = raw_dir / "keyframe_images"
    gt_dir = tmp_path / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir()

    for idx in range(5):
        Image.new("RGB", (8, 8), color=(idx, 0, 0)).save(image_dir / f"{idx:06d}.png")

    (raw_dir / "keyframes_with_images.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "keyframe_idx": idx,
                    "timestamp": float(idx),
                    "has_image": True,
                    "image_path": f"keyframe_images/{idx:06d}.png",
                }
            )
            + "\n"
            for idx in range(5)
        ),
        encoding="utf-8",
    )
    (raw_dir / "trajectory_keyframes.txt").write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n"
        "2.0 2 0 0 0 0 0 1\n"
        "3.0 3 0 0 0 0 0 1\n"
        "4.0 0.1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    gt_file = gt_dir / "handheld_test.txt"
    gt_file.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n"
        "2.0 2 0 0 0 0 0 1\n"
        "3.0 3 0 0 0 0 0 1\n"
        "4.0 0.1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    cache_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=_config(tmp_path),
        max_gt_delta_sec=0.01,
    )

    manifest = read_json(cache_dir / "manifest.json")
    keyframes = list(read_jsonl(cache_dir / "keyframes.jsonl"))
    positives = list(read_jsonl(cache_dir / "positives.jsonl"))

    assert manifest["dataset_name"] == "FusionPortableV2"
    assert manifest["platform"] == "handheld"
    assert manifest["sequence_name"] == "handheld_test"
    assert manifest["keyframe_count"] == 5
    assert len(keyframes) == 5
    assert len(positives) == 5
    assert keyframes[0]["idx"] == 0
    assert isinstance(keyframes[0]["image_path"], str)
    assert len(keyframes[0]["odom_pose"]) == 16
    assert len(keyframes[0]["gt_pose"]) == 16
    assert positives[-1]["query_idx"] == 4
    assert positives[-1]["positive_indices"] == [0]
