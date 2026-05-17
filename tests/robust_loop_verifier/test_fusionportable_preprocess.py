import json
from pathlib import Path

import pytest
from PIL import Image


def _fusionportable_config(tmp_path, **overrides):
    from robust_loop_verifier.schema import RobustLoopVerifierConfig

    data = {
        "dataset_name": "FusionPortableV2",
        "platform": "handheld",
        "input_root": str(tmp_path / "dataset"),
        "output_root": str(tmp_path / "cache"),
        "gt_root": str(tmp_path / "gt"),
        "positive_radius_m": 0.5,
        "recent_exclusion_keyframes": 2,
        "retrieval_top_k_main": 10,
        "retrieval_top_k_ablations": [5, 20],
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
    data.update(overrides)
    return RobustLoopVerifierConfig.from_mapping(data)


def _write_fusionportable_raw_fixture(tmp_path, keyframe_rows=None, raw_dir=None):
    raw_dir = Path(raw_dir) if raw_dir is not None else tmp_path / "dataset" / "raw"
    image_dir = raw_dir / "keyframe_images"
    gt_dir = tmp_path / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(exist_ok=True)

    if keyframe_rows is None:
        keyframe_rows = [
            {
                "keyframe_idx": idx,
                "timestamp": float(idx),
                "has_image": True,
                "image_path": "keyframe_images/{:06d}.png".format(idx),
            }
            for idx in range(3)
        ]

    for row in keyframe_rows:
        image_path = row.get("image_path")
        if (
            image_path
            and not Path(str(image_path)).is_absolute()
            and ".." not in Path(str(image_path)).parts
        ):
            target = raw_dir / str(image_path)
            if not target.is_dir():
                image = Image.new("RGB", (8, 8), color=(int(row["keyframe_idx"]), 0, 0))
                image.save(target)

    with (raw_dir / "keyframes_with_images.jsonl").open("w", encoding="utf-8") as handle:
        for row in keyframe_rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    (raw_dir / "trajectory_keyframes.txt").write_text(
        "".join(
            "{}  {} 0 0 0 0 0 1\n".format(float(row["timestamp"]), float(row["keyframe_idx"]))
            for row in keyframe_rows
        ),
        encoding="utf-8",
    )

    gt_file = gt_dir / "handheld_test.txt"
    gt_file.write_text(
        "".join(
            "{}  {} 0 0 0 0 0 1\n".format(float(row["timestamp"]), float(row["keyframe_idx"]))
            for row in keyframe_rows
        ),
        encoding="utf-8",
    )

    return raw_dir, gt_file


def test_preprocess_fusionportable_sequence_writes_online_causal_cache(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
    from robust_loop_verifier.io import read_json, read_jsonl

    raw_dir = tmp_path / "dataset" / "raw"
    image_dir = raw_dir / "keyframe_images"
    gt_dir = tmp_path / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir()

    for idx in range(6):
        image = Image.new("RGB", (8, 8), color=(idx, 0, 0))
        image.save(image_dir / "{:06d}.png".format(idx))

    (raw_dir / "sequence_meta.json").write_text(
        json.dumps(
            {
                "sequence_name": "handheld_test",
                "trajectory_keyframes_file": "trajectory_keyframes.txt",
            }
        ),
        encoding="utf-8",
    )

    with (raw_dir / "keyframes_with_images.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(6):
            handle.write(
                json.dumps(
                    {
                        "keyframe_idx": idx,
                        "timestamp": float(idx),
                        "has_image": True,
                        "image_path": "keyframe_images/{:06d}.png".format(idx),
                    }
                )
            )
            handle.write("\n")

    (raw_dir / "trajectory_keyframes.txt").write_text(
        "".join("{}  {} 0 0 0 0 0 1\n".format(float(idx), float(idx)) for idx in range(6)),
        encoding="utf-8",
    )

    gt_file = gt_dir / "handheld_test.txt"
    gt_file.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n"
        "2.0 2 0 0 0 0 0 1\n"
        "3.0 10 0 0 0 0 0 1\n"
        "4.0 0.2 0 0 0 0 0 1\n"
        "5.0 0.1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    config = _fusionportable_config(tmp_path, gt_root=str(gt_dir), platform="ugv")

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=config,
        max_gt_delta_sec=0.01,
    )

    positives = list(read_jsonl(out_dir / "positives.jsonl"))
    manifest = read_json(out_dir / "manifest.json")

    assert positives[-1]["query_idx"] == 5
    assert positives[-1]["positive_indices"] == [0]
    assert manifest["keyframe_count"] == 6
    assert "num_keyframes" not in manifest
    assert (out_dir / "images" / "000005.png").exists()


def test_preprocess_fusionportable_handheld_uses_aster_slam_trajectory_as_gt_by_default(
    tmp_path,
):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
    from robust_loop_verifier.io import read_json, read_jsonl

    raw_dir, gt_file = _write_fusionportable_raw_fixture(
        tmp_path,
        keyframe_rows=[
            {
                "keyframe_idx": idx,
                "timestamp": float(idx),
                "has_image": True,
                "image_path": "keyframe_images/{:06d}.png".format(idx),
            }
            for idx in range(4)
        ],
    )
    (raw_dir / "trajectory_keyframes.txt").write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 10 0 0 0 0 0 1\n"
        "2.0 20 0 0 0 0 0 1\n"
        "3.0 0.2 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    gt_file.write_text(
        "0.0 100 0 0 0 0 0 1\n"
        "1.0 101 0 0 0 0 0 1\n"
        "2.0 102 0 0 0 0 0 1\n"
        "3.0 103 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=_fusionportable_config(tmp_path, positive_radius_m=0.5),
        max_gt_delta_sec=0.01,
    )

    keyframes = list(read_jsonl(out_dir / "keyframes.jsonl"))
    positives = list(read_jsonl(out_dir / "positives.jsonl"))
    manifest = read_json(out_dir / "manifest.json")

    assert keyframes[0]["gt_pose"][3] == 0.0
    assert keyframes[3]["gt_pose"][3] == 0.2
    assert positives[-1]["positive_indices"] == [0]
    assert manifest["gt_label_source"] == "aster_slam_trajectory_keyframes"
    assert manifest["gt_trajectory_file"] == str(raw_dir / "trajectory_keyframes.txt")


def test_preprocess_fusionportable_positive_indices_require_rotation_overlap(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
    from robust_loop_verifier.io import read_json, read_jsonl

    raw_dir, gt_file = _write_fusionportable_raw_fixture(
        tmp_path,
        keyframe_rows=[
            {
                "keyframe_idx": idx,
                "timestamp": float(idx),
                "has_image": True,
                "image_path": "keyframe_images/{:06d}.png".format(idx),
            }
            for idx in range(4)
        ],
    )
    gt_file.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 5 0 0 0 0 0 1\n"
        "2.0 6 0 0 0 0 0 1\n"
        "3.0 0 0 0 0 0 0.7071067811865475 0.7071067811865476\n",
        encoding="utf-8",
    )

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=_fusionportable_config(tmp_path, platform="ugv", positive_radius_m=0.5),
        max_gt_delta_sec=0.01,
    )

    positives = list(read_jsonl(out_dir / "positives.jsonl"))
    manifest = read_json(out_dir / "manifest.json")

    assert positives[-1]["query_idx"] == 3
    assert positives[-1]["positive_indices"] == []
    assert manifest["positive_max_rotation_deg"] == 45.0


def test_preprocess_fusionportable_positive_rotation_threshold_is_configurable(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
    from robust_loop_verifier.io import read_jsonl

    raw_dir, gt_file = _write_fusionportable_raw_fixture(
        tmp_path,
        keyframe_rows=[
            {
                "keyframe_idx": idx,
                "timestamp": float(idx),
                "has_image": True,
                "image_path": "keyframe_images/{:06d}.png".format(idx),
            }
            for idx in range(4)
        ],
    )
    gt_file.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 5 0 0 0 0 0 1\n"
        "2.0 6 0 0 0 0 0 1\n"
        "3.0 0 0 0 0 0 0.7071067811865475 0.7071067811865476\n",
        encoding="utf-8",
    )

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=_fusionportable_config(
            tmp_path,
            platform="ugv",
            positive_radius_m=0.5,
            positive_max_rotation_deg=100.0,
        ),
        max_gt_delta_sec=0.01,
    )

    positives = list(read_jsonl(out_dir / "positives.jsonl"))

    assert positives[-1]["query_idx"] == 3
    assert positives[-1]["positive_indices"] == [0]


@pytest.mark.parametrize("sequence_kind", ["traversal", "absolute"])
def test_preprocess_fusionportable_sequence_rejects_unsafe_sequence_name(tmp_path, sequence_kind):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)
    config = _fusionportable_config(tmp_path)
    sequence_name = "../escape" if sequence_kind == "traversal" else str(tmp_path / "abs_escape")
    outside_path = (
        config.output_root / config.dataset_name / config.platform / sequence_name
    ).resolve()

    with pytest.raises(ValueError):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name=sequence_name,
            config=config,
            max_gt_delta_sec=0.01,
        )

    assert not outside_path.exists()


@pytest.mark.parametrize("image_path", ["../escape.png", "/tmp/escape.png"])
def test_preprocess_fusionportable_sequence_rejects_unsafe_image_path(tmp_path, image_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    rows = [
        {
            "keyframe_idx": 0,
            "timestamp": 0.0,
            "has_image": True,
            "image_path": image_path,
        }
    ]
    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path, keyframe_rows=rows)

    with pytest.raises(ValueError):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name="handheld_test",
            config=_fusionportable_config(tmp_path),
            max_gt_delta_sec=0.01,
        )


def test_preprocess_fusionportable_sequence_rejects_unsorted_keyframe_indices(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    keyframe_rows = [
        {"keyframe_idx": 1, "timestamp": 0.0, "has_image": False, "image_path": None},
        {"keyframe_idx": 0, "timestamp": 1.0, "has_image": False, "image_path": None},
    ]
    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path, keyframe_rows=keyframe_rows)

    with pytest.raises(ValueError):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name="handheld_test",
            config=_fusionportable_config(tmp_path),
            max_gt_delta_sec=0.01,
        )


def test_preprocess_fusionportable_sequence_links_images_from_relative_raw_dir(
    tmp_path, monkeypatch
):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative_raw_dir = raw_dir.relative_to(tmp_path)
    relative_gt_file = gt_file.relative_to(tmp_path)

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=relative_raw_dir,
        gt_trajectory_file=relative_gt_file,
        sequence_name="handheld_test",
        config=_fusionportable_config(tmp_path),
        max_gt_delta_sec=0.01,
    )

    output_image = out_dir / "images" / "000000.png"
    assert output_image.exists()
    if output_image.is_symlink():
        assert output_image.resolve(strict=True).exists()


def test_preprocess_fusionportable_sequence_uses_odom_rows_by_export_order(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
    from robust_loop_verifier.io import read_jsonl

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)
    (raw_dir / "trajectory_keyframes.txt").write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.25 10 0 0 0 0 0 1\n"
        "2.0 2 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=_fusionportable_config(tmp_path),
        max_gt_delta_sec=0.01,
    )
    keyframes = list(read_jsonl(out_dir / "keyframes.jsonl"))

    assert keyframes[1]["idx"] == 1
    assert keyframes[1]["timestamp"] == 1.0
    assert keyframes[1]["odom_pose"][3] == 10.0


def test_preprocess_fusionportable_sequence_rejects_odom_keyframe_length_mismatch(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)
    (raw_dir / "trajectory_keyframes.txt").write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="length"):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name="handheld_test",
            config=_fusionportable_config(tmp_path),
            max_gt_delta_sec=0.01,
        )


def test_preprocess_fusionportable_sequence_rejects_directory_image_source(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    rows = [
        {
            "keyframe_idx": 0,
            "timestamp": 0.0,
            "has_image": True,
            "image_path": "keyframe_images",
        }
    ]
    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path, keyframe_rows=rows)

    with pytest.raises(ValueError, match="regular file"):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name="handheld_test",
            config=_fusionportable_config(tmp_path),
            max_gt_delta_sec=0.01,
        )


def test_preprocess_fusionportable_sequence_replaces_stale_output_symlink(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)
    config = _fusionportable_config(tmp_path)
    output_image_dir = (
        config.output_root / config.dataset_name / config.platform / "handheld_test" / "images"
    )
    output_image_dir.mkdir(parents=True)
    stale_source = tmp_path / "stale.png"
    stale_source.write_bytes(b"stale")
    output_image = output_image_dir / "000000.png"
    output_image.symlink_to(stale_source)

    out_dir = preprocess_fusionportable_sequence(
        raw_dir=raw_dir,
        gt_trajectory_file=gt_file,
        sequence_name="handheld_test",
        config=config,
        max_gt_delta_sec=0.01,
    )

    expected_source = raw_dir / "keyframe_images" / "000000.png"
    output_image = out_dir / "images" / "000000.png"
    assert output_image.exists()
    if output_image.is_symlink():
        assert output_image.resolve(strict=True) == expected_source.resolve(strict=True)
    else:
        assert output_image.read_bytes() == expected_source.read_bytes()


def test_preprocess_fusionportable_sequence_rejects_symlinked_output_image_dir(tmp_path):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)
    config = _fusionportable_config(tmp_path)
    output_sequence_dir = config.output_root / config.dataset_name / config.platform / "handheld_test"
    outside_dir = tmp_path / "outside_images"
    outside_dir.mkdir()
    output_sequence_dir.mkdir(parents=True)
    (output_sequence_dir / "images").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="output"):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name="handheld_test",
            config=config,
            max_gt_delta_sec=0.01,
        )

    assert list(outside_dir.iterdir()) == []
    assert not (outside_dir / "000000.png").exists()


@pytest.mark.parametrize(
    ("field_name", "config_overrides", "sequence_name"),
    [
        ("dataset_name", {"dataset_name": "."}, "handheld_test"),
        ("platform", {"platform": "."}, "handheld_test"),
        ("sequence_name", {}, "."),
    ],
)
def test_preprocess_fusionportable_sequence_rejects_dot_path_segment(
    tmp_path, field_name, config_overrides, sequence_name
):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)

    with pytest.raises(ValueError, match=field_name):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name=sequence_name,
            config=_fusionportable_config(tmp_path, **config_overrides),
            max_gt_delta_sec=0.01,
        )


@pytest.mark.parametrize(
    ("field_name", "config_overrides", "sequence_name"),
    [
        ("dataset_name", {"dataset_name": "   "}, "handheld_test"),
        ("platform", {"platform": "\t"}, "handheld_test"),
        ("sequence_name", {}, " \n "),
    ],
)
def test_preprocess_fusionportable_sequence_rejects_whitespace_only_path_segment(
    tmp_path, field_name, config_overrides, sequence_name
):
    from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence

    raw_dir, gt_file = _write_fusionportable_raw_fixture(tmp_path)

    with pytest.raises(ValueError, match=field_name):
        preprocess_fusionportable_sequence(
            raw_dir=raw_dir,
            gt_trajectory_file=gt_file,
            sequence_name=sequence_name,
            config=_fusionportable_config(tmp_path, **config_overrides),
            max_gt_delta_sec=0.01,
        )
