import math
from pathlib import Path

import numpy as np


def _yaw_from_quat_xyzw(quat):
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def test_replace_identity_rotations_with_interpolated_translation_heading(tmp_path):
    from robust_loop_verifier.gt_heading import convert_tum_file_to_translation_heading

    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n"
        "2.0 1 1 0 0 0 0 1\n",
        encoding="utf-8",
    )

    summary = convert_tum_file_to_translation_heading(input_file, output_file)

    rows = [line.split() for line in output_file.read_text(encoding="utf-8").splitlines()]
    yaws = [_yaw_from_quat_xyzw([float(value) for value in row[4:8]]) for row in rows]

    assert summary["record_count"] == 3
    assert summary["replaced_identity_count"] == 3
    assert np.allclose(yaws, [0.0, math.pi / 4.0, math.pi / 2.0], atol=1e-6)


def test_preserve_non_identity_rotation_when_requested(tmp_path):
    from robust_loop_verifier.gt_heading import convert_tum_file_to_translation_heading

    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0.7071067811865475 0.7071067811865476\n"
        "2.0 2 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    summary = convert_tum_file_to_translation_heading(input_file, output_file)

    rows = [line.split() for line in output_file.read_text(encoding="utf-8").splitlines()]
    middle_quat = [float(value) for value in rows[1][4:8]]

    assert summary["replaced_identity_count"] == 2
    assert np.allclose(middle_quat, [0.0, 0.0, 0.707106781187, 0.707106781187])


def test_batch_process_fusionportable_platforms_mirrors_layout(tmp_path):
    from robust_loop_verifier.gt_heading import process_fusionportable_platform_gt

    input_root = tmp_path / "FusionPortable"
    output_root = tmp_path / "processed_handheld_legged_gt"
    for platform, sequence in [
        ("handheld", "handheld_room00"),
        ("legged", "legged_grass00"),
        ("ugv", "ugv_parking01"),
    ]:
        sequence_dir = input_root / platform / sequence
        sequence_dir.mkdir(parents=True)
        (sequence_dir / f"{sequence}.txt").write_text(
            "0.0 0 0 0 0 0 0 1\n"
            "1.0 1 0 0 0 0 0 1\n",
            encoding="utf-8",
        )

    summaries = process_fusionportable_platform_gt(input_root, output_root)

    written = {Path(summary["output_file"]).relative_to(output_root) for summary in summaries}
    assert written == {
        Path("handheld/handheld_room00/handheld_room00.txt"),
        Path("legged/legged_grass00/legged_grass00.txt"),
    }
    assert not (output_root / "ugv").exists()
