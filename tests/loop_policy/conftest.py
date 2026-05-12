import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def synthetic_raw_sequence(tmp_path: Path) -> Path:
    raw = tmp_path / "handheld" / "handheld_room01" / "raw"
    images = raw / "keyframe_images"
    images.mkdir(parents=True)

    for idx in range(6):
        Image.new("RGB", (8, 6), color=(idx * 20, 10, 30)).save(images / f"{idx:06d}.jpg")

    (raw / "sequence_meta.json").write_text(
        json.dumps(
            {
                "sequence_name": "handheld_room01",
                "platform": "handheld",
                "loop_closure_enabled": False,
                "image_topic": "/stereo/frame_left/image_raw/compressed",
                "T_camera_lidar": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ),
        encoding="utf-8",
    )

    lines = []
    for idx in range(6):
        lines.append(
            json.dumps(
                {
                    "keyframe_idx": idx,
                    "timestamp": 100.0 + idx,
                    "trajectory_idx": idx,
                    "image_path": str(images / f"{idx:06d}.jpg"),
                }
            )
        )
    (raw / "keyframes_with_images.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (raw / "trajectory.txt").write_text(
        "\n".join(f"{100.0 + idx:.3f} {idx:.3f} 0 0 0 0 0 1" for idx in range(6)) + "\n",
        encoding="utf-8",
    )
    (raw / "trajectory_indices.txt").write_text(
        "\n".join(str(idx) for idx in range(6)) + "\n",
        encoding="utf-8",
    )
    (raw / "trajectory_keyframes.txt").write_text(
        (raw / "trajectory.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (raw / "trajectory_keyframe_indices.txt").write_text(
        "\n".join(str(idx) for idx in range(6)) + "\n",
        encoding="utf-8",
    )
    (raw / "keyframes.jsonl").write_text(
        (raw / "keyframes_with_images.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return raw
