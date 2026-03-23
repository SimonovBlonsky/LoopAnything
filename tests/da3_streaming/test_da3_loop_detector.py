import torch

from da3_streaming.loop_utils.da3_loop_detector import DA3LoopDetector


def test_find_loop_closures_applies_threshold_gap_and_topk(tmp_path):
    detector = DA3LoopDetector(
        image_dir=str(tmp_path),
        config={
            "Loop": {
                "DA3": {
                    "top_k": 2,
                    "similarity_threshold": 0.8,
                    "use_nms": False,
                    "nms_threshold": 0,
                }
            },
            "Model": {"loop_temporal_exclusion": 1},
        },
    )
    detector.image_paths = [tmp_path / f"{idx}.png" for idx in range(4)]
    detector.descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.0],
            [1.0, 0.0],
            [0.9, 0.0],
        ]
    )

    loops = detector.find_loop_closures()

    assert (0, 1) not in [(a, b) for a, b, _ in loops]
    assert any({a, b} == {0, 2} or {a, b} == {1, 3} for a, b, _ in loops)
