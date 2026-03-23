from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from da3_streaming.loop_utils.training_free_gate import (
    combine_loop_scores,
    relative_pose_from_extrinsics,
)


def test_relative_pose_from_extrinsics_matches_manual_transform():
    a = np.eye(4, dtype=np.float32)
    b = np.eye(4, dtype=np.float32)
    b[0, 3] = 2.0
    rel = relative_pose_from_extrinsics(a, b)
    assert np.allclose(rel[:3, 3], np.array([-2.0, 0.0, 0.0]), atol=1e-6)


def test_combine_loop_scores_is_monotonic_in_overlap():
    low = combine_loop_scores(desc_sim=0.9, pair_sim=0.8, overlap=0.1)
    high = combine_loop_scores(desc_sim=0.9, pair_sim=0.8, overlap=0.8)
    assert high > low
