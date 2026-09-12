# VINS-Fusion Dataset Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VINS-Fusion run FusionPortableV2 and NTU-VIRAL sequences with reproducible odometry output and evo ATE evaluation.

**Architecture:** Keep VINS-Fusion minimally modified. Add dataset-specific configuration, small ROS/data utility scripts, and batch runners around the existing `vins_node` and optional stock `loop_fusion_node`. Do not integrate LoopAnything loop verification in this phase; this phase produces the VINS odometry baseline needed by later runtime loop-closure experiments.

**Tech Stack:** ROS Noetic, catkin, VINS-Fusion, Python 3, `rosbag`, `image_transport`, `evo_ape`, FusionPortableV2 calibration YAML, NTU-VIRAL calibration YAML.

---

## File Structure

- Create `VINS_Fusion_ws/src/VINS-Fusion/config/ntuviral/`: local copy of the noetic fork's NTU-VIRAL VINS config.
- Create `VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/`: generated VINS configs for FusionPortableV2 platforms.
- Create `VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_ntuviral_dataset.launch`: local NTU-VIRAL launch wrapper.
- Create `VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_fusionportable_v2_dataset.launch`: FusionPortableV2 launch wrapper with compressed-image republishers.
- Create `LoopAnything/robust_loop_verification_scripts/vins_fusion_dataset_config.py`: deterministic config generator and calibration converter.
- Create `LoopAnything/robust_loop_verification_scripts/run_vins_fusion_dataset.py`: one-sequence runner for roslaunch, rosbag playback, trajectory collection, and evo.
- Create `LoopAnything/robust_loop_verification_scripts/run_vins_fusion_dataset_batch.sh`: batch wrapper for selected FusionPortableV2/NTU-VIRAL sequences.
- Create `LoopAnything/tests/robust_loop_verifier/test_vins_fusion_dataset_config.py`: unit tests for calibration conversion and config generation.
- Create `LoopAnything/tests/robust_loop_verifier/test_vins_fusion_dataset_runner.py`: dry-run tests for runner command construction.

## Task 1: Add NTU-VIRAL Baseline Configs

**Files:**
- Create: `VINS_Fusion_ws/src/VINS-Fusion/config/ntuviral/viral_stereo_imu_config.yaml`
- Create: `VINS_Fusion_ws/src/VINS-Fusion/config/ntuviral/camLeft.yaml`
- Create: `VINS_Fusion_ws/src/VINS-Fusion/config/ntuviral/camRight.yaml`
- Create: `VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_ntuviral_dataset.launch`

- [ ] **Step 1: Copy the noetic fork's NTU-VIRAL camera configs**

Use `/tmp/vins_fusion_brytsknguyen_noetic/config/ntuviral/camLeft.yaml` and `camRight.yaml` as the source. If `/tmp/vins_fusion_brytsknguyen_noetic` is missing, recreate it with:

```bash
git clone --depth 1 --branch noetic https://github.com/brytsknguyen/VINS-Fusion.git /tmp/vins_fusion_brytsknguyen_noetic
```

Expected `camLeft.yaml` contents:

```yaml
%YAML:1.0
---
model_type:   PINHOLE
camera_name:  camera
image_width:  752
image_height: 480
distortion_parameters:
   k1: -0.288105327549552
   k2:  0.074578284234601
   p1:  7.784489598138802e-04
   p2: -2.277853975035461e-04
projection_parameters:
   fx: 4.250258563372763e+02
   fy: 4.267976260903337e+02
   cx: 3.860151866550880e+02
   cy: 2.419130336743440e+02
```

Expected `camRight.yaml` contents:

```yaml
%YAML:1.0
---
model_type:   PINHOLE
camera_name:  camera
image_width:  752
image_height: 480
distortion_parameters:
   k1: -0.300267420221178
   k2:  0.090544063693053
   p1:  3.330220891093334e-05
   p2:  8.989607188457415e-05
projection_parameters:
   fx: 4.313364265799752e+02
   fy: 4.327527965378035e+02
   cx: 3.548956286992647e+02
   cy: 2.325508916495161e+02
```

- [ ] **Step 2: Add local NTU-VIRAL VINS config**

Create `viral_stereo_imu_config.yaml` from the noetic fork, but change paths to local-safe defaults:

```yaml
%YAML:1.0

imu:          1
num_of_cam:   2

imu_topic:    "/imu/imu"
image0_topic: "/right/image_raw"
image1_topic: "/left/image_raw"
output_path:  "/tmp/vins_fusion_output/"

cam0_calib:   "camRight.yaml"
cam1_calib:   "camLeft.yaml"
image_width:  752
image_height: 480

estimate_extrinsic: 1

body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [-0.01916508, -0.01496218,  0.99970437,  0.00519443,
           0.99974371,  0.01176483,  0.01934191,  0.1347802,
          -0.01205075,  0.99981884,  0.01473287,  0.01465067,
           0.00000000,  0.00000000,  0.00000000,  1.00000000]

body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ 0.02183084, -0.01312053,  0.99967558,  0.00552943,
           0.99975965,  0.00230088, -0.02180248, -0.12431302,
          -0.00201407,  0.99991127,  0.01316761,  0.01614686,
           0.00000000,  0.00000000,  0.00000000,  1.00000000 ]

multiple_thread: 1
max_cnt:     150
min_dist:    30
freq:        10
F_threshold: 1.0
show_track:  1
flow_back:   1

max_solver_time: 0.04
max_num_iterations: 8
keyframe_parallax: 10.0

acc_n:  6.0e-2
gyr_n:  5.0e-3
acc_w:  8.0e-5
gyr_w:  3.0e-6
g_norm: 9.81007

estimate_td: 0
td:          0.0

load_previous_pose_graph: 0
pose_graph_save_path:     "/tmp/vins_fusion_output/pose_graph/"
save_image:               0
```

- [ ] **Step 3: Add NTU-VIRAL launch wrapper**

Create `run_ntuviral_dataset.launch`:

```xml
<?xml version="1.0"?>
<launch>
    <arg name="bag_file" default="/data/datasets/NTU-VIRAL/data/eee_01/eee_01.bag" />
    <arg name="config_file" default="$(find vins)/../config/ntuviral/viral_stereo_imu_config.yaml" />
    <arg name="run_loop_fusion" default="false" />
    <arg name="play_bag" default="true" />
    <arg name="play_rate" default="1.0" />
    <arg name="output_dir" default="/tmp/vins_fusion_output" />

    <param name="/use_sim_time" value="true" />

    <node pkg="vins" type="vins_node" name="vins_estimator"
          args="$(arg config_file)" output="screen" />

    <node if="$(arg run_loop_fusion)" pkg="loop_fusion" type="loop_fusion_node"
          name="loop_fusion" args="$(arg config_file)" output="screen">
        <param name="pose_graph_save_path" value="$(arg output_dir)/pose_graph/" />
    </node>

    <node if="$(arg play_bag)" pkg="rosbag" type="play" name="rosbag_play"
          args="--clock -r $(arg play_rate) $(arg bag_file)" output="screen" />
</launch>
```

- [ ] **Step 4: Verify files can be parsed by ROS tools**

Run:

```bash
source /opt/ros/noetic/setup.bash
source /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/devel/setup.bash
roslaunch VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_ntuviral_dataset.launch --args
```

Expected: command prints launch args or exits without XML parse errors.

## Task 2: Add FusionPortableV2 Config Generator

**Files:**
- Create: `LoopAnything/robust_loop_verification_scripts/vins_fusion_dataset_config.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_vins_fusion_dataset_config.py`
- Create generated outputs under `VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/`

- [ ] **Step 1: Write tests for matrix parsing and VINS camera YAML rendering**

Create `test_vins_fusion_dataset_config.py`:

```python
from pathlib import Path

import numpy as np

from robust_loop_verification_scripts.vins_fusion_dataset_config import (
    opencv_matrix_yaml,
    render_vins_camera_yaml,
)


def test_opencv_matrix_yaml_formats_4x4_matrix():
    mat = np.eye(4)
    text = opencv_matrix_yaml(mat, indent="   ")
    assert "rows: 4" in text
    assert "cols: 4" in text
    assert "dt: d" in text
    assert "1.000000000000" in text


def test_render_vins_camera_yaml_pinhole():
    text = render_vins_camera_yaml(
        model_type="PINHOLE",
        image_width=1032,
        image_height=772,
        fx=600.0,
        fy=601.0,
        cx=516.0,
        cy=386.0,
        k1=-0.1,
        k2=0.01,
        p1=0.001,
        p2=-0.002,
    )
    assert "model_type: PINHOLE" in text
    assert "image_width: 1032" in text
    assert "fx: 600.000000000000" in text
    assert "p2: -0.002000000000" in text
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. pytest tests/robust_loop_verifier/test_vins_fusion_dataset_config.py -q
```

Expected: import failure because `vins_fusion_dataset_config.py` does not exist.

- [ ] **Step 3: Implement deterministic renderer**

Create `vins_fusion_dataset_config.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def opencv_matrix_yaml(matrix: np.ndarray, *, indent: str = "   ") -> str:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {arr.shape}")
    values = ", ".join(f"{value:.12f}" for value in arr.reshape(-1))
    return (
        "!!opencv-matrix\n"
        f"{indent}rows: 4\n"
        f"{indent}cols: 4\n"
        f"{indent}dt: d\n"
        f"{indent}data: [{values}]"
    )


def render_vins_camera_yaml(
    *,
    model_type: str,
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
) -> str:
    return (
        "%YAML:1.0\n"
        "---\n"
        "#camera calibration\n"
        f"model_type: {model_type}\n"
        "camera_name: camera\n"
        f"image_width: {int(image_width)}\n"
        f"image_height: {int(image_height)}\n\n"
        "distortion_parameters:\n"
        f"   k1: {float(k1):.12f}\n"
        f"   k2: {float(k2):.12f}\n"
        f"   p1: {float(p1):.12f}\n"
        f"   p2: {float(p2):.12f}\n"
        "projection_parameters:\n"
        f"   fx: {float(fx):.12f}\n"
        f"   fy: {float(fy):.12f}\n"
        f"   cx: {float(cx):.12f}\n"
        f"   cy: {float(cy):.12f}\n"
    )
```

- [ ] **Step 4: Add FusionPortable platform specs**

Extend the script with an explicit mapping for the first bring-up targets:

```python
FUSIONPORTABLE_PLATFORM_SPECS = {
    "handheld": {
        "calib_root": "/data/datasets/FusionPortable/calibration_files/20220209_calib/calib",
        "left_camera_yaml": "frame_cam00.yaml",
        "right_camera_yaml": "frame_cam01.yaml",
        "imu_yaml": "body_imu.yaml",
        "left_topic_compressed": "/stereo/frame_left/image_raw/compressed",
        "right_topic_compressed": "/stereo/frame_right/image_raw/compressed",
        "imu_topic": "/stim300/imu/data_raw",
        "output_subdir": "handheld",
    },
    "ugv": {
        "calib_root": "/data/datasets/FusionPortable/calibration_files/20230309_calib/calib",
        "left_camera_yaml": "frame_cam00.yaml",
        "right_camera_yaml": "frame_cam01.yaml",
        "imu_yaml": "body_imu.yaml",
        "left_topic_compressed": "/stereo/frame_left/image_raw/compressed",
        "right_topic_compressed": "/stereo/frame_right/image_raw/compressed",
        "imu_topic": "/3dm_ins/imu/data_raw",
        "output_subdir": "ugv",
    },
}
```

This mapping is intentionally explicit. If a sequence uses a different calibration date, add a new platform spec rather than guessing.

- [ ] **Step 5: Implement camera YAML reader**

Add:

```python
def load_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    cleaned = "\n".join(line for line in text.splitlines() if not line.startswith("%YAML:"))
    return yaml.safe_load(cleaned)


def camera_params_from_fusionportable(path: Path) -> dict[str, float | int | str]:
    data = load_yaml(path)
    intr = data["projection_parameters"]
    dist = data["distortion_parameters"]
    return {
        "model_type": "PINHOLE",
        "image_width": int(data["image_width"]),
        "image_height": int(data["image_height"]),
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "k1": float(dist["k1"]),
        "k2": float(dist["k2"]),
        "p1": float(dist["p1"]),
        "p2": float(dist["p2"]),
    }
```

- [ ] **Step 6: Implement body_T_cam extraction**

FusionPortable calibration files may store transforms under different keys. Add strict key handling:

```python
def matrix_from_yaml_value(value) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        rows = int(value.get("rows", 4))
        cols = int(value.get("cols", 4))
        return np.asarray(value["data"], dtype=float).reshape(rows, cols)
    if isinstance(value, list):
        arr = np.asarray(value, dtype=float)
        if arr.size == 16:
            return arr.reshape(4, 4)
    raise ValueError(f"unsupported transform value: {value!r}")


def find_transform(data: dict, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in data:
            mat = matrix_from_yaml_value(data[key])
            if mat.shape != (4, 4):
                raise ValueError(f"{key} must be 4x4, got {mat.shape}")
            return mat
    raise KeyError(f"none of transform keys found: {keys}")
```

Use these candidate keys in order: `("T_Body_Cam", "T_body_cam", "T_imu_cam", "body_T_cam")`. If a platform file does not contain one of these, stop and inspect that calibration manually. Do not silently invert unknown transforms.

- [ ] **Step 7: Generate FusionPortable config files**

Add `generate_fusionportable_config(platform: str, output_root: Path)` that writes:

```text
VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/<platform>/camLeft.yaml
VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/<platform>/camRight.yaml
VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/<platform>/fusionportable_stereo_imu_config.yaml
```

The generated main config should use raw republished image topics:

```yaml
imu_topic: "/vins_fusion/imu"
image0_topic: "/vins_fusion/right/image_raw"
image1_topic: "/vins_fusion/left/image_raw"
cam0_calib: "camRight.yaml"
cam1_calib: "camLeft.yaml"
estimate_extrinsic: 1
estimate_td: 0
save_image: 0
```

- [ ] **Step 8: Run unit tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. pytest tests/robust_loop_verifier/test_vins_fusion_dataset_config.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Generate first configs**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. python robust_loop_verification_scripts/vins_fusion_dataset_config.py \
  generate-fusionportable --platform handheld \
  --output-root /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2
PYTHONPATH=. python robust_loop_verification_scripts/vins_fusion_dataset_config.py \
  generate-fusionportable --platform ugv \
  --output-root /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2
```

Expected: generated files exist and include `body_T_cam0` and `body_T_cam1`.

## Task 3: Add FusionPortableV2 Launch Wrapper

**Files:**
- Create: `VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_fusionportable_v2_dataset.launch`
- Modify: none

- [ ] **Step 1: Create launch file**

Use `image_transport/republish` to convert compressed camera topics into raw image topics:

```xml
<?xml version="1.0"?>
<launch>
    <arg name="bag_file" default="/data/datasets/FusionPortable/handheld/handheld_room00/handheld_room00.bag" />
    <arg name="config_file" default="$(find vins)/../config/fusionportable_v2/handheld/fusionportable_stereo_imu_config.yaml" />
    <arg name="left_compressed_topic" default="/stereo/frame_left/image_raw/compressed" />
    <arg name="right_compressed_topic" default="/stereo/frame_right/image_raw/compressed" />
    <arg name="imu_topic" default="/stim300/imu/data_raw" />
    <arg name="run_loop_fusion" default="false" />
    <arg name="play_bag" default="true" />
    <arg name="play_rate" default="1.0" />
    <arg name="output_dir" default="/tmp/vins_fusion_output" />

    <param name="/use_sim_time" value="true" />

    <node pkg="image_transport" type="republish" name="left_image_republisher"
          args="compressed in:=$(arg left_compressed_topic) raw out:=/vins_fusion/left/image_raw" />
    <node pkg="image_transport" type="republish" name="right_image_republisher"
          args="compressed in:=$(arg right_compressed_topic) raw out:=/vins_fusion/right/image_raw" />
    <node pkg="topic_tools" type="relay" name="imu_relay"
          args="$(arg imu_topic) /vins_fusion/imu" />

    <node pkg="vins" type="vins_node" name="vins_estimator"
          args="$(arg config_file)" output="screen" />

    <node if="$(arg run_loop_fusion)" pkg="loop_fusion" type="loop_fusion_node"
          name="loop_fusion" args="$(arg config_file)" output="screen">
        <param name="pose_graph_save_path" value="$(arg output_dir)/pose_graph/" />
    </node>

    <node if="$(arg play_bag)" pkg="rosbag" type="play" name="rosbag_play"
          args="--clock -r $(arg play_rate) $(arg bag_file)" output="screen" />
</launch>
```

- [ ] **Step 2: Verify launch arguments parse**

Run:

```bash
source /opt/ros/noetic/setup.bash
source /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/devel/setup.bash
roslaunch /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_fusionportable_v2_dataset.launch --args
```

Expected: command exits without XML parse errors.

## Task 4: Add One-Sequence Runner and evo Evaluation

**Files:**
- Create: `LoopAnything/robust_loop_verification_scripts/run_vins_fusion_dataset.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_vins_fusion_dataset_runner.py`

- [ ] **Step 1: Write dry-run test for runner command construction**

Create `test_vins_fusion_dataset_runner.py`:

```python
from pathlib import Path

from robust_loop_verification_scripts.run_vins_fusion_dataset import build_roslaunch_command


def test_build_roslaunch_command_for_ntu_viral():
    cmd = build_roslaunch_command(
        launch_file=Path("/repo/run.launch"),
        bag_file=Path("/data/eee_01.bag"),
        config_file=Path("/repo/config.yaml"),
        output_dir=Path("/tmp/out"),
        play_rate=0.5,
        extra_args={"run_loop_fusion": "false"},
    )
    assert cmd[:2] == ["roslaunch", "/repo/run.launch"]
    assert "bag_file:=/data/eee_01.bag" in cmd
    assert "config_file:=/repo/config.yaml" in cmd
    assert "play_rate:=0.5" in cmd
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. pytest tests/robust_loop_verifier/test_vins_fusion_dataset_runner.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement runner skeleton**

Create `run_vins_fusion_dataset.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def build_roslaunch_command(
    *,
    launch_file: Path,
    bag_file: Path,
    config_file: Path,
    output_dir: Path,
    play_rate: float,
    extra_args: dict[str, str],
) -> list[str]:
    cmd = [
        "roslaunch",
        str(launch_file),
        f"bag_file:={bag_file}",
        f"config_file:={config_file}",
        f"output_dir:={output_dir}",
        f"play_rate:={play_rate}",
    ]
    for key, value in sorted(extra_args.items()):
        cmd.append(f"{key}:={value}")
    return cmd
```

- [ ] **Step 4: Add CLI arguments**

Extend the file:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VINS-Fusion on one dataset sequence.")
    parser.add_argument("--dataset", choices=("fusionportable_v2", "ntu_viral"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--bag-file", type=Path, required=True)
    parser.add_argument("--gt-tum", type=Path, required=True)
    parser.add_argument("--launch-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--play-rate", type=float, default=1.0)
    parser.add_argument("--run-loop-fusion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
```

- [ ] **Step 5: Implement process execution**

Add:

```python
def run_command(cmd: list[str], *, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN: " + " ".join(cmd) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.dataset / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_roslaunch_command(
        launch_file=args.launch_file,
        bag_file=args.bag_file,
        config_file=args.config_file,
        output_dir=output_dir,
        play_rate=args.play_rate,
        extra_args={"run_loop_fusion": "true" if args.run_loop_fusion else "false"},
    )
    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    rc = run_command(cmd, log_path=output_dir / "roslaunch.log", dry_run=args.dry_run)
    if rc != 0:
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Add evo hook after trajectory path is confirmed**

After the first manual VINS run, identify the produced trajectory path. VINS-Fusion usually writes to `output_path` from the YAML. Add an explicit config output path in generated YAML and copy the resulting VINS file into `output_dir`. Then run:

```bash
evo_ape tum <GT_TUM> <EST_TUM> --align --correct_scale --save_results <OUTPUT_DIR>/evo_ape.zip
```

If the VINS output is not TUM format, add a converter function in this runner and a unit test for one sample row before enabling evo.

- [ ] **Step 7: Run tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. pytest tests/robust_loop_verifier/test_vins_fusion_dataset_runner.py -q
```

Expected: all tests pass.

## Task 5: Build and Smoke-Test VINS-Fusion

**Files:**
- Modify only if needed: VINS-Fusion CMake/source files for Noetic compile compatibility.

- [ ] **Step 1: Build workspace**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

Expected: all VINS-Fusion packages build.

- [ ] **Step 2: If OpenCV macro errors occur, apply minimal Noetic patch**

Expected common replacements:

```cpp
CV_LOAD_IMAGE_GRAYSCALE -> cv::IMREAD_GRAYSCALE
CV_LOAD_IMAGE_UNCHANGED -> cv::IMREAD_UNCHANGED
```

Do not do broad refactors. Patch only compile errors.

- [ ] **Step 3: Rebuild**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

Expected: build succeeds.

## Task 6: Run First Two Bring-Up Sequences

**Files:**
- No new files unless a sequence-specific config correction is needed.

- [ ] **Step 1: NTU-VIRAL eee_01 dry run**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. python robust_loop_verification_scripts/run_vins_fusion_dataset.py \
  --dataset ntu_viral \
  --sequence eee_01 \
  --bag-file /data/datasets/NTU-VIRAL/data/eee_01/eee_01.bag \
  --gt-tum /data/datasets/NTU-VIRAL/processed_gt_tum/eee_01.txt \
  --launch-file /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_ntuviral_dataset.launch \
  --config-file /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/config/ntuviral/viral_stereo_imu_config.yaml \
  --output-root /home/chenguyuan/code/NeurIPS26/LoopAnything/workspace/vins_fusion_runtime \
  --play-rate 1.0 \
  --dry-run
```

Expected: `command.txt` and `roslaunch.log` are written under `workspace/vins_fusion_runtime/ntu_viral/eee_01`.

- [ ] **Step 2: NTU-VIRAL eee_01 real run**

Run the same command without `--dry-run`.

Expected:
- `roslaunch.log` shows VINS initialization and frame processing.
- VINS trajectory output exists.
- evo report exists after Task 4 Step 6 is enabled.

- [ ] **Step 3: FusionPortable handheld_room00 dry run**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=. python robust_loop_verification_scripts/run_vins_fusion_dataset.py \
  --dataset fusionportable_v2 \
  --sequence handheld_room00 \
  --bag-file /data/datasets/FusionPortable/handheld/handheld_room00/handheld_room00.bag \
  --gt-tum /data/datasets/FusionPortable/handheld/handheld_room00/handheld_room00.txt \
  --launch-file /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_fusionportable_v2_dataset.launch \
  --config-file /home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/handheld/fusionportable_stereo_imu_config.yaml \
  --output-root /home/chenguyuan/code/NeurIPS26/LoopAnything/workspace/vins_fusion_runtime \
  --play-rate 1.0 \
  --dry-run
```

Expected: dry-run files are written.

- [ ] **Step 4: FusionPortable handheld_room00 real run**

Run the same command without `--dry-run`.

Expected:
- republisher nodes publish `/vins_fusion/left/image_raw` and `/vins_fusion/right/image_raw`.
- `vins_node` receives images and IMU.
- VINS trajectory output exists.
- evo report exists after Task 4 Step 6 is enabled.

## Task 7: Add Batch Wrapper

**Files:**
- Create: `LoopAnything/robust_loop_verification_scripts/run_vins_fusion_dataset_batch.sh`

- [ ] **Step 1: Create batch script**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chenguyuan/code/NeurIPS26"
LOOP_ROOT="${REPO_ROOT}/LoopAnything"
OUT_ROOT="${LOOP_ROOT}/workspace/vins_fusion_runtime"

PYTHONPATH="${LOOP_ROOT}" python "${LOOP_ROOT}/robust_loop_verification_scripts/run_vins_fusion_dataset.py" \
  --dataset ntu_viral \
  --sequence eee_01 \
  --bag-file /data/datasets/NTU-VIRAL/data/eee_01/eee_01.bag \
  --gt-tum /data/datasets/NTU-VIRAL/processed_gt_tum/eee_01.txt \
  --launch-file "${REPO_ROOT}/VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_ntuviral_dataset.launch" \
  --config-file "${REPO_ROOT}/VINS_Fusion_ws/src/VINS-Fusion/config/ntuviral/viral_stereo_imu_config.yaml" \
  --output-root "${OUT_ROOT}" \
  --play-rate 1.0

PYTHONPATH="${LOOP_ROOT}" python "${LOOP_ROOT}/robust_loop_verification_scripts/run_vins_fusion_dataset.py" \
  --dataset fusionportable_v2 \
  --sequence handheld_room00 \
  --bag-file /data/datasets/FusionPortable/handheld/handheld_room00/handheld_room00.bag \
  --gt-tum /data/datasets/FusionPortable/handheld/handheld_room00/handheld_room00.txt \
  --launch-file "${REPO_ROOT}/VINS_Fusion_ws/src/VINS-Fusion/vins_estimator/launch/run_fusionportable_v2_dataset.launch" \
  --config-file "${REPO_ROOT}/VINS_Fusion_ws/src/VINS-Fusion/config/fusionportable_v2/handheld/fusionportable_stereo_imu_config.yaml" \
  --output-root "${OUT_ROOT}" \
  --play-rate 1.0
```

- [ ] **Step 2: Make executable**

Run:

```bash
chmod +x /home/chenguyuan/code/NeurIPS26/LoopAnything/robust_loop_verification_scripts/run_vins_fusion_dataset_batch.sh
```

- [ ] **Step 3: Run dry version manually before overnight run**

Temporarily add `--dry-run` to both commands and run:

```bash
/home/chenguyuan/code/NeurIPS26/LoopAnything/robust_loop_verification_scripts/run_vins_fusion_dataset_batch.sh
```

Expected: both command files are produced.

Remove `--dry-run` after validation.

## Self-Review

- Spec coverage: The plan covers Noetic/NTU config import, FusionPortable config generation, compressed image adaptation, launch wrappers, runner/evo hooks, build, and first-sequence validation.
- Scope control: LoopAnything runtime retrieval/verifier integration is intentionally excluded from this plan. That should be a separate plan after VINS odometry and stock loop fusion are reliable.
- Ambiguity: FusionPortable transform keys vary by calibration file; the plan requires strict key lookup and manual inspection instead of guessing inverse transforms.
- Testing: Includes unit tests for config rendering and runner command construction, plus ROS launch parse checks and real sequence smoke tests.
