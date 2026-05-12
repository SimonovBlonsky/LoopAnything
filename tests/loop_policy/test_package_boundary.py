from pathlib import Path


def test_loop_policy_package_imports():
    import loop_policy

    assert loop_policy.__version__ == "0.1.0"


def test_loop_policy_code_lives_outside_depth_anything_tree():
    repo = Path(__file__).resolve().parents[2]
    loop_policy_dir = repo / "src" / "loop_policy"
    da3_dir = repo / "src" / "depth_anything_3"

    assert loop_policy_dir.is_dir()
    assert not list(da3_dir.rglob("*loop_policy*"))
