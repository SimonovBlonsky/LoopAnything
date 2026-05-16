from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "robust_loop_verifier"


def test_robust_loop_verifier_does_not_import_legacy_loop_policy():
    forbidden = "loop_policy"
    offenders = []
    for path in SRC.rglob("*.py"):
        if forbidden in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT))

    assert offenders == []


def test_plan_does_not_allow_loop_policy_as_reference_source():
    plan = ROOT / "docs" / "superpowers" / "plans" / "2026-05-15-robust-loop-verifier-offline.md"
    text = plan.read_text(encoding="utf-8")

    assert "Do not reference, import, copy, or use `LoopAnything/src/loop_policy`" in text
    assert "`LoopAnything/tests/loop_policy`" in text


def test_pyproject_packages_robust_loop_verifier():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"src/robust_loop_verifier"' in text
