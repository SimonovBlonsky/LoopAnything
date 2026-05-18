import json
import os
import subprocess
from pathlib import Path


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / script_name
    return subprocess.run(
        ["bash", str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def test_support_ensemble_run_script_has_help():
    result = _run_help("run_support_ensemble_verifier.sh")
    blocked_terms = ("sta" + "ge_" + "a", "sta" + "ge_" + "b")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_support_ensemble_verifier.sh" in result.stdout
    assert "SEQUENCE_NAME" in result.stdout
    assert "CONFIG" in result.stdout
    assert "QUERY_LIMIT" in result.stdout
    assert all(term not in result.stdout.lower() for term in blocked_terms)


def test_support_ensemble_sweep_script_has_help():
    result = _run_help("run_support_ensemble_score_sweep.sh")
    blocked_terms = ("sta" + "ge_" + "a", "sta" + "ge_" + "b")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_support_ensemble_score_sweep.sh" in result.stdout
    assert "RUN_ROOT" in result.stdout
    assert "CANDIDATE_RECORDS" in result.stdout
    assert all(term not in result.stdout.lower() for term in blocked_terms)


def _write_fake_python(path: Path, record_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import os",
                "import sys",
                f"record_path = {str(record_path)!r}",
                "with open(record_path, 'w', encoding='utf-8') as handle:",
                "    payload = {",
                "        'argv': sys.argv[1:],",
                "        'pythonpath': os.environ.get('PYTHONPATH'),",
                "    }",
                "    json.dump(payload, handle)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_support_ensemble_sweep_resolves_relative_run_root_before_repo_cd(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_support_ensemble_score_sweep.sh"
    caller_dir = tmp_path / "caller"
    run_root = caller_dir / "relative_run"
    record_path = tmp_path / "record.json"
    fake_python = tmp_path / "fake_python.py"
    caller_dir.mkdir()
    run_root.mkdir()
    (run_root / "candidate_records.jsonl").write_text("", encoding="utf-8")
    _write_fake_python(fake_python, record_path)

    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "RUN_ROOT": "relative_run",
    }
    result = subprocess.run(
        ["bash", str(script)],
        cwd=caller_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    expected_records = run_root / "candidate_records.jsonl"
    expected_output = run_root / "support_ensemble_score_sweep"
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    argv = recorded["argv"]

    assert result.returncode == 0, result.stdout + result.stderr
    assert argv[argv.index("--candidate-records") + 1] == str(expected_records)
    assert argv[argv.index("--output-root") + 1] == str(expected_output)
    assert f"score_sweep_json={expected_output}/score_sweep.json" in result.stdout


def test_support_ensemble_sweep_resolves_relative_candidate_records_before_repo_cd(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_support_ensemble_score_sweep.sh"
    caller_dir = tmp_path / "caller"
    run_root = caller_dir / "relative_run"
    record_path = tmp_path / "record.json"
    fake_python = tmp_path / "fake_python.py"
    caller_dir.mkdir()
    run_root.mkdir()
    (run_root / "candidate_records.jsonl").write_text("", encoding="utf-8")
    _write_fake_python(fake_python, record_path)

    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "CANDIDATE_RECORDS": "relative_run/candidate_records.jsonl",
    }
    result = subprocess.run(
        ["bash", str(script)],
        cwd=caller_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    expected_records = run_root / "candidate_records.jsonl"
    expected_output = run_root / "support_ensemble_score_sweep"
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    argv = recorded["argv"]

    assert result.returncode == 0, result.stdout + result.stderr
    assert argv[argv.index("--candidate-records") + 1] == str(expected_records)
    assert argv[argv.index("--output-root") + 1] == str(expected_output)
    assert f"score_sweep_json={expected_output}/score_sweep.json" in result.stdout


def test_support_ensemble_sweep_resolves_relative_output_root_before_repo_cd(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_support_ensemble_score_sweep.sh"
    caller_dir = tmp_path / "caller"
    run_root = caller_dir / "relative_run"
    output_root = caller_dir / "relative_out"
    record_path = tmp_path / "record.json"
    fake_python = tmp_path / "fake_python.py"
    caller_dir.mkdir()
    run_root.mkdir()
    output_root.mkdir()
    (run_root / "candidate_records.jsonl").write_text("", encoding="utf-8")
    _write_fake_python(fake_python, record_path)

    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "RUN_ROOT": "relative_run",
        "OUTPUT_ROOT": "relative_out",
    }
    result = subprocess.run(
        ["bash", str(script)],
        cwd=caller_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    expected_records = run_root / "candidate_records.jsonl"
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    argv = recorded["argv"]

    assert result.returncode == 0, result.stdout + result.stderr
    assert argv[argv.index("--candidate-records") + 1] == str(expected_records)
    assert argv[argv.index("--output-root") + 1] == str(output_root)
    assert f"score_sweep_json={output_root}/score_sweep.json" in result.stdout
