from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from robust_loop_verifier import rover_evaluation
from robust_loop_verifier.rover_annotation import (
    ANNOTATION_SEAL_VERSION,
    ANNOTATION_VERSION,
)
from robust_loop_verifier.rover_benchmark import BenchmarkPair
from robust_loop_verifier.rover_evaluation import (
    evaluate_methods,
    evaluate_score_file,
    read_complete_annotations,
    render_table1_markdown,
    validate_scores,
    validate_sequence_groups,
    verify_score_manifest,
    write_metrics_outputs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(pair_id: str, sequence: str = "s0") -> dict:
    return {
        "pair_id": pair_id,
        "dataset": "d",
        "platform": "p",
        "sequence": sequence,
    }


def _sequence_order(count: int = 10) -> list[str]:
    return [f"d/p/s{index}" for index in range(count)]


def _benchmark_pairs(sequence_order: list[str] | None = None) -> list[dict]:
    order = _sequence_order() if sequence_order is None else sequence_order
    return [
        {
            "pair_id": f"pair-{index}",
            "dataset": key.split("/")[0],
            "platform": key.split("/")[1],
            "sequence": key.split("/")[2],
        }
        for index, key in enumerate(order)
    ]


def _write_benchmark(
    root: Path,
    *,
    pairs: list[dict] | None = None,
    sequence_order: list[str] | None = None,
    labels: list[int] | None = None,
) -> tuple[list[dict], list[str]]:
    order = _sequence_order() if sequence_order is None else sequence_order
    pair_rows = _benchmark_pairs(order) if pairs is None else pairs
    pairs_path = root / "benchmark_pairs.jsonl"
    annotations_path = root / "annotations.jsonl"
    manifest_path = root / "manifest.json"
    seal_path = root / "annotation_seal.json"
    _write_jsonl(pairs_path, pair_rows)
    _write_json(
        manifest_path,
        {
            "benchmark_version": "benchmark_v1",
            "pair_manifest_sha256": _sha256(pairs_path),
            "sequence_order": order,
            "sequences": [
                {
                    "dataset": key.split("/")[0],
                    "platform": key.split("/")[1],
                    "sequence": key.split("/")[2],
                }
                for key in order
            ],
        },
    )
    label_values = (
        [index % 2 for index in range(len(pair_rows))] if labels is None else labels
    )
    annotation_rows = [
        {
            "pair_id": pair["pair_id"],
            "label": label,
            "annotated_at": f"2026-06-10T00:00:{index:02d}+00:00",
            "annotation_version": ANNOTATION_VERSION,
        }
        for index, (pair, label) in enumerate(zip(pair_rows, label_values))
    ]
    _write_jsonl(annotations_path, annotation_rows)
    _write_json(
        seal_path,
        {
            "version": ANNOTATION_SEAL_VERSION,
            "annotation_version": ANNOTATION_VERSION,
            "benchmark_version": "benchmark_v1",
            "pair_manifest_sha256": _sha256(pairs_path),
            "annotation_sha256": _sha256(annotations_path),
            "count": len(annotation_rows),
            "completed_at": "2026-06-10T00:01:00+00:00",
        },
    )
    return pair_rows, order


def _write_score(
    root: Path,
    method_name: str,
    relative_path: str,
    pairs: list[dict],
    *,
    rows: list[dict] | None = None,
    manifest_updates: dict | None = None,
) -> Path:
    score_path = root / relative_path
    score_rows = (
        [
            {"pair_id": pair["pair_id"], "score": float(index), "status": "ok"}
            for index, pair in enumerate(pairs)
        ]
        if rows is None
        else rows
    )
    _write_jsonl(score_path, score_rows)
    manifest = {
        "method": method_name,
        "pair_manifest_sha256": _sha256(root / "benchmark_pairs.jsonl"),
        "score_file_sha256": _sha256(score_path),
        "source_commit": "abc123",
        "command": f"score --method {method_name}",
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    _write_json(score_path.with_suffix(".manifest.json"), manifest)
    return score_path


def test_score_validation_requires_exact_pair_coverage():
    pairs = [{"pair_id": "a"}, {"pair_id": "b"}]

    with pytest.raises(ValueError, match="missing"):
        validate_scores(pairs, [{"pair_id": "a", "score": 0.9, "status": "ok"}])
    with pytest.raises(ValueError, match="unknown"):
        validate_scores(
            pairs,
            [
                {"pair_id": "a", "score": 0.9, "status": "ok"},
                {"pair_id": "b", "score": 0.8, "status": "ok"},
                {"pair_id": "x", "score": 0.1, "status": "ok"},
            ],
        )


def test_score_validation_rejects_duplicate_rows():
    with pytest.raises(ValueError, match="duplicate"):
        validate_scores(
            [{"pair_id": "a"}],
            [
                {"pair_id": "a", "score": 0.9, "status": "ok"},
                {"pair_id": "a", "score": 0.8, "status": "ok"},
            ],
        )


def test_score_validation_rejects_failed_row_without_score():
    with pytest.raises(ValueError, match="score.*None|None.*score"):
        validate_scores([{"pair_id": "a"}], [{"pair_id": "a", "status": "failed"}])


@pytest.mark.parametrize(
    "row, message",
    [
        ({"pair_id": "a", "score": 0.1}, "status"),
        ({"pair_id": "a", "score": 0.1, "status": "unknown"}, "status"),
        ({"pair_id": "a", "score": None, "status": "ok"}, "finite numeric"),
        ({"pair_id": "a", "score": True, "status": "ok"}, "finite numeric"),
        ({"pair_id": "a", "score": math.nan, "status": "ok"}, "finite numeric"),
        ({"pair_id": "a", "score": 0.1, "status": "failed"}, "None"),
    ],
)
def test_score_validation_rejects_invalid_status_score_combinations(row, message):
    with pytest.raises(ValueError, match=message):
        validate_scores([{"pair_id": "a"}], [row])


def test_failed_rows_share_one_tied_worst_score():
    pairs = [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "c"}]
    rows = [
        {"pair_id": "a", "score": 0.5, "status": "ok"},
        {"pair_id": "b", "score": None, "status": "failed"},
        {"pair_id": "c", "score": None, "status": "failed"},
    ]

    scores = validate_scores(pairs, rows)

    assert scores["b"] == scores["c"] == -math.pi / 2.0
    assert scores["b"] < scores["a"]
    assert scores["a"] == math.atan(0.5)


def test_failure_transform_handles_lowest_finite_float():
    scores = validate_scores(
        [{"pair_id": "ok"}, {"pair_id": "failed"}],
        [
            {"pair_id": "ok", "score": -sys.float_info.max, "status": "ok"},
            {"pair_id": "failed", "score": None, "status": "failed"},
        ],
    )

    assert scores["failed"] == -math.pi / 2.0
    assert scores["ok"] > scores["failed"]


def test_failure_transform_preserves_extreme_successful_score_order():
    scores = validate_scores(
        [
            {"pair_id": "most_negative"},
            {"pair_id": "less_negative"},
            {"pair_id": "failed"},
        ],
        [
            {
                "pair_id": "most_negative",
                "score": -sys.float_info.max,
                "status": "ok",
            },
            {"pair_id": "less_negative", "score": -1e20, "status": "ok"},
            {"pair_id": "failed", "score": None, "status": "failed"},
        ],
    )

    assert scores["failed"] < scores["most_negative"] < scores["less_negative"]


def test_all_failed_scores_tie_at_zero():
    scores = validate_scores(
        [{"pair_id": "a"}, {"pair_id": "b"}],
        [
            {"pair_id": "a", "score": None, "status": "failed"},
            {"pair_id": "b", "score": None, "status": "failed"},
        ],
    )

    assert scores == {"a": 0.0, "b": 0.0}


def test_successful_scores_are_not_transformed_without_failures():
    scores = validate_scores(
        [{"pair_id": "a"}],
        [{"pair_id": "a", "score": 3.0, "status": "ok"}],
    )

    assert scores == {"a": 3.0}


def test_read_annotations_requires_complete_manifest_order(tmp_path):
    pairs, _ = _write_benchmark(tmp_path)
    annotations_path = tmp_path / "annotations.jsonl"
    rows = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(annotations_path, rows[:-1])
    seal = json.loads((tmp_path / "annotation_seal.json").read_text(encoding="utf-8"))
    seal["annotation_sha256"] = _sha256(annotations_path)
    seal["count"] = len(rows) - 1
    _write_json(tmp_path / "annotation_seal.json", seal)

    with pytest.raises(ValueError, match="coverage"):
        read_complete_annotations(
            pairs,
            tmp_path / "manifest.json",
            annotations_path,
            tmp_path / "annotation_seal.json",
        )


def test_read_annotations_rejects_broken_seal_before_use(tmp_path):
    pairs, _ = _write_benchmark(tmp_path)
    seal = json.loads((tmp_path / "annotation_seal.json").read_text(encoding="utf-8"))
    seal["annotation_sha256"] = "broken"
    _write_json(tmp_path / "annotation_seal.json", seal)

    with pytest.raises(ValueError, match="annotation hash"):
        read_complete_annotations(
            pairs,
            tmp_path / "manifest.json",
            tmp_path / "annotations.jsonl",
            tmp_path / "annotation_seal.json",
        )


def test_sequence_groups_require_exact_ten_nonempty_canonical_groups():
    order = _sequence_order()
    validate_sequence_groups(_benchmark_pairs(order), order)

    with pytest.raises(ValueError, match="exactly 10"):
        validate_sequence_groups(_benchmark_pairs(order[:-1]), order[:-1])
    with pytest.raises(ValueError, match="missing"):
        validate_sequence_groups(_benchmark_pairs(order[:-1]), order)
    with pytest.raises(ValueError, match="extra"):
        validate_sequence_groups(
            [*_benchmark_pairs(order), _pair("extra", "extra")],
            order,
        )
    with pytest.raises(ValueError, match="order"):
        validate_sequence_groups(list(reversed(_benchmark_pairs(order))), order)


def test_evaluation_is_per_sequence_then_macro_averaged():
    pairs = [
        _pair("a", "s1"),
        _pair("b", "s1"),
        _pair("c", "s2"),
        _pair("d", "s2"),
    ]
    labels = {"a": 1, "b": 0, "c": 0, "d": 1}
    scores = {"a": 0.9, "b": 0.1, "c": 0.9, "d": 0.1}

    result = evaluate_score_file(
        "method",
        pairs,
        labels,
        scores,
        sequence_order=["d/p/s1", "d/p/s2"],
    )

    assert result["sequences"]["d/p/s1"]["AP"] == 1.0
    assert result["sequences"]["d/p/s2"]["AP"] == 0.5
    assert result["macro_average"]["AP"] == 0.75
    assert result["sequences"]["d/p/s1"]["candidate_count"] == 2
    assert result["sequences"]["d/p/s1"]["positive_count"] == 1


def test_zero_positive_sequence_and_tied_scores_use_existing_metrics():
    pairs = [
        _pair("a", "s1"),
        _pair("b", "s1"),
        _pair("c", "s2"),
        _pair("d", "s2"),
    ]
    result = evaluate_score_file(
        "method",
        pairs,
        {"a": 0, "b": 0, "c": 1, "d": 0},
        {"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.5},
        ["d/p/s1", "d/p/s2"],
    )

    assert result["sequences"]["d/p/s1"]["AP"] == 0.0
    assert result["sequences"]["d/p/s1"]["MR@100P"] == 0.0
    assert result["sequences"]["d/p/s2"]["AP"] == 0.5
    assert result["sequences"]["d/p/s2"]["MR@100P"] == 0.0


def test_score_manifest_requires_hashes_provenance_and_method_consistency(tmp_path):
    pairs, _ = _write_benchmark(tmp_path)
    score_path = _write_score(tmp_path, "Method", "scores/method.jsonl", pairs)
    manifest_path = score_path.with_suffix(".manifest.json")
    verify_score_manifest(
        tmp_path / "benchmark_pairs.jsonl",
        score_path,
        manifest_path,
        method_name="Method",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("pair_manifest_sha256", "score_file_sha256", "source_commit", "command"):
        broken = dict(manifest)
        broken.pop(field)
        _write_json(manifest_path, broken)
        with pytest.raises(ValueError, match=field):
            verify_score_manifest(
                tmp_path / "benchmark_pairs.jsonl",
                score_path,
                manifest_path,
                method_name="Method",
            )
    _write_json(manifest_path, {**manifest, "score_file_sha256": "broken"})
    with pytest.raises(ValueError, match="score-file hash"):
        verify_score_manifest(
            tmp_path / "benchmark_pairs.jsonl",
            score_path,
            manifest_path,
        )
    _write_json(manifest_path, {**manifest, "method": "Other"})
    with pytest.raises(ValueError, match="method"):
        verify_score_manifest(
            tmp_path / "benchmark_pairs.jsonl",
            score_path,
            manifest_path,
            method_name="Method",
        )


def test_score_manifest_score_file_is_relative_to_benchmark_root(tmp_path):
    pairs, _ = _write_benchmark(tmp_path)
    score_path = _write_score(
        tmp_path,
        "Method",
        "alternate/nested/method.jsonl",
        pairs,
        manifest_updates={"score_file": "alternate/nested/method.jsonl"},
    )

    verify_score_manifest(
        tmp_path / "benchmark_pairs.jsonl",
        score_path,
        score_path.with_suffix(".manifest.json"),
        method_name="Method",
    )


def test_evaluate_methods_requires_score_manifest_before_output(tmp_path):
    pairs, _ = _write_benchmark(tmp_path)
    score_path = tmp_path / "scores" / "method.jsonl"
    _write_jsonl(
        score_path,
        [
            {"pair_id": pair["pair_id"], "score": 0.1, "status": "ok"}
            for pair in pairs
        ],
    )

    with pytest.raises(FileNotFoundError, match="manifest"):
        evaluate_methods(tmp_path, [("Method", score_path)])


def test_evaluate_methods_checks_seal_before_method_files(tmp_path):
    _write_benchmark(tmp_path)
    seal = json.loads((tmp_path / "annotation_seal.json").read_text(encoding="utf-8"))
    seal["annotation_sha256"] = "broken"
    _write_json(tmp_path / "annotation_seal.json", seal)

    with pytest.raises(ValueError, match="annotation hash"):
        evaluate_methods(tmp_path, [("Method", tmp_path / "missing.jsonl")])


def test_evaluate_methods_preserves_supplied_method_order(tmp_path):
    pairs, order = _write_benchmark(tmp_path)
    second = _write_score(tmp_path, "Second", "scores/second.jsonl", pairs)
    first = _write_score(tmp_path, "First", "scores/first.jsonl", pairs)

    results = evaluate_methods(tmp_path, [("Second", second), ("First", first)])

    assert results["sequence_order"] == order
    assert results["method_order"] == ["Second", "First"]
    assert list(results["methods"]) == ["Second", "First"]


def _table_results() -> dict:
    sequence_order = ["d/p/s1"]
    method_order = ["A", "B", "C"]
    values = {
        "A": (0.9, 0.2),
        "B": (0.9, 0.9),
        "C": (0.8, 0.5),
    }
    return {
        "sequence_order": sequence_order,
        "method_order": method_order,
        "methods": {
            method: {
                "method": method,
                "sequences": {
                    "d/p/s1": {
                        "AP": ap,
                        "MR@100P": mr,
                        "candidate_count": 4,
                        "positive_count": 2,
                    }
                },
                "macro_average": {"AP": ap, "MR@100P": mr},
            }
            for method, (ap, mr) in values.items()
        },
    }


def test_table_has_counts_order_and_independent_deterministic_highlights():
    results = _table_results()

    markdown = render_table1_markdown(
        results,
        results["sequence_order"],
        results["method_order"],
    )

    assert "Candidates: 4; positives: 2" in markdown
    assert markdown.index("| A |") < markdown.index("| B |") < markdown.index("| C |")
    row_a = next(line for line in markdown.splitlines() if line.startswith("| A |"))
    row_b = next(line for line in markdown.splitlines() if line.startswith("| B |"))
    row_c = next(line for line in markdown.splitlines() if line.startswith("| C |"))
    assert "**0.9000** / 0.2000" in row_a
    assert "<u>0.9000</u> / **0.9000**" in row_b
    assert "0.8000 / <u>0.5000</u>" in row_c


def test_metrics_outputs_are_deterministic_and_complete(tmp_path):
    results = _table_results()
    output_dir = tmp_path / "metrics"

    write_metrics_outputs(output_dir, results)

    csv_lines = (output_dir / "metrics_per_sequence.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    table = (output_dir / "table1.md").read_text(encoding="utf-8")
    assert csv_lines[0] == "method,sequence,candidate_count,positive_count,AP,MR@100P"
    assert csv_lines[1].startswith("A,d/p/s1,4,2,")
    assert csv_lines[2].startswith("B,d/p/s1,4,2,")
    assert summary["method_order"] == ["A", "B", "C"]
    assert table == render_table1_markdown(results, ["d/p/s1"], ["A", "B", "C"])


def test_output_validation_failure_creates_no_partial_files(tmp_path):
    results = _table_results()
    results["methods"]["A"]["sequences"]["d/p/s1"]["candidate_count"] = 3
    output_dir = tmp_path / "metrics"

    with pytest.raises(ValueError, match="counts"):
        write_metrics_outputs(output_dir, results)

    assert not output_dir.exists()


def test_output_write_failure_preserves_existing_final_files(tmp_path):
    results = _table_results()
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    originals = {
        "metrics_per_sequence.csv": "old csv\n",
        "metrics_summary.json": '{"old": true}\n',
        "table1.md": "old markdown\n",
    }
    for filename, content in originals.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    blocking_file = output_dir / "metrics_summary.json"
    blocking_file.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(OSError):
            write_metrics_outputs(output_dir, results)
    finally:
        tmp_path.chmod(0o755)
        blocking_file.chmod(0o644)

    for filename, content in originals.items():
        assert (output_dir / filename).read_text(encoding="utf-8") == content


def test_interrupt_after_existing_final_moves_to_backup_restores_final(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    final_path = output_dir / "metrics_summary.json"
    final_path.write_text('{"old": true}\n', encoding="utf-8")

    original_replace = rover_evaluation.os.replace
    interrupted = False

    def interrupt_after_backup_move(src: Path, dst: Path) -> None:
        nonlocal interrupted
        source_path = Path(src)
        destination_path = Path(dst)
        original_replace(src, dst)
        if source_path == final_path and destination_path.parent.name == "backup":
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(rover_evaluation.os, "replace", interrupt_after_backup_move)

    with pytest.raises(KeyboardInterrupt):
        rover_evaluation._write_payload_set_atomically(
            output_dir,
            {"metrics_summary.json": '{"new": true}\n'},
        )

    assert interrupted
    assert final_path.exists()
    assert final_path.read_text(encoding="utf-8") == '{"old": true}\n'


def test_keyboard_interrupt_during_publish_restores_existing_final_files(
    tmp_path,
    monkeypatch,
):
    results = _table_results()
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    originals = {
        "metrics_per_sequence.csv": "old csv\n",
        "metrics_summary.json": '{"old": true}\n',
        "table1.md": "old markdown\n",
    }
    for filename, content in originals.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    original_replace = rover_evaluation.os.replace
    replace_calls = 0
    interrupted_at = None

    def interrupt_after_first_final_replacement(src: Path, dst: Path) -> None:
        nonlocal interrupted_at, replace_calls
        replace_calls += 1
        if replace_calls == 3:
            interrupted_at = replace_calls
            raise KeyboardInterrupt
        original_replace(src, dst)

    monkeypatch.setattr(
        rover_evaluation.os,
        "replace",
        interrupt_after_first_final_replacement,
    )

    with pytest.raises(KeyboardInterrupt):
        write_metrics_outputs(output_dir, results)

    assert interrupted_at == 3
    for filename, content in originals.items():
        assert (output_dir / filename).read_text(encoding="utf-8") == content


def test_cli_method_parser_rejects_malformed_and_duplicate_entries():
    evaluator = pytest.importorskip(
        "robust_loop_verification_scripts.evaluate_rover_aligned_benchmark"
    )

    with pytest.raises(ValueError, match="NAME=PATH"):
        evaluator.parse_method_specs(["missing-separator"])
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.parse_method_specs(["A=one=two"])
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.parse_method_specs(["A=one.jsonl", "A=two.jsonl"])


def test_cli_resolves_relative_paths_and_fails_before_writing(tmp_path):
    evaluator = pytest.importorskip(
        "robust_loop_verification_scripts.evaluate_rover_aligned_benchmark"
    )
    _write_benchmark(tmp_path)
    seal = json.loads((tmp_path / "annotation_seal.json").read_text(encoding="utf-8"))
    seal["annotation_sha256"] = "broken"
    _write_json(tmp_path / "annotation_seal.json", seal)
    output_dir = tmp_path / "metrics"

    with pytest.raises(ValueError, match="annotation hash"):
        evaluator.main(
            [
                str(tmp_path),
                "--method",
                "Method=scores/method.jsonl",
                "--output-dir",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()


def test_cli_relative_benchmark_root_resolves_relative_method_once(
    tmp_path,
    monkeypatch,
):
    evaluator = pytest.importorskip(
        "robust_loop_verification_scripts.evaluate_rover_aligned_benchmark"
    )
    benchmark_root = tmp_path / "benchmark"
    pairs, _ = _write_benchmark(benchmark_root)
    _write_score(benchmark_root, "Method", "scores/foo.jsonl", pairs)
    monkeypatch.chdir(tmp_path)

    assert (
        evaluator.main(
            [
                benchmark_root.name,
                "--method",
                "Method=scores/foo.jsonl",
                "--output-dir",
                "metrics",
            ]
        )
        == 0
    )

    output_dir = tmp_path / "metrics"
    assert (output_dir / "metrics_per_sequence.csv").is_file()
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["method_order"] == ["Method"]
    assert (output_dir / "table1.md").is_file()


def test_task2_manifest_exposes_canonical_sequence_order():
    builder = pytest.importorskip(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    sequences = [
        builder.BenchmarkSequence("d", "p", f"s{index}", Path(f"/cache/s{index}"))
        for index in range(10)
    ]
    keys = [(sequence.dataset, sequence.platform, sequence.sequence) for sequence in sequences]
    manifest = builder.build_benchmark_manifest(
        config={},
        sequences=sequences,
        helper_fingerprints={
            "vocabulary_path": "/vocab",
            "vocabulary_sha256": "v",
            "orb_slam3_git_commit": "commit",
            "helper_source_path": "/source",
            "helper_source_sha256": "source",
            "helper_binary_path": "/binary",
            "helper_binary_sha256": "binary",
            "dbow2_shared_library_path": "/library",
            "dbow2_shared_library_sha256": "library",
        },
        recent_exclusion_by_sequence={key: 5 for key in keys},
        eligible_query_counts={key: 1 for key in keys},
        selected_query_counts={key: 1 for key in keys},
        pair_counts={key: 10 for key in keys},
        pair_manifest_sha256="pairs",
    )

    assert manifest["sequence_order"] == _sequence_order()


def test_task2_dbow2_score_manifest_is_verifiable(tmp_path):
    builder = pytest.importorskip(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    pairs_path = tmp_path / "benchmark_pairs.jsonl"
    pair = BenchmarkPair(
        pair_id="pair-a",
        dataset="d",
        platform="p",
        sequence="s",
        query_idx=10,
        candidate_idx=1,
        rank=1,
        dbow2_score=0.5,
        query_image="q.png",
        candidate_image="c.png",
        query_context=(None, "q.png", None),
        candidate_context=(None, "c.png", None),
    )
    _write_jsonl(pairs_path, [pair.__dict__])
    builder._write_score_files(
        tmp_path,
        [pair],
        {"orb_slam3_git_commit": "commit"},
        pair_manifest_sha256=_sha256(pairs_path),
    )

    verify_score_manifest(
        pairs_path,
        tmp_path / "scores" / "dbow2.jsonl",
        tmp_path / "scores" / "dbow2.manifest.json",
        method_name="DBoW2",
    )
