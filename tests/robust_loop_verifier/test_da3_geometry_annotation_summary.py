from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCRIPT = (
    LOOPANYTHING_ROOT
    / "robust_loop_verification_scripts"
    / "summarize_da3_geometry_annotations.py"
)
WRAPPER_SCRIPT = (
    LOOPANYTHING_ROOT
    / "robust_loop_verification_scripts"
    / "run_da3_geometry_annotation_prototype.sh"
)


def _load_summary():
    spec = importlib.util.spec_from_file_location(
        "summarize_da3_geometry_annotations",
        SUMMARY_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _pair_ids_sha256(pair_ids: list[str]) -> str:
    return _sha256("".join(f"{pair_id}\n" for pair_id in pair_ids).encode("utf-8"))


def _canonical_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _mock_da3_config() -> dict:
    return {
        "backend": "mock",
        "device": "cpu",
        "verifier_configs": {"d": "configs/d.yaml"},
        "verifier_config_sha256": {"d": "a" * 64},
        "salad_score_file": "scores/salad.jsonl",
        "salad_score_file_sha256": "b" * 64,
        "salad_score_manifest": "scores/salad.manifest.json",
        "salad_score_manifest_sha256": "c" * 64,
        "da3_model_name": None,
        "da3_model_dir": None,
        "da3_model_path": None,
        "da3_model_path_sha256": None,
        "da3_checkpoint": None,
        "da3_checkpoint_sha256": None,
        "da3_snapshot": None,
        "da3_snapshot_sha256": None,
        "da3_runtime_by_dataset": {
            "d": {
                "process_res": 504,
                "ref_view_strategy": "first",
                "triplet_batch_size": 4,
            }
        },
    }


def _write_annotation_bundle(
    root: Path,
    pairs_content: bytes,
    manifest_content: bytes,
    labels: list[int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_pairs.jsonl").write_bytes(pairs_content)
    (root / "manifest.json").write_bytes(manifest_content)
    pairs = [json.loads(line) for line in pairs_content.decode("utf-8").splitlines() if line]
    annotations = [
        {
            "pair_id": pair["pair_id"],
            "label": label,
            "annotated_at": f"2026-06-13T00:00:0{index}+00:00",
            "annotation_version": 1,
        }
        for index, (pair, label) in enumerate(zip(pairs, labels))
    ]
    annotation_content = _jsonl_bytes(annotations)
    (root / "annotations.jsonl").write_bytes(annotation_content)
    manifest = json.loads(manifest_content)
    seal = {
        "version": "rover_annotation_seal_v1",
        "annotation_version": 1,
        "benchmark_version": manifest["benchmark_version"],
        "pair_manifest_sha256": _sha256(pairs_content),
        "annotation_sha256": _sha256(annotation_content),
        "count": len(annotations),
        "completed_at": "2026-06-13T01:00:00+00:00",
    }
    (root / "annotation_seal.json").write_bytes(_json_bytes(seal))


def _prediction(
    pair: dict,
    automatic_label: int,
    *,
    factor_status: str = "ok",
    translation_error_m: float | None = 0.0,
    rotation_error_deg: float | None = 0.0,
    translation_direction_error_deg: float | None = 0.0,
) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "automatic_label": automatic_label,
        "factor_status": factor_status,
        "support_idx": None if factor_status == "support_failed" else 1,
        "sim3_valid": factor_status == "ok",
        "translation_error_m": translation_error_m,
        "rotation_error_deg": rotation_error_deg,
        "translation_direction_error_deg": translation_direction_error_deg,
    }


def _bundle(
    tmp_path: Path,
    *,
    old_labels: list[int] | None = None,
    reviewed_labels: list[int] | None = None,
    pair_limit: int | None = None,
) -> tuple[Path, Path, list[dict]]:
    benchmark_root = tmp_path / "benchmark"
    output_root = tmp_path / "geometry"
    pairs = [
        {
            "pair_id": f"p{index}",
            "dataset": "d",
            "platform": "robot",
            "sequence": "s1" if index < 4 else "s2",
            "query_idx": index,
            "candidate_idx": index + 10,
            "rank": 1,
        }
        for index in range(8)
    ]
    pairs_content = _jsonl_bytes(pairs)
    manifest = {
        "benchmark_version": "benchmark_v1",
        "pair_manifest_sha256": _sha256(pairs_content),
    }
    manifest_content = _json_bytes(manifest)
    old_labels = old_labels or [0, 1, 1, 0, 0, 0, 0, 0]
    reviewed_labels = reviewed_labels or [1, 1, 0, 0, 0, 0, 1, 1]
    _write_annotation_bundle(
        benchmark_root,
        pairs_content,
        manifest_content,
        old_labels,
    )
    _write_annotation_bundle(
        output_root,
        pairs_content,
        manifest_content,
        reviewed_labels,
    )

    all_predictions = [
        _prediction(
            pairs[0],
            1,
            translation_error_m=1.0,
            rotation_error_deg=10.0,
            translation_direction_error_deg=5.0,
        ),
        _prediction(
            pairs[1],
            1,
            translation_error_m=3.0,
            rotation_error_deg=30.0,
            translation_direction_error_deg=None,
        ),
        _prediction(
            pairs[2],
            1,
            factor_status="support_failed",
            translation_error_m=None,
            rotation_error_deg=None,
            translation_direction_error_deg=None,
        ),
        _prediction(
            pairs[3],
            0,
            factor_status="da3_failed",
            translation_error_m=None,
            rotation_error_deg=None,
            translation_direction_error_deg=None,
        ),
        _prediction(
            pairs[4],
            0,
            factor_status="sim3_failed",
            translation_error_m=None,
            rotation_error_deg=None,
            translation_direction_error_deg=None,
        ),
        _prediction(
            pairs[5],
            0,
            factor_status="invalid_loop_factor",
            translation_error_m=None,
            rotation_error_deg=None,
            translation_direction_error_deg=None,
        ),
        _prediction(
            pairs[6],
            1,
            translation_error_m=2.0,
            rotation_error_deg=20.0,
            translation_direction_error_deg=10.0,
        ),
        _prediction(
            pairs[7],
            0,
            translation_error_m=4.0,
            rotation_error_deg=40.0,
            translation_direction_error_deg=20.0,
        ),
    ]
    predictions = all_predictions[:pair_limit] if pair_limit is not None else all_predictions
    prediction_content = _jsonl_bytes(predictions)
    (output_root / "geometry_predictions.jsonl").write_bytes(prediction_content)
    thresholds = {
        "max_translation_error_m": 1.0,
        "max_rotation_error_deg": 15.0,
        "max_translation_direction_error_deg": 20.0,
        "min_direction_baseline_m": 0.5,
    }
    request_contract = {
        "backend": "mock",
        "device": "cpu",
        "pair_limit": pair_limit,
        "thresholds": thresholds,
    }
    da3_config = _mock_da3_config()
    target_pairs = pairs[:pair_limit] if pair_limit is not None else pairs
    prediction_manifest = {
        "format_version": 1,
        "prediction_file": "geometry_predictions.jsonl",
        "source_pair_manifest_sha256": _sha256(pairs_content),
        "source_manifest_sha256": _sha256(manifest_content),
        "source_pair_count": len(pairs),
        "target_pair_count": len(target_pairs),
        "target_pair_ids_sha256": _pair_ids_sha256([pair["pair_id"] for pair in target_pairs]),
        "pair_limit": pair_limit,
        "thresholds": thresholds,
        "request_contract": request_contract,
        "da3_config": da3_config,
        "da3_config_sha256": _canonical_sha256(da3_config),
        "source_commit": "test-commit",
        "command": ["build_da3_geometry_annotation.py", str(benchmark_root)],
        "prediction_record_count": len(predictions),
        "prediction_file_sha256": _sha256(prediction_content),
        "complete": True,
    }
    (output_root / "geometry_prediction_manifest.json").write_bytes(
        _json_bytes(prediction_manifest)
    )
    return benchmark_root, output_root, pairs


def _run_summary(module, benchmark_root: Path, output_root: Path) -> dict:
    assert module.main([str(benchmark_root), "--geometry-root", str(output_root)]) == 0
    return json.loads((output_root / "label_comparison.json").read_text(encoding="utf-8"))


def test_threshold_validation_accepts_scale_adaptive_schema():
    module = _load_summary()
    thresholds = {
        "min_translation_error_m": 1.0,
        "max_translation_error_m": 5.0,
        "translation_error_scale_ratio": 0.2,
        "max_rotation_error_deg": 15.0,
        "max_translation_direction_error_deg": 20.0,
        "min_direction_baseline_m": 0.5,
    }

    assert module._validate_thresholds(thresholds, "thresholds") == thresholds


def test_summary_reports_sequence_overall_macro_transitions_and_pose_statistics(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)

    report = _run_summary(module, benchmark_root, output_root)

    by_sequence = {row["sequence"]: row for row in report["sequences"]}
    s1 = by_sequence["s1"]
    assert s1["pair_count"] == 4
    assert s1["old_positive_count"] == 2
    assert s1["old_positive_rate"] == 0.5
    assert s1["automatic_positive_count"] == 3
    assert s1["automatic_positive_rate"] == 0.75
    assert s1["reviewed_positive_count"] == 2
    assert s1["reviewed_positive_rate"] == 0.5
    assert s1["old_negative_to_reviewed_positive"] == 1
    assert s1["old_positive_to_reviewed_negative"] == 1
    assert s1["automatic_precision"] == pytest.approx(2 / 3)
    assert s1["automatic_recall"] == 1.0
    assert s1["support_failed_count"] == 1
    assert s1["da3_failed_count"] == 1
    assert s1["sim3_failed_count"] == 0
    assert s1["other_factor_failure_count"] == 0
    assert s1["reviewed_positive_translation_error_m_median"] == 2.0
    assert s1["reviewed_positive_translation_error_m_p90"] == pytest.approx(2.8)
    assert s1["reviewed_negative_translation_error_m_median"] is None

    s2 = by_sequence["s2"]
    assert s2["automatic_precision"] == 1.0
    assert s2["automatic_recall"] == 0.5
    assert s2["sim3_failed_count"] == 1
    assert s2["other_factor_failure_count"] == 1
    assert s2["reviewed_positive_rotation_error_deg_median"] == 30.0
    assert s2["reviewed_positive_rotation_error_deg_p90"] == 38.0
    assert s2["reviewed_negative_translation_error_m_median"] is None

    overall = report["overall"]
    assert overall["pair_count"] == 8
    assert overall["old_positive_count"] == 2
    assert overall["reviewed_positive_count"] == 4
    assert overall["automatic_precision"] == 0.75
    assert overall["automatic_recall"] == 0.75
    assert overall["support_failed_count"] == 1
    assert overall["da3_failed_count"] == 1
    assert overall["sim3_failed_count"] == 1
    assert overall["other_factor_failure_count"] == 1

    macro = report["macro"]
    assert macro["sequence_count"] == 2
    assert macro["pair_count"] == 8
    assert macro["old_positive_rate"] == 0.25
    assert macro["reviewed_positive_rate"] == 0.5
    assert macro["automatic_precision"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert macro["automatic_recall"] == 0.75
    assert macro["reviewed_positive_translation_error_m_median"] == 2.5
    assert macro["reviewed_positive_translation_error_m_p90"] == 3.3
    assert report["macro_definition"] == {
        "count_fields": "sum across sequences",
        "rate_precision_recall_pose_fields": ("unweighted mean over sequences with finite values"),
        "pose_fields": ("mean of per-sequence percentiles, not pooled percentiles"),
    }

    provenance = report["provenance"]
    assert provenance["thresholds"]["max_translation_error_m"] == 1.0
    assert provenance["source_pair_manifest_sha256"] == _sha256(
        (benchmark_root / "benchmark_pairs.jsonl").read_bytes()
    )
    for field in (
        "geometry_prediction_manifest_sha256",
        "old_annotation_seal_sha256",
        "new_annotation_seal_sha256",
    ):
        assert len(provenance[field]) == 64

    csv_text = (output_root / "label_comparison.csv").read_text(encoding="utf-8")
    markdown = (output_root / "label_comparison.md").read_text(encoding="utf-8")
    assert "macro" in csv_text
    assert "N/A" in csv_text
    assert "| macro |" in markdown
    assert "N/A" in markdown
    assert "mean of per-sequence percentiles, not pooled percentiles" in markdown


def test_zero_precision_and_recall_denominators_are_none_and_na(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(
        tmp_path,
        old_labels=[0] * 8,
        reviewed_labels=[0] * 8,
    )
    predictions = [
        json.loads(line)
        for line in (output_root / "geometry_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for prediction in predictions:
        prediction["automatic_label"] = 0
    content = _jsonl_bytes(predictions)
    (output_root / "geometry_predictions.jsonl").write_bytes(content)
    manifest_path = output_root / "geometry_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_file_sha256"] = _sha256(content)
    manifest_path.write_bytes(_json_bytes(manifest))

    report = _run_summary(module, benchmark_root, output_root)

    assert report["overall"]["automatic_precision"] is None
    assert report["overall"]["automatic_recall"] is None
    assert report["macro"]["automatic_precision"] is None
    assert report["macro"]["automatic_recall"] is None
    assert all(row["automatic_precision"] is None for row in report["sequences"])
    assert all(row["automatic_recall"] is None for row in report["sequences"])
    assert "N/A" in (output_root / "label_comparison.md").read_text(encoding="utf-8")


def test_nonfinite_prediction_pose_errors_are_ignored_not_rejected(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)
    prediction_path = output_root / "geometry_predictions.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    for index in (0, 1, 6, 7):
        predictions[index]["translation_error_m"] = [
            float("inf"),
            float("nan"),
            float("-inf"),
            float("nan"),
        ][[0, 1, 6, 7].index(index)]
    prediction_content = _jsonl_bytes(predictions)
    prediction_path.write_bytes(prediction_content)
    manifest_path = output_root / "geometry_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_file_sha256"] = _sha256(prediction_content)
    manifest_path.write_bytes(_json_bytes(manifest))

    report = _run_summary(module, benchmark_root, output_root)

    by_sequence = {row["sequence"]: row for row in report["sequences"]}
    assert by_sequence["s1"]["reviewed_positive_translation_error_m_median"] is None
    assert by_sequence["s1"]["reviewed_positive_translation_error_m_p90"] is None
    assert by_sequence["s2"]["reviewed_positive_translation_error_m_median"] is None
    assert report["overall"]["reviewed_positive_translation_error_m_median"] is None
    assert report["macro"]["reviewed_positive_translation_error_m_median"] is None
    assert "N/A" in (output_root / "label_comparison.md").read_text(encoding="utf-8")
    report_json = (output_root / "label_comparison.json").read_text(encoding="utf-8")
    assert "Infinity" not in report_json
    assert "NaN" not in report_json


def test_nonfinite_prediction_non_error_fields_are_rejected(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)
    prediction_path = output_root / "geometry_predictions.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    predictions[0]["support_idx"] = float("nan")
    prediction_content = _jsonl_bytes(predictions)
    prediction_path.write_bytes(prediction_content)
    manifest_path = output_root / "geometry_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_file_sha256"] = _sha256(prediction_content)
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match="support_idx.*non-finite"):
        module.summarize(benchmark_root, output_root)


@pytest.mark.parametrize("which", ["old", "new"])
def test_summary_rejects_invalid_annotation_seals(tmp_path, which):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)
    root = benchmark_root if which == "old" else output_root
    seal_path = root / "annotation_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["annotation_sha256"] = "0" * 64
    seal_path.write_bytes(_json_bytes(seal))

    with pytest.raises(ValueError, match="annotation hash"):
        module.summarize(benchmark_root, output_root)


@pytest.mark.parametrize("filename", ["benchmark_pairs.jsonl", "manifest.json"])
def test_summary_rejects_frozen_pair_bundle_byte_mismatch(tmp_path, filename):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)
    path = output_root / filename
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="frozen.*byte|exact"):
        module.summarize(benchmark_root, output_root)


@pytest.mark.parametrize("case", ["incomplete", "tampered_hash", "reordered", "bool_label"])
def test_prediction_manifest_and_rows_are_strictly_validated(tmp_path, case):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)
    manifest_path = output_root / "geometry_prediction_manifest.json"
    prediction_path = output_root / "geometry_predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if case == "incomplete":
        manifest["complete"] = False
        manifest_path.write_bytes(_json_bytes(manifest))
    elif case == "tampered_hash":
        manifest["prediction_file_sha256"] = "0" * 64
        manifest_path.write_bytes(_json_bytes(manifest))
    else:
        rows = [
            json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
        ]
        if case == "reordered":
            rows[0], rows[1] = rows[1], rows[0]
        else:
            rows[0]["automatic_label"] = True
        content = _jsonl_bytes(rows)
        prediction_path.write_bytes(content)
        manifest["prediction_file_sha256"] = _sha256(content)
        manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match="complete|hash|order|automatic_label"):
        module.summarize(benchmark_root, output_root)


def test_validate_only_requires_full_coverage_unless_partial_is_explicit(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path, pair_limit=3)

    with pytest.raises(ValueError, match="full pair.*coverage"):
        module.main(
            [
                str(benchmark_root),
                "--geometry-root",
                str(output_root),
                "--validate-predictions-only",
            ]
        )

    assert (
        module.main(
            [
                str(benchmark_root),
                "--geometry-root",
                str(output_root),
                "--validate-predictions-only",
                "--allow-partial-predictions",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="full pair.*coverage"):
        module.summarize(benchmark_root, output_root)


def test_allow_partial_predictions_requires_validate_only(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        module.main(
            [
                str(benchmark_root),
                "--geometry-root",
                str(output_root),
                "--allow-partial-predictions",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("mutation", "error", "rehash_da3_config"),
    [
        (
            lambda manifest: manifest.update(format_version=2),
            "format_version",
            False,
        ),
        (
            lambda manifest: manifest.update(prediction_file="other.jsonl"),
            "prediction_file",
            False,
        ),
        (
            lambda manifest: manifest.update(source_manifest_sha256="0" * 64),
            "source_manifest_sha256",
            False,
        ),
        (
            lambda manifest: manifest.update(source_pair_count=7),
            "source_pair_count",
            False,
        ),
        (
            lambda manifest: manifest.update(pair_limit=3),
            "pair_limit",
            False,
        ),
        (
            lambda manifest: manifest["thresholds"].update(max_translation_error_m=1),
            "thresholds",
            False,
        ),
        (
            lambda manifest: manifest["thresholds"].update(extra=1.0),
            "thresholds",
            False,
        ),
        (
            lambda manifest: manifest["request_contract"].update(device="cuda"),
            "request_contract.device|da3_config device",
            False,
        ),
        (
            lambda manifest: manifest["request_contract"].update(pair_limit=3),
            "request_contract.pair_limit",
            False,
        ),
        (
            lambda manifest: manifest["request_contract"]["thresholds"].update(
                max_rotation_error_deg=16.0
            ),
            "request_contract.thresholds",
            False,
        ),
        (
            lambda manifest: manifest.update(da3_config_sha256="0" * 64),
            "da3_config_sha256",
            False,
        ),
        (
            lambda manifest: manifest["da3_config"].update(device="cuda"),
            "da3_config.*device",
            True,
        ),
        (
            lambda manifest: manifest["da3_config"]["da3_runtime_by_dataset"]["d"].update(
                process_res=True
            ),
            "process_res",
            True,
        ),
    ],
)
def test_prediction_manifest_matches_task3_contract(
    tmp_path,
    mutation,
    error,
    rehash_da3_config,
):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path)
    manifest_path = output_root / "geometry_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    if rehash_da3_config:
        manifest["da3_config_sha256"] = _canonical_sha256(manifest["da3_config"])
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match=error):
        module.validate_prediction_bundle(
            benchmark_root,
            output_root,
            require_full_pair_coverage=True,
        )


def test_pair_limit_must_match_target_prefix_contract(tmp_path):
    module = _load_summary()
    benchmark_root, output_root, _ = _bundle(tmp_path, pair_limit=3)
    manifest_path = output_root / "geometry_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pair_limit"] = 4
    manifest["request_contract"]["pair_limit"] = 4
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match="pair_limit.*target_pair_count"):
        module.validate_prediction_bundle(
            benchmark_root,
            output_root,
            require_full_pair_coverage=False,
        )


@pytest.mark.parametrize("failed_publication", [2, 3])
def test_atomic_report_publication_rolls_back_all_old_files(
    tmp_path,
    monkeypatch,
    failed_publication,
):
    module = _load_summary()
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    old_payloads = {
        "label_comparison.csv": "old csv\n",
        "label_comparison.md": "old md\n",
        "label_comparison.json": "old json\n",
    }
    for filename, content in old_payloads.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    real_replace = module.os.replace
    publication_count = 0

    def fail_selected_publication(source, destination):
        nonlocal publication_count
        source_path = Path(source)
        if source_path.parent.name == "staged":
            publication_count += 1
            if publication_count == failed_publication:
                raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_selected_publication)
    with pytest.raises(OSError, match="injected"):
        module._write_payload_set_atomically(
            output_dir,
            {
                "label_comparison.csv": "new csv\n",
                "label_comparison.md": "new md\n",
                "label_comparison.json": "new json\n",
            },
        )

    assert {
        filename: (output_dir / filename).read_text(encoding="utf-8") for filename in old_payloads
    } == old_payloads


def test_atomic_report_first_publication_failure_preserves_existing_empty_root(
    tmp_path,
    monkeypatch,
):
    module = _load_summary()
    output_dir = tmp_path / "geometry"
    output_dir.mkdir()
    real_replace = module.os.replace

    def fail_first_publication(source, destination):
        if Path(source).parent.name == "staged":
            raise OSError("injected first publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_first_publication)
    with pytest.raises(OSError, match="injected"):
        module._write_payload_set_atomically(
            output_dir,
            {
                "label_comparison.csv": "new csv\n",
                "label_comparison.md": "new md\n",
                "label_comparison.json": "new json\n",
            },
        )

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_atomic_report_rollback_removes_new_files_and_restores_old_files(
    tmp_path,
    monkeypatch,
):
    module = _load_summary()
    output_dir = tmp_path / "geometry"
    output_dir.mkdir()
    old_csv = output_dir / "label_comparison.csv"
    old_csv.write_text("old csv\n", encoding="utf-8")
    real_replace = module.os.replace
    publication_count = 0

    def fail_third_publication(source, destination):
        nonlocal publication_count
        if Path(source).parent.name == "staged":
            publication_count += 1
            if publication_count == 3:
                raise OSError("injected third publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_third_publication)
    with pytest.raises(OSError, match="injected"):
        module._write_payload_set_atomically(
            output_dir,
            {
                "label_comparison.csv": "new csv\n",
                "label_comparison.md": "new md\n",
                "label_comparison.json": "new json\n",
            },
        )

    assert old_csv.read_text(encoding="utf-8") == "old csv\n"
    assert not (output_dir / "label_comparison.md").exists()
    assert not (output_dir / "label_comparison.json").exists()


def _fake_python(tmp_path: Path) -> tuple[Path, Path, Path]:
    log_path = tmp_path / "python.log"
    context_log_path = tmp_path / "python-context.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_LOG"\n'
        'printf "PWD=%s|PYTHONPATH=%s\\n" "$PWD" "${PYTHONPATH:-}" '
        '>> "$FAKE_PYTHON_CONTEXT_LOG"\n'
        'if [[ "${FAIL_VALIDATION:-0}" == "1" && "$*" == *"--validate-predictions-only"* ]]; then\n'
        '  echo "full pair coverage required" >&2\n'
        "  exit 9\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    return fake_python, log_path, context_log_path


def _run_wrapper(
    tmp_path: Path,
    *args: str,
    fail_validation: bool = False,
    cwd: Path = LOOPANYTHING_ROOT,
):
    fake_python, log_path, context_log_path = _fake_python(tmp_path)
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "FAKE_PYTHON_LOG": str(log_path),
        "FAKE_PYTHON_CONTEXT_LOG": str(context_log_path),
        "FAIL_VALIDATION": "1" if fail_validation else "0",
    }
    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT), *args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    contexts = (
        context_log_path.read_text(encoding="utf-8").splitlines()
        if context_log_path.exists()
        else []
    )
    return result, calls, contexts


def test_wrapper_pair_limit_builds_validates_partial_and_exits_without_ui(tmp_path):
    benchmark = tmp_path / "benchmark root"
    output = tmp_path / "output root"
    result, calls, _ = _run_wrapper(
        tmp_path,
        "--benchmark-root",
        str(benchmark),
        "--output-root",
        str(output),
        "--device",
        "cuda:1",
        "--backend",
        "mock",
        "--pair-limit",
        "20",
        "--port",
        "9001",
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert "build_da3_geometry_annotation.py" in calls[0]
    assert f"{benchmark} --output-root {output}" in calls[0]
    assert "--device cuda:1 --backend mock" in calls[0]
    assert "--pair-limit 20" in calls[0]
    assert "summarize_da3_geometry_annotations.py" in calls[1]
    assert "--validate-predictions-only" in calls[1]
    assert "--allow-partial-predictions" in calls[1]
    assert all("annotate_rover_aligned_benchmark.py" not in call for call in calls)
    assert "smoke bundle ready" in result.stdout.lower()


def test_wrapper_forwards_scale_adaptive_translation_thresholds(tmp_path):
    result, calls, _ = _run_wrapper(
        tmp_path,
        "--pair-limit",
        "1",
        "--min-translation-error-m",
        "0.8",
        "--max-translation-error-m",
        "6.0",
        "--translation-error-scale-ratio",
        "0.3",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "--min-translation-error-m 0.8 --max-translation-error-m 6.0 "
        "--translation-error-scale-ratio 0.3"
    ) in calls[0]


def test_wrapper_skip_build_still_validates_before_launch(tmp_path):
    result, calls, _ = _run_wrapper(tmp_path, "--skip-build")

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert "--validate-predictions-only" in calls[0]
    assert "--allow-partial-predictions" not in calls[0]
    assert "annotate_rover_aligned_benchmark.py" in calls[1]
    assert all("build_da3_geometry_annotation.py" not in call for call in calls)


def test_wrapper_summary_only_only_runs_full_summary(tmp_path):
    result, calls, _ = _run_wrapper(tmp_path, "--summary-only")

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert "summarize_da3_geometry_annotations.py" in calls[0]
    assert "--validate-predictions-only" not in calls[0]
    assert "build_da3_geometry_annotation.py" not in calls[0]
    assert "annotate_rover_aligned_benchmark.py" not in calls[0]


def test_wrapper_does_not_launch_when_complete_prediction_validation_fails(tmp_path):
    result, calls, _ = _run_wrapper(
        tmp_path,
        "--skip-build",
        fail_validation=True,
    )

    assert result.returncode == 9
    assert len(calls) == 1
    assert "--validate-predictions-only" in calls[0]
    assert "--allow-partial-predictions" not in calls[0]
    assert "full pair coverage required" in result.stderr


def test_wrapper_resolves_paths_and_pythonpath_from_repo_root_for_any_cwd(tmp_path):
    invocation_cwd = tmp_path / "elsewhere"
    invocation_cwd.mkdir()

    result, calls, contexts = _run_wrapper(
        tmp_path,
        "--skip-build",
        "--benchmark-root",
        "relative/benchmark",
        "--output-root",
        "relative/output",
        cwd=invocation_cwd,
    )

    assert result.returncode == 0, result.stderr
    expected_benchmark = LOOPANYTHING_ROOT / "relative/benchmark"
    expected_output = LOOPANYTHING_ROOT / "relative/output"
    assert f"{expected_benchmark} --geometry-root {expected_output}" in calls[0]
    assert f"{expected_output} --port 8765 --open" in calls[1]
    assert all(f"PWD={LOOPANYTHING_ROOT}|" in context for context in contexts)
    assert all(f"PYTHONPATH={LOOPANYTHING_ROOT / 'src'}" in context for context in contexts)


def test_wrapper_default_paths_are_repo_root_absolute_from_any_cwd(tmp_path):
    result, calls, _ = _run_wrapper(
        tmp_path,
        "--skip-build",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    benchmark = LOOPANYTHING_ROOT / "workspace/rover_aligned_benchmark/benchmark_v1"
    output = benchmark / "da3_geometry_annotation_v1"
    assert f"{benchmark} --geometry-root {output}" in calls[0]


@pytest.mark.parametrize(
    "args",
    [
        ("--unknown",),
        ("--benchmark-root",),
        ("--benchmark-root", "--summary-only"),
        ("--backend", "invalid"),
        ("--pair-limit", "0"),
        ("--port", "not-an-int"),
    ],
)
def test_wrapper_rejects_unknown_missing_and_invalid_arguments(tmp_path, args):
    result, calls, _ = _run_wrapper(tmp_path, *args)

    assert result.returncode != 0
    assert calls == []
