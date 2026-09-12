#!/usr/bin/env python3
"""Run per-platform shared LoopAnything score sweeps on the reviewed main-8 benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from robust_loop_verifier.platform_shared_sweep import (
    MAIN8_PLATFORM_GROUPS,
    evaluate_platform_shared_sweep,
    generate_paper_safe_score_specs,
    read_records_with_review_labels,
    write_platform_shared_sweep_outputs,
)
from robust_loop_verifier.hard_gate_platform_sweep import (
    evaluate_hard_gate_global_shared_sweep,
    evaluate_hard_gate_platform_shared_sweep,
    read_hard_gate_records,
    write_hard_gate_selected_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_root", type=Path)
    parser.add_argument(
        "--candidate-records",
        type=Path,
        default=Path("workspace/rover_aligned_benchmark/benchmark_v1/candidate_records.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mr-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--hard-gate-candidate-scores",
        type=Path,
        default=None,
        help="Sweep maximum-violation Hard Gate thresholds using cached candidate metrics.",
    )
    parser.add_argument(
        "--hard-gate-global-shared",
        action="store_true",
        help="Use one Hard Gate threshold set across all eight sequences.",
    )
    args = parser.parse_args()

    if args.hard_gate_global_shared and args.hard_gate_candidate_scores is None:
        parser.error("--hard-gate-global-shared requires --hard-gate-candidate-scores")

    review_root = args.review_root
    output_dir = args.output_dir
    if output_dir is None:
        run_id = "platform_shared_score_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = review_root / "metrics" / run_id

    allowed_sequence_keys = [
        sequence_key
        for sequence_keys in MAIN8_PLATFORM_GROUPS.values()
        for sequence_key in sequence_keys
    ]
    if args.hard_gate_candidate_scores is not None:
        records = read_hard_gate_records(
            args.hard_gate_candidate_scores,
            review_root / "annotations.jsonl",
            allowed_sequence_keys=allowed_sequence_keys,
        )
        specs = []
        if args.hard_gate_global_shared:
            result = evaluate_hard_gate_global_shared_sweep(
                records,
                sequence_keys=allowed_sequence_keys,
                mr_tolerance=args.mr_tolerance,
            )
        else:
            result = evaluate_hard_gate_platform_shared_sweep(
                records,
                platform_groups=MAIN8_PLATFORM_GROUPS,
                mr_tolerance=args.mr_tolerance,
            )
    else:
        records = read_records_with_review_labels(
            args.candidate_records,
            review_root / "annotations.jsonl",
            allowed_sequence_keys=allowed_sequence_keys,
        )
        specs = generate_paper_safe_score_specs(records)
        result = evaluate_platform_shared_sweep(
            records,
            specs,
            platform_groups=MAIN8_PLATFORM_GROUPS,
            mr_tolerance=args.mr_tolerance,
        )
    write_platform_shared_sweep_outputs(output_dir, result)
    if args.hard_gate_candidate_scores is not None:
        write_hard_gate_selected_outputs(output_dir, result)

    manifest = {
        "review_root": str(review_root),
        "candidate_records": str(args.candidate_records),
        "hard_gate_candidate_scores": (
            str(args.hard_gate_candidate_scores)
            if args.hard_gate_candidate_scores is not None
            else None
        ),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "method_count": len(result["group_rows"]),
        "platform_groups": (
            {"all": allowed_sequence_keys}
            if args.hard_gate_global_shared
            else MAIN8_PLATFORM_GROUPS
        ),
        "sharing_scope": result.get("sharing_scope", "platform"),
        "mr_tolerance": args.mr_tolerance,
        "selection_rule": result.get(
            "selection_used_for_table",
            "balanced keeps methods within mr_tolerance macro MR@100P of group-best MR, "
            "then maximizes macro AP",
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    aggregate = result["selected_aggregate"]
    print(f"output_dir={output_dir}")
    print(
        "selected aggregate: AP={:.4f} MR@100P={:.4f} TP@100P={}/{}".format(
            aggregate["macro_AP"],
            aggregate["macro_MR@100P"],
            aggregate["TP@100P"],
            aggregate["positive_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
