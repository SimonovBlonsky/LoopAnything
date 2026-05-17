#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from robust_loop_verifier.gt_heading import process_fusionportable_platform_gt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FusionPortable handheld/legged GT trajectories whose identity "
            "rotation columns are replaced by yaw inferred from GT translation."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/datasets/FusionPortable"),
        help="FusionPortable root containing handheld/ and legged/ GT folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/FusionPortable/processed_handheld_legged_gt"),
        help="Output root for processed GT trajectories.",
    )
    parser.add_argument(
        "--platform",
        dest="platforms",
        action="append",
        choices=("handheld", "legged"),
        help="Platform to process. Can be passed multiple times. Defaults to both.",
    )
    parser.add_argument(
        "--min-segment-translation-m",
        type=float,
        default=1e-3,
        help="Minimum XY translation needed for a segment heading to be valid.",
    )
    parser.add_argument(
        "--replace-all-rotations",
        action="store_true",
        help="Replace every quaternion instead of only identity quaternions.",
    )
    args = parser.parse_args()

    platforms = tuple(args.platforms) if args.platforms else ("handheld", "legged")
    summaries = process_fusionportable_platform_gt(
        args.input_root,
        args.output_root,
        platforms=platforms,
        min_segment_translation_m=args.min_segment_translation_m,
        replace_only_identity=not args.replace_all_rotations,
    )

    print(f"wrote {len(summaries)} processed GT trajectories to {args.output_root}")
    for summary in summaries:
        print(
            "{platform}/{sequence_name}: records={record_count}, replaced={replaced_identity_count}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
