from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
from robust_loop_verifier.pipeline import run_cached_sequence, run_mock_sequence_evaluation
from robust_loop_verifier.schema import RobustLoopVerifierConfig
from robust_loop_verifier.score_sweep import (
    compute_score_sweep,
    read_candidate_records,
    write_score_sweep_markdown,
)
from robust_loop_verifier.artifacts import write_json


app = typer.Typer(no_args_is_help=True)


def _raise_cli_error(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


def _parse_weight_csv(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"invalid weight list: {value!r}") from exc
    if not weights:
        raise ValueError("weight list must not be empty")
    return weights


@app.command("preprocess-fusionportable")
def preprocess_fusionportable(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    raw_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    gt_trajectory_file: Optional[Path] = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False
    ),
    sequence_name: str = typer.Option(...),
    max_gt_delta_sec: float = typer.Option(0.05),
) -> None:
    try:
        cfg = RobustLoopVerifierConfig.from_yaml(config)
        out = preprocess_fusionportable_sequence(
            raw_dir,
            gt_trajectory_file,
            sequence_name,
            cfg,
            max_gt_delta_sec,
        )
    except (OSError, ValueError) as error:
        _raise_cli_error(error)
    typer.echo(str(out))


@app.command("run-mock")
def run_mock(output_root: Path = typer.Option(..., file_okay=False, dir_okay=True)) -> None:
    try:
        result = run_mock_sequence_evaluation(output_root)
    except (OSError, ValueError) as error:
        _raise_cli_error(error)
    typer.echo(f"candidate_count={result['candidate_count']}")


@app.command("run-cache")
def run_cache(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    sequence_cache: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    query_limit: int = typer.Option(20),
    backend: str = typer.Option("real"),
) -> None:
    try:
        cfg = RobustLoopVerifierConfig.from_yaml(config)
        result = run_cached_sequence(
            cfg,
            sequence_cache=sequence_cache,
            output_root=output_root,
            query_limit=query_limit,
            backend=backend,
        )
    except (OSError, ValueError) as error:
        _raise_cli_error(error)
    typer.echo(f"candidate_count={result['candidate_count']}")


@app.command("sweep-scores")
def sweep_scores(
    candidate_records: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    graph_weights: str = typer.Option("0,0.25,0.5,1,2,4"),
    fusion_weights: str = typer.Option("0,0.25,0.5,1,2,4"),
) -> None:
    try:
        result = compute_score_sweep(
            read_candidate_records(candidate_records),
            graph_weights=_parse_weight_csv(graph_weights),
            fusion_weights=_parse_weight_csv(fusion_weights),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "score_sweep.json", result)
        write_score_sweep_markdown(output_root / "score_sweep.md", result)
    except (OSError, ValueError) as error:
        _raise_cli_error(error)
    best_ap = result["best_by_ap"]
    best_mr = result["best_by_mr"]
    typer.echo(
        "best_ap={} AP={:.6f} MR@100P={:.6f}".format(
            best_ap["name"],
            best_ap["AP"],
            best_ap["MR@100P"],
        )
    )
    typer.echo(
        "best_mr={} AP={:.6f} MR@100P={:.6f}".format(
            best_mr["name"],
            best_mr["AP"],
            best_mr["MR@100P"],
        )
    )


if __name__ == "__main__":
    app()
