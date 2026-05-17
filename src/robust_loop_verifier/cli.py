from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
from robust_loop_verifier.pipeline import run_cached_sequence, run_mock_sequence_evaluation
from robust_loop_verifier.schema import RobustLoopVerifierConfig


app = typer.Typer(no_args_is_help=True)


def _raise_cli_error(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()
