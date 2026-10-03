"""Terminal entry point for the learned page-replacement experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

from src.experiment import MAX_SEED, ExperimentConfig, run_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional non-interactive experiment settings."""

    parser = argparse.ArgumentParser(
        description="Compare classical and learned page-replacement policies."
    )
    parser.add_argument("--pages", type=int, default=None, help="number of unique pages")
    parser.add_argument("--frames", type=int, default=None, help="number of page frames")
    parser.add_argument("--seed", type=int, default=None, help="evaluation random seed")
    parser.add_argument(
        "--references-per-phase",
        type=int,
        default=1000,
        help="references before and after the midpoint shift (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for the CSV and figures (default: results)",
    )
    return parser.parse_args(argv)


def prompt_for_int(
    label: str,
    default: int,
    predicate: Callable[[int], bool],
    error_message: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> int:
    """Prompt until the user supplies an integer accepted by `predicate`."""

    while True:
        raw_value = input_fn(f"{label} [{default}]: ").strip()
        if not raw_value:
            value = default
        else:
            try:
                value = int(raw_value)
            except ValueError:
                output_fn("Please enter a whole integer value.")
                continue

        if predicate(value):
            return value
        output_fn(error_message)


def resolve_config(
    args: argparse.Namespace,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> ExperimentConfig:
    """Combine command-line values with prompts for omitted core settings."""

    if args.pages is not None and args.pages < 3:
        raise ValueError("page_count must be at least 3")
    if args.frames is not None and args.frames < 2:
        raise ValueError("frame_count must be at least 2")
    if args.seed is not None and not 0 <= args.seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")

    page_count = args.pages
    if page_count is None:
        page_count = prompt_for_int(
            "Number of unique pages",
            10,
            lambda value: value >= 3,
            "Page count must be at least 3.",
            input_fn,
            output_fn,
        )

    frame_count = args.frames
    if frame_count is None:
        frame_count = prompt_for_int(
            "Number of frames",
            4,
            lambda value: 2 <= value < page_count,
            f"Frame count must be at least 2 and smaller than {page_count} pages.",
            input_fn,
            output_fn,
        )
    elif frame_count >= page_count:
        raise ValueError("frame_count must be smaller than page_count")

    seed = args.seed
    if seed is None:
        seed = prompt_for_int(
            "Random seed",
            42,
            lambda value: 0 <= value <= MAX_SEED,
            f"Seed must be between 0 and {MAX_SEED}.",
            input_fn,
            output_fn,
        )

    return ExperimentConfig(
        page_count=page_count,
        frame_count=frame_count,
        seed=seed,
        references_per_phase=args.references_per_phase,
        output_dir=args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured experiment and print a concise terminal report."""

    args = parse_args(argv)
    print("Learned Page Replacement Experiment")
    print("-----------------------------------")
    try:
        config = resolve_config(args)
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 2

    print(
        "Configuration: "
        f"pages={config.page_count}, frames={config.frame_count}, "
        f"seed={config.seed}, references/phase={config.references_per_phase}"
    )
    report = run_experiment(config)

    print(f"Workload shift index: {report.shift_index}")
    print(f"Training seeds: {report.training_seeds}")
    print(f"Training accuracy (diagnostic only): {report.training_accuracy:.4f}")
    print()
    print(report.results.to_string(index=False))
    print()
    print(f"CSV saved to: {report.csv_path}")
    for figure_path in report.figure_paths:
        print(f"Figure saved to: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
