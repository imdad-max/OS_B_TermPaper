"""Experiment orchestration, metric aggregation, and artifact generation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Sequence

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "learned-page-replacement-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CONFIG_DIR)

import matplotlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.ml_model import (
    generate_training_data,
    simulate_learned,
    train_eviction_model,
)
from src.page_replacement import (
    SimulationResult,
    simulate_fifo,
    simulate_lru,
    simulate_optimal,
)
from src.workload import Workload, generate_workload


MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class ExperimentConfig:
    """User-adjustable and reproducible experiment settings."""

    page_count: int = 10
    frame_count: int = 4
    seed: int = 42
    references_per_phase: int = 1000
    output_dir: Path = Path("results")

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seed, Integral)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= MAX_SEED
        ):
            raise ValueError(f"seed must be an integer between 0 and {MAX_SEED}")
        if self.page_count < 3:
            raise ValueError("page_count must be at least 3")
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        if self.frame_count >= self.page_count:
            raise ValueError("frame_count must be smaller than page_count")
        if self.references_per_phase < 25:
            raise ValueError("references_per_phase must be at least 25")


@dataclass(frozen=True)
class ExperimentReport:
    """Results and artifact locations from a completed experiment."""

    results: pd.DataFrame
    training_accuracy: float
    evaluation_seed: int
    training_seeds: tuple[int, int, int]
    shift_index: int
    trace_length: int
    csv_path: Path
    figure_paths: tuple[Path, ...]


def aggregate_metrics(
    result: SimulationResult,
    shift_index: int,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    """Calculate before-shift, after-shift, and overall result rows."""

    if not 0 < shift_index < len(result.events):
        raise ValueError("shift_index must divide the event sequence")

    periods = (
        ("Before Shift", result.events[:shift_index]),
        ("After Shift", result.events[shift_index:]),
        ("Overall", result.events),
    )
    rows: list[dict[str, object]] = []
    for period, events in periods:
        references = len(events)
        hits = sum(events)
        page_faults = references - hits
        rows.append(
            {
                "algorithm": result.algorithm,
                "period": period,
                "references": references,
                "hits": hits,
                "page_faults": page_faults,
                "hit_ratio": hits / references,
                "pages": config.page_count,
                "frames": config.frame_count,
                "seed": config.seed,
                "shift_index": shift_index,
                "trace_length": len(result.events),
            }
        )
    return rows


def evaluate_policies(
    trace: Sequence[int],
    frame_count: int,
    learned_model: DecisionTreeClassifier,
) -> dict[str, SimulationResult]:
    """Run every policy on the same supplied reference sequence."""

    return {
        "FIFO": simulate_fifo(trace, frame_count),
        "LRU": simulate_lru(trace, frame_count),
        "Optimal": simulate_optimal(trace, frame_count),
        "Learned": simulate_learned(trace, frame_count, learned_model),
    }


def _plot_workload(workload: Workload, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))
    before_indices = np.arange(workload.shift_index)
    after_indices = np.arange(workload.shift_index, len(workload.trace))
    axis.scatter(
        before_indices,
        workload.trace[: workload.shift_index],
        s=9,
        alpha=0.65,
        label="Phase 1: locality-heavy / sequential",
    )
    axis.scatter(
        after_indices,
        workload.trace[workload.shift_index :],
        s=9,
        alpha=0.65,
        label="Phase 2: random / bursty",
    )
    axis.axvline(
        workload.shift_index,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Workload shift",
    )
    axis.set_title("Synthetic Page-Reference Workload Shift")
    axis.set_xlabel("Reference index")
    axis.set_ylabel("Page identifier")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_grouped_metric(
    results: pd.DataFrame,
    periods: Sequence[str],
    metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    algorithms = ("FIFO", "LRU", "Optimal", "Learned")
    positions = np.arange(len(periods))
    width = 0.19
    figure, axis = plt.subplots(figsize=(9, 5.2))

    for algorithm_index, algorithm in enumerate(algorithms):
        values = [
            float(
                results.loc[
                    (results["algorithm"] == algorithm)
                    & (results["period"] == period),
                    metric,
                ].iloc[0]
            )
            for period in periods
        ]
        offset = (algorithm_index - 1.5) * width
        axis.bar(positions + offset, values, width=width, label=algorithm)

    axis.set_title(title)
    axis.set_xlabel("Workload period")
    axis.set_ylabel(y_label)
    axis.set_xticks(positions, periods)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    if metric == "hit_ratio":
        axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_experiment(config: ExperimentConfig) -> ExperimentReport:
    """Train on separate traces, evaluate once, and write real artifacts."""

    evaluation_workload = generate_workload(
        config.page_count,
        config.frame_count,
        config.references_per_phase,
        config.seed,
    )
    training_seeds = (config.seed + 1, config.seed + 2, config.seed + 3)
    training_traces = [
        generate_workload(
            config.page_count,
            config.frame_count,
            config.references_per_phase,
            training_seed,
        ).trace
        for training_seed in training_seeds
    ]
    training_data = generate_training_data(training_traces, config.frame_count)
    learned_model = train_eviction_model(training_data, config.seed)
    training_accuracy = float(
        learned_model.score(training_data.features, training_data.labels)
    )

    policy_results = evaluate_policies(
        evaluation_workload.trace,
        config.frame_count,
        learned_model,
    )
    rows = [
        row
        for result in policy_results.values()
        for row in aggregate_metrics(result, evaluation_workload.shift_index, config)
    ]
    columns = [
        "algorithm",
        "period",
        "references",
        "hits",
        "page_faults",
        "hit_ratio",
        "pages",
        "frames",
        "seed",
        "shift_index",
        "trace_length",
    ]
    results = pd.DataFrame(rows, columns=columns)

    output_dir = config.output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_results.csv"
    results.to_csv(csv_path, index=False)

    figure_paths = (
        figures_dir / "workload_shift.png",
        figures_dir / "hit_ratio_comparison.png",
        figures_dir / "page_faults_by_phase.png",
    )
    _plot_workload(evaluation_workload, figure_paths[0])
    _plot_grouped_metric(
        results,
        ("Before Shift", "After Shift", "Overall"),
        "hit_ratio",
        "Hit Ratio by Policy and Workload Period",
        "Hit ratio",
        figure_paths[1],
    )
    _plot_grouped_metric(
        results,
        ("Before Shift", "After Shift"),
        "page_faults",
        "Page Faults Before and After the Workload Shift",
        "Page-fault count",
        figure_paths[2],
    )

    return ExperimentReport(
        results=results,
        training_accuracy=training_accuracy,
        evaluation_seed=config.seed,
        training_seeds=training_seeds,
        shift_index=evaluation_workload.shift_index,
        trace_length=len(evaluation_workload.trace),
        csv_path=csv_path,
        figure_paths=figure_paths,
    )
