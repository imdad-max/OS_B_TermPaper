import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.experiment as experiment
from src.experiment import (
    ExperimentConfig,
    aggregate_metrics,
    evaluate_policies,
)
from src.ml_model import generate_training_data, train_eviction_model
from src.page_replacement import SimulationResult
from src.workload import generate_workload


class ExperimentMetricTests(unittest.TestCase):
    def test_matplotlib_cache_uses_a_writable_temporary_directory(self):
        cache_dir = Path(os.environ["MPLCONFIGDIR"])
        self.assertEqual(cache_dir.parent, Path(tempfile.gettempdir()))
        self.assertEqual(cache_dir.name, "learned-page-replacement-matplotlib")
        self.assertTrue(cache_dir.is_dir())

    def test_metrics_are_split_at_shift_and_sum_to_overall(self):
        result = SimulationResult(
            algorithm="Example",
            hits=3,
            faults=3,
            events=(False, True, True, False, True, False),
            evictions=(),
        )
        config = ExperimentConfig(
            page_count=10,
            frame_count=4,
            seed=42,
            references_per_phase=25,
            output_dir=Path("results"),
        )

        rows = aggregate_metrics(result, shift_index=3, config=config)

        self.assertEqual([row["period"] for row in rows], ["Before Shift", "After Shift", "Overall"])
        self.assertEqual(
            [(row["references"], row["hits"], row["page_faults"]) for row in rows],
            [(3, 2, 1), (3, 1, 2), (6, 3, 3)],
        )
        self.assertAlmostEqual(rows[0]["hit_ratio"], 2 / 3)
        self.assertAlmostEqual(rows[1]["hit_ratio"], 1 / 3)
        self.assertAlmostEqual(rows[2]["hit_ratio"], 0.5)
        self.assertEqual(rows[0]["pages"], 10)
        self.assertEqual(rows[0]["frames"], 4)
        self.assertEqual(rows[0]["seed"], 42)
        self.assertEqual(rows[0]["shift_index"], 3)
        self.assertEqual(rows[0]["trace_length"], 6)
        self.assertEqual(rows[0]["hits"] + rows[1]["hits"], rows[2]["hits"])
        self.assertEqual(
            rows[0]["page_faults"] + rows[1]["page_faults"],
            rows[2]["page_faults"],
        )

    def test_all_policies_evaluate_the_same_supplied_trace_length(self):
        training_trace = generate_workload(6, 3, 50, 12).trace
        data = generate_training_data([training_trace], frame_count=3)
        model = train_eviction_model(data, random_state=12)
        trace = [1, 2, 3, 1, 4, 2, 5, 1, 6, 2]

        results = evaluate_policies(trace, 3, model)

        self.assertEqual(set(results), {"FIFO", "LRU", "Optimal", "Learned"})
        for result in results.values():
            self.assertEqual(len(result.events), len(trace))

    def test_experiment_config_rejects_invalid_bounds(self):
        invalid = (
            {"page_count": 2, "frame_count": 2, "references_per_phase": 25},
            {"page_count": 3, "frame_count": 1, "references_per_phase": 25},
            {"page_count": 3, "frame_count": 3, "references_per_phase": 25},
            {"page_count": 10, "frame_count": 4, "references_per_phase": 24},
            {
                "page_count": 10,
                "frame_count": 4,
                "seed": -1,
                "references_per_phase": 25,
            },
            {
                "page_count": 10,
                "frame_count": 4,
                "seed": 2**32,
                "references_per_phase": 25,
            },
            {
                "page_count": 10,
                "frame_count": 4,
                "seed": 1.5,
                "references_per_phase": 25,
            },
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ExperimentConfig(output_dir=Path("results"), **values)

    def test_experiment_config_accepts_largest_supported_seed(self):
        config = ExperimentConfig(seed=2**32 - 1)
        self.assertEqual(config.seed, 2**32 - 1)


class ArtifactIntegrationTests(unittest.TestCase):
    def test_run_experiment_writes_replaceable_csv_and_three_figures(self):
        with tempfile.TemporaryDirectory(prefix="experiment output ") as temp_dir:
            output_dir = Path(temp_dir) / "results with spaces"
            config = ExperimentConfig(
                page_count=6,
                frame_count=3,
                seed=42,
                references_per_phase=50,
                output_dir=output_dir,
            )

            report = experiment.run_experiment(config)

            expected_columns = [
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
            self.assertEqual(list(report.results.columns), expected_columns)
            self.assertEqual(len(report.results), 12)
            self.assertEqual(report.evaluation_seed, 42)
            self.assertEqual(report.training_seeds, (43, 44, 45))
            self.assertTrue(report.csv_path.is_file())
            self.assertEqual(
                {path.name for path in report.figure_paths},
                {
                    "workload_shift.png",
                    "hit_ratio_comparison.png",
                    "page_faults_by_phase.png",
                },
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in report.figure_paths))
            self.assertTrue(np.isfinite(report.results["hit_ratio"]).all())
            self.assertTrue(report.results["hit_ratio"].between(0, 1).all())

            report.csv_path.write_text("stale", encoding="utf-8")
            repeated = experiment.run_experiment(config)
            self.assertEqual(len(pd.read_csv(repeated.csv_path)), 12)
            self.assertEqual(len(list((output_dir / "figures").glob("*.png"))), 3)

    def test_minimum_configuration_trains_and_evaluates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = experiment.run_experiment(
                ExperimentConfig(
                    page_count=3,
                    frame_count=2,
                    seed=7,
                    references_per_phase=25,
                    output_dir=Path(temp_dir) / "minimum",
                )
            )
            self.assertEqual(len(report.results), 12)
            self.assertTrue(np.isfinite(report.training_accuracy))
            self.assertTrue(report.results["hit_ratio"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
