import unittest
from collections import Counter

import numpy as np

import src.ml_model as ml_model
from src.ml_model import (
    FEATURE_NAMES,
    TrainingData,
    build_candidate_features,
    generate_training_data,
)
from src.workload import generate_workload


class LearnedFeatureTests(unittest.TestCase):
    def test_candidate_features_are_exactly_past_only_state(self):
        features = build_candidate_features(
            page=2,
            time_index=6,
            last_access={2: 3},
            access_counts={2: 4},
            recent_window=[1, 2, 3, 2, 4],
            loaded_at={2: 1},
        )
        self.assertEqual(features, (3, 4, 2, 5))

    def test_feature_names_exclude_page_identity_and_future_use(self):
        self.assertEqual(
            FEATURE_NAMES,
            ("recency", "total_frequency", "recent_frequency", "residency_time"),
        )
        normalized = " ".join(FEATURE_NAMES).lower()
        self.assertNotIn("page_id", normalized)
        self.assertNotIn("future", normalized)
        self.assertNotIn("next_use", normalized)


class TrainingLabelTests(unittest.TestCase):
    def test_each_full_memory_decision_has_one_positive_label(self):
        trace = [1, 2, 3, 1, 4, 2, 5, 1, 2, 3]
        data = generate_training_data([trace], frame_count=3)

        self.assertEqual(data.features.shape[1], 4)
        self.assertEqual(len(data.features), len(data.labels))
        self.assertEqual(len(data.labels), len(data.decision_ids))

        counts = Counter(data.decision_ids.tolist())
        for decision_id, candidate_count in counts.items():
            with self.subTest(decision_id=decision_id):
                mask = data.decision_ids == decision_id
                self.assertEqual(candidate_count, 3)
                self.assertEqual(int(data.labels[mask].sum()), 1)


class ClassifierTrainingTests(unittest.TestCase):
    def test_tree_uses_the_documented_small_configuration(self):
        trace = generate_workload(6, 3, 50, 11).trace
        data = generate_training_data([trace], frame_count=3)
        model = ml_model.train_eviction_model(data, random_state=11)

        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.min_samples_leaf, 5)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 11)
        self.assertEqual(model.classes_.tolist(), [0, 1])

    def test_training_rejects_empty_or_one_class_data(self):
        empty = TrainingData(
            features=np.empty((0, 4)),
            labels=np.array([], dtype=int),
            decision_ids=np.array([], dtype=int),
        )
        one_class = TrainingData(
            features=np.ones((5, 4)),
            labels=np.ones(5, dtype=int),
            decision_ids=np.arange(5),
        )
        with self.assertRaises(ValueError):
            ml_model.train_eviction_model(empty, random_state=42)
        with self.assertRaises(ValueError):
            ml_model.train_eviction_model(one_class, random_state=42)


class LearnedSimulationTests(unittest.TestCase):
    def test_equal_probabilities_use_recency_then_lowest_page(self):
        victim = ml_model.choose_learned_victim(
            candidates=[7, 2, 5],
            feature_rows=[(4, 1, 0, 6), (10, 1, 0, 11), (10, 2, 0, 12)],
            eviction_probabilities=[0.5, 0.5, 0.5],
        )
        self.assertEqual(victim, 2)

    def test_simulation_uses_classes_to_find_label_one_probability(self):
        class ReversedClassModel:
            classes_ = np.array([1, 0])

            def predict_proba(self, rows):
                recencies = np.asarray(rows, dtype=float)[:, 0]
                label_one = recencies / (recencies + 1.0)
                return np.column_stack((label_one, 1.0 - label_one))

        result = ml_model.simulate_learned(
            trace=[1, 2, 3, 1, 4],
            frame_count=3,
            model=ReversedClassModel(),
        )
        self.assertEqual(result.evictions, (2,))

    def test_real_tree_simulation_records_one_event_per_reference(self):
        training_trace = generate_workload(6, 3, 50, 9).trace
        data = generate_training_data([training_trace], frame_count=3)
        model = ml_model.train_eviction_model(data, random_state=9)
        evaluation_trace = [1, 2, 3, 1, 4, 2, 5, 1, 6, 2]

        result = ml_model.simulate_learned(evaluation_trace, 3, model)

        self.assertEqual(result.algorithm, "Learned")
        self.assertEqual(len(result.events), len(evaluation_trace))
        self.assertEqual(result.hits + result.faults, len(evaluation_trace))


if __name__ == "__main__":
    unittest.main()
