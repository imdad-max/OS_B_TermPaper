import unittest

from src.page_replacement import (
    choose_optimal_victim,
    simulate_fifo,
    simulate_lru,
    simulate_optimal,
)


CLASSIC_TRACE = [8, 1, 2, 3, 1, 4, 1, 5, 3, 4, 1, 4, 3]


class ClassicalPageReplacementTests(unittest.TestCase):
    def test_fifo_matches_hand_calculated_counts(self):
        result = simulate_fifo(CLASSIC_TRACE, 3)
        self.assertEqual((result.hits, result.faults), (3, 10))

    def test_lru_matches_hand_calculated_counts(self):
        result = simulate_lru(CLASSIC_TRACE, 3)
        self.assertEqual((result.hits, result.faults), (4, 9))

    def test_optimal_matches_hand_calculated_counts(self):
        result = simulate_optimal(CLASSIC_TRACE, 3)
        self.assertEqual((result.hits, result.faults), (6, 7))

    def test_every_reference_has_one_event(self):
        for simulator in (simulate_fifo, simulate_lru, simulate_optimal):
            with self.subTest(simulator=simulator.__name__):
                result = simulator(CLASSIC_TRACE, 3)
                self.assertEqual(len(result.events), len(CLASSIC_TRACE))
                self.assertEqual(result.hits + result.faults, len(CLASSIC_TRACE))

    def test_victim_orders_show_each_policy_rule(self):
        trace = [1, 2, 3, 1, 4, 5]
        self.assertEqual(simulate_fifo(trace, 3).evictions, (1, 2))
        self.assertEqual(simulate_lru(trace, 3).evictions, (2, 3))
        self.assertEqual(simulate_optimal(trace, 3).evictions, (1, 2))

    def test_optimal_tie_uses_lowest_page_identifier(self):
        self.assertEqual(choose_optimal_victim([3, 1, 2], []), 1)

    def test_invalid_inputs_are_rejected(self):
        for simulator in (simulate_fifo, simulate_lru, simulate_optimal):
            with self.subTest(simulator=simulator.__name__, case="empty"):
                with self.assertRaises(ValueError):
                    simulator([], 3)
            with self.subTest(simulator=simulator.__name__, case="bad page"):
                with self.assertRaises(ValueError):
                    simulator([1, 0, 2], 3)
            with self.subTest(simulator=simulator.__name__, case="bad frames"):
                with self.assertRaises(ValueError):
                    simulator([1, 2, 3], 0)


if __name__ == "__main__":
    unittest.main()
