import unittest

from src.workload import generate_workload


def sequential_cycle_rate(segment, working_set_size):
    matching = sum(
        previous <= working_set_size
        and current == (previous % working_set_size) + 1
        for previous, current in zip(segment, segment[1:])
    )
    return matching / (len(segment) - 1)


class WorkloadTests(unittest.TestCase):
    def test_workload_has_exact_length_shift_and_page_range(self):
        workload = generate_workload(10, 4, references_per_phase=1000, seed=42)
        self.assertEqual(len(workload.trace), 2000)
        self.assertEqual(workload.shift_index, 1000)
        self.assertTrue(all(1 <= page <= 10 for page in workload.trace))

    def test_same_seed_repeats_and_different_seed_changes_trace(self):
        first = generate_workload(10, 4, 1000, 42)
        repeated = generate_workload(10, 4, 1000, 42)
        changed = generate_workload(10, 4, 1000, 43)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.trace, changed.trace)

    def test_shift_reduces_sequential_cycle_structure(self):
        workload = generate_workload(10, 4, 1000, 42)
        before = workload.trace[: workload.shift_index]
        after = workload.trace[workload.shift_index :]
        before_rate = sequential_cycle_rate(before, working_set_size=4)
        after_rate = sequential_cycle_rate(after, working_set_size=4)
        self.assertGreaterEqual(before_rate - after_rate, 0.20)

    def test_minimum_meaningful_configuration_is_supported(self):
        workload = generate_workload(3, 2, 25, 7)
        self.assertEqual(len(workload.trace), 50)
        self.assertEqual(workload.shift_index, 25)
        self.assertTrue(all(1 <= page <= 3 for page in workload.trace))

    def test_invalid_configurations_are_rejected(self):
        invalid_arguments = (
            (2, 2, 25),
            (3, 1, 25),
            (3, 3, 25),
            (10, 4, 24),
        )
        for pages, frames, phase_length in invalid_arguments:
            with self.subTest(
                pages=pages,
                frames=frames,
                references_per_phase=phase_length,
            ):
                with self.assertRaises(ValueError):
                    generate_workload(pages, frames, phase_length, 42)


if __name__ == "__main__":
    unittest.main()
