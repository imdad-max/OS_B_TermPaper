import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import main as app


class TerminalConfigurationTests(unittest.TestCase):
    def test_blank_prompts_choose_documented_defaults(self):
        answers = iter(["", "", ""])
        config = app.resolve_config(
            app.parse_args([]),
            input_fn=lambda _prompt: next(answers),
            output_fn=lambda _message: None,
        )
        self.assertEqual(config.page_count, 10)
        self.assertEqual(config.frame_count, 4)
        self.assertEqual(config.seed, 42)

    def test_non_integer_and_out_of_range_answers_retry(self):
        answers = iter(["abc", "2", "8", "8", "3", ""])
        messages = []
        config = app.resolve_config(
            app.parse_args([]),
            input_fn=lambda _prompt: next(answers),
            output_fn=messages.append,
        )
        self.assertEqual((config.page_count, config.frame_count, config.seed), (8, 3, 42))
        self.assertEqual(len(messages), 3)
        self.assertTrue(all("integer" in message.lower() or "must" in message.lower() for message in messages))

    def test_minimum_meaningful_values_are_accepted(self):
        args = app.parse_args(
            [
                "--pages",
                "3",
                "--frames",
                "2",
                "--seed",
                "0",
                "--references-per-phase",
                "25",
            ]
        )
        config = app.resolve_config(args, input_fn=lambda _prompt: self.fail("prompted"))
        self.assertEqual((config.page_count, config.frame_count, config.seed), (3, 2, 0))

    def test_complete_cli_configuration_does_not_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = app.parse_args(
                [
                    "--pages",
                    "8",
                    "--frames",
                    "3",
                    "--seed",
                    "7",
                    "--references-per-phase",
                    "50",
                    "--output-dir",
                    temp_dir,
                ]
            )
            config = app.resolve_config(
                args,
                input_fn=lambda _prompt: self.fail("complete CLI should not prompt"),
            )
            self.assertEqual(config.page_count, 8)
            self.assertEqual(config.frame_count, 3)
            self.assertEqual(config.seed, 7)
            self.assertEqual(config.references_per_phase, 50)
            self.assertEqual(config.output_dir, Path(temp_dir))

    def test_invalid_cli_combination_is_rejected_before_execution(self):
        args = app.parse_args(["--pages", "4", "--frames", "4", "--seed", "42"])
        with self.assertRaises(ValueError):
            app.resolve_config(args, input_fn=lambda _prompt: self.fail("prompted"))

    def test_invalid_partial_cli_page_count_is_rejected_before_prompting(self):
        args = app.parse_args(["--pages", "2", "--seed", "42"])
        with self.assertRaisesRegex(ValueError, "page_count"):
            app.resolve_config(args, input_fn=lambda _prompt: self.fail("prompted"))

    def test_invalid_partial_cli_seed_is_rejected_before_prompting(self):
        args = app.parse_args(["--seed", "-1"])
        with self.assertRaisesRegex(ValueError, "seed"):
            app.resolve_config(args, input_fn=lambda _prompt: self.fail("prompted"))

    def test_interactive_seed_retries_outside_supported_range(self):
        answers = iter(["", "", "-1", "4294967296", "42"])
        messages = []
        config = app.resolve_config(
            app.parse_args([]),
            input_fn=lambda _prompt: next(answers),
            output_fn=messages.append,
        )
        self.assertEqual(config.seed, 42)
        self.assertEqual(
            messages,
            [
                "Seed must be between 0 and 4294967295.",
                "Seed must be between 0 and 4294967295.",
            ],
        )


class MainEntryPointTests(unittest.TestCase):
    def test_short_non_interactive_run_prints_summary_and_writes_outputs(self):
        with tempfile.TemporaryDirectory(prefix="main output ") as temp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = app.main(
                    [
                        "--pages",
                        "8",
                        "--frames",
                        "3",
                        "--seed",
                        "7",
                        "--references-per-phase",
                        "50",
                        "--output-dir",
                        temp_dir,
                    ]
                )

            text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("Training accuracy (diagnostic only)", text)
            self.assertIn("Workload shift index", text)
            self.assertIn("raw_results.csv", text)
            self.assertTrue((Path(temp_dir) / "raw_results.csv").is_file())
            self.assertEqual(len(list((Path(temp_dir) / "figures").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
