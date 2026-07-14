"""SB5 · the suite matrix + RESULTS.md reporter."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.cli import main as cli_main          # noqa: E402
from june_bench.report import suite_markdown          # noqa: E402


class TestSuiteReport(unittest.TestCase):
    def test_suite_markdown_has_a_row_per_cell_and_headline(self):
        rows = [
            {"system": "echo", "dataset": "hotpot", "summary": {
                "n": 2, "em": 1.0, "f1": 1.0, "coverage": 1.0, "context_recall": 1.0,
                "calls_per_answer": 1.0, "cost_per_answer": 0.0}},
            {"system": "june-api", "dataset": "locomo", "error": "set JUNE_BENCH_JUNE_URL"},
        ]
        md = suite_markdown(rows, split="smoke", model="gpt-4o-mini")
        self.assertIn("| echo | hotpot | em/f1 |", md)     # headline profile shown
        self.assertIn("_error_", md)                        # fail-soft cell rendered
        self.assertIn("split `smoke`", md)


class TestSuiteCLI(unittest.TestCase):
    def test_suite_runs_matrix_offline(self):
        self.assertEqual(cli_main(["suite", "--systems", "echo,null",
                                   "--datasets", "smoke", "--split", "smoke"]), 0)

    def test_suite_writes_results_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "RESULTS.md"
            rc = cli_main(["suite", "--systems", "echo", "--datasets", "hotpot,locomo",
                           "--split", "smoke", "--out", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("| echo | hotpot |", text)
            self.assertIn("| echo | locomo |", text)

    def test_suite_failsoft_on_unconfigured_system(self):
        # june-api with no URL must record an error cell, not crash the matrix.
        import os
        old = os.environ.pop("JUNE_BENCH_JUNE_URL", None)
        try:
            self.assertEqual(cli_main(["suite", "--systems", "june-api,echo",
                                       "--datasets", "smoke"]), 0)
        finally:
            if old is not None:
                os.environ["JUNE_BENCH_JUNE_URL"] = old


if __name__ == "__main__":
    unittest.main()
