"""SB0 · no-deps end-to-end smoke — load → run → score → report, with no model/network/datasets.

Proves the whole harness is wired: the oracle EchoSystem scores EM 1.0 over the bundled fixture, the
NullSystem honestly scores 0 with full abstention, the ports are structurally satisfied, and the CLI
runs offline. Written as a `unittest.TestCase` so it runs under plain `unittest` (in-sandbox, no
pytest) AND under pytest on the dev machine.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import Dataset, System, run, score          # noqa: E402
from june_bench import datasets, systems                     # noqa: E402
from june_bench.cli import main as cli_main                  # noqa: E402
from june_bench.systems.base import EchoSystem, NullSystem   # noqa: E402


def _smoke():
    return datasets.registry.get("smoke")


class TestSB0Smoke(unittest.TestCase):
    def test_ports_are_satisfied(self):
        self.assertIsInstance(EchoSystem(), System)
        self.assertIsInstance(_smoke(), Dataset)

    def test_registry_lists_smoke_and_oracles(self):
        self.assertIn("smoke", datasets.registry.names())
        self.assertTrue({"echo", "null"} <= set(systems.names()))

    def test_echo_oracle_scores_em_1(self):
        s = score(run(EchoSystem(), _smoke(), split="smoke"))
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["answered"], 4)
        self.assertEqual(s["em"], 1.0)
        self.assertEqual(s["coverage"], 1.0)
        self.assertEqual(s["calls_per_answer"], 1.0)

    def test_null_system_scores_zero_and_abstains(self):
        s = score(run(NullSystem(), _smoke(), split="smoke"))
        self.assertEqual(s["answered"], 0)
        self.assertEqual(s["abstained"], 4)
        self.assertEqual(s["coverage"], 0.0)
        self.assertEqual(s["em"], 0.0)

    def test_context_recall_is_measured(self):
        s = score(run(NullSystem(), _smoke(), split="smoke"))
        self.assertEqual(s["context_recall"], 1.0)   # gold present in each fixture's context

    def test_limit_truncates(self):
        self.assertEqual(len(run(EchoSystem(), _smoke(), split="smoke", limit=2)), 2)

    def test_cli_run_offline_returns_zero(self):
        self.assertEqual(cli_main(["run", "--system", "echo", "--dataset", "smoke", "--json"]), 0)
        self.assertEqual(cli_main(["list"]), 0)

    def test_run_is_failsoft_per_example(self):
        # A system that raises on every answer must NOT crash the run — each example becomes an errored
        # miss (abstained, meta['system_error']) and the run completes (so a flaky competitor can't void
        # a head-to-head). Scored as 0 EM / full abstention, exactly like NullSystem but flagged errored.
        import asyncio

        from june_bench.ports import Example
        from june_bench.runner import run_async

        class Boom:
            name = "boom"

            async def answer(self, ex):
                raise RuntimeError("provider 422")

        exs = [Example(qid=str(i), question="q", golds=("x",), context=("c",)) for i in range(3)]
        recs = asyncio.run(run_async(Boom(), exs))
        self.assertEqual(len(recs), 3)                                  # nothing lost
        self.assertTrue(all(r.abstained for r in recs))
        self.assertTrue(all(r.meta.get("system_error") for r in recs))
        self.assertEqual(score(recs)["em"], 0.0)

    def test_cascade_of_errors_aborts(self):
        # When the endpoint is DOWN (every call fails), the run must ABORT after N consecutive errors
        # — not score a cascade of phantom misses and report a misleading partial result.
        import asyncio
        import os

        from june_bench.ports import Example
        from june_bench.runner import run_async

        class Down:
            name = "down"

            async def answer(self, ex):
                raise ConnectionError("All connection attempts failed")

        exs = [Example(qid=str(i), question="q", golds=("x",)) for i in range(10)]
        saved = os.environ.get("JUNE_BENCH_MAX_CONSEC_ERRORS")
        os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = "5"
        try:
            with self.assertRaises(RuntimeError) as cm:
                asyncio.run(run_async(Down(), exs))
            self.assertIn("consecutive errors", str(cm.exception))
        finally:
            if saved is None:
                os.environ.pop("JUNE_BENCH_MAX_CONSEC_ERRORS", None)
            else:
                os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = saved

    def test_scattered_errors_do_not_abort(self):
        # isolated errors (interleaved with successes) stay fail-soft — only a CONSECUTIVE run aborts.
        import asyncio
        import os

        from june_bench.ports import Example
        from june_bench.runner import run_async

        class Flaky:
            name = "flaky"

            def __init__(self):
                self.n = 0

            async def answer(self, ex):
                self.n += 1
                if self.n % 2 == 0:
                    raise RuntimeError("blip")
                from june_bench.ports import Prediction
                return Prediction(text="x", meta={"calls": 1})

        exs = [Example(qid=str(i), question="q", golds=("x",)) for i in range(12)]
        os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = "5"
        try:
            recs = asyncio.run(run_async(Flaky(), exs))     # alternating fail/ok → never 5-in-a-row
            self.assertEqual(len(recs), 12)
        finally:
            os.environ.pop("JUNE_BENCH_MAX_CONSEC_ERRORS", None)

    def test_fail_fast_reraises(self):
        # Opt-out: fail_fast=True (or JUNE_BENCH_FAIL_FAST=1) re-raises for debugging an adapter.
        import asyncio

        from june_bench.ports import Example
        from june_bench.runner import run_async

        class Boom:
            name = "boom"

            async def answer(self, ex):
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(run_async(Boom(), [Example(qid="1", question="q", golds=("x",))],
                                  fail_fast=True))


if __name__ == "__main__":
    unittest.main()
