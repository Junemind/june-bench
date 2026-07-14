"""SB6 · wiring sweep — every registered system/dataset is reachable, profiles cover the datasets,
and the CLI surface is intact. Nothing the suite advertises is dead or unreachable."""
from __future__ import annotations

import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import datasets, profiles, systems   # noqa: E402
from june_bench.cli import build_parser               # noqa: E402
from june_bench.ports import Dataset, System          # noqa: E402

_REAL = ("hotpot", "2wiki", "musique", "locomo", "longmemeval", "financebench")


class TestWiring(unittest.TestCase):
    def test_every_dataset_loads_smoke_and_satisfies_the_port(self):
        for name in (*_REAL, "smoke"):
            ds = datasets.registry.get(name)
            self.assertIsInstance(ds, Dataset)
            self.assertTrue(list(ds.load("smoke")), f"{name} smoke empty")

    def test_every_dataset_has_a_headline_profile(self):
        for name in _REAL:
            self.assertTrue(profiles.headline_metrics(name))

    def test_no_deps_systems_resolve_and_satisfy_the_port(self):
        for name in ("echo", "null"):
            self.assertIsInstance(systems.get(name), System)

    def test_heavy_systems_are_registered_and_error_clearly_not_silently(self):
        # june-local / cognee / june-api (no URL) must raise a clear error, never return a broken object.
        os.environ.pop("JUNE_BENCH_JUNE_URL", None)
        for name in ("june-api", "june-local", "cognee"):
            self.assertIn(name, systems.names())
            try:
                import cognee  # noqa: F401
                cognee_present = True
            except Exception:
                cognee_present = False
            if name == "cognee" and cognee_present:
                continue
            with self.assertRaises((RuntimeError, NotImplementedError, ValueError)):
                systems.get(name)

    def test_cli_exposes_run_list_suite(self):
        ap = build_parser()
        # argparse subparsers are registered; parsing each command's minimal form works.
        self.assertEqual(ap.parse_args(["list"]).cmd, "list")
        self.assertEqual(ap.parse_args(["run", "--system", "echo"]).cmd, "run")
        self.assertEqual(ap.parse_args(["suite", "--systems", "echo"]).cmd, "suite")


if __name__ == "__main__":
    unittest.main()
