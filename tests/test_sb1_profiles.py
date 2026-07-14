"""SB1 · per-dataset scoring profiles."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import profiles                    # noqa: E402
from june_bench.score import score                 # noqa: E402
from june_bench.ports import Record                # noqa: E402


class TestProfiles(unittest.TestCase):
    def test_hotpot_is_em_f1(self):
        self.assertEqual(profiles.headline_metrics("hotpot"), ("em", "f1"))

    def test_unknown_dataset_falls_back_to_default(self):
        self.assertEqual(profiles.headline_metrics("does-not-exist"),
                         profiles.headline_metrics("default"))

    def test_longmemeval_includes_context_recall(self):
        self.assertIn("context_recall", profiles.headline_metrics("longmemeval"))

    def test_headline_projects_summary(self):
        summary = score([Record(qid="1", question="q", golds=("Paris",),
                                prediction="Paris", context=("Paris",), calls=1)])
        head = profiles.headline(summary, "hotpot")
        self.assertEqual(list(head.keys()), ["em", "f1"])
        self.assertEqual(head["em"], 1.0)

    def test_register_profile(self):
        profiles.register_profile("custom-bench", ("em", "coverage"))
        self.assertEqual(profiles.headline_metrics("custom-bench"), ("em", "coverage"))


if __name__ == "__main__":
    unittest.main()
