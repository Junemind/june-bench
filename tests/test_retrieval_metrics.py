"""Retrieval metrics + turn-grain — pure unit tests (no June, no network)."""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.retrieval import (  # noqa: E402
    collapse_turns,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)


class TestMetrics(unittest.TestCase):
    def test_recall_at_k_multi_gold(self):
        ranking = ["d3", "d1", "d9", "d2"]
        gold = {"d1", "d2"}
        self.assertEqual(recall_at_k(ranking, gold, 1), 0.0)        # d3 not gold
        self.assertEqual(recall_at_k(ranking, gold, 2), 0.5)        # d1 in top-2 → 1/2
        self.assertEqual(recall_at_k(ranking, gold, 4), 1.0)        # both by rank 4
        self.assertEqual(recall_at_k(ranking, set(), 5), 0.0)       # no gold → 0

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a", "g", "b"], {"g"}), 0.5)   # first relevant at rank 2
        self.assertEqual(reciprocal_rank(["g", "a"], {"g"}), 1.0)
        self.assertEqual(reciprocal_rank(["a", "b"], {"g"}, k=2), 0.0)   # none in top-2

    def test_ndcg_at_k(self):
        # gold ranked first → perfect nDCG; gold at rank 2 with one gold → 1/log2(3) / 1
        self.assertAlmostEqual(ndcg_at_k(["g", "x"], {"g"}, 2), 1.0)
        self.assertAlmostEqual(ndcg_at_k(["x", "g"], {"g"}, 2), (1 / math.log2(3)) / 1.0)

    def test_turn_grain_collapse(self):
        # chunk ids collapse to parent, keeping the best (earliest) rank per parent
        ranking = ["c1::t3", "c2::t1", "c1::t1", "c3::t2"]
        self.assertEqual(collapse_turns(ranking), ["c1", "c2", "c3"])

    def test_score_retrieval_excludes_goldless_and_does_turn_grain(self):
        rankings = {
            "q1": ["c1::t2", "c2::t1"],     # gold parent c1 at rank 1 after collapse
            "q2": ["c9::t1"],               # no gold → excluded
        }
        golds = {"q1": ["c1::t5"], "q2": []}
        s = score_retrieval(rankings, golds, ks=(1, 5), turn_grain=True)
        self.assertEqual(s["n_queries"], 1)                  # q2 excluded (no gold)
        self.assertEqual(s["recall"][1], 1.0)                # c1 found at collapsed rank 1
        self.assertEqual(s["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
