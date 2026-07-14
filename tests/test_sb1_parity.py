"""SB1 · scorer parity — `june_bench.score` is behaviourally identical to the in-repo canonical
metrics (`benchmarks/apex_qa/metrics.py` + `answer_gate.evaluate`).

`june_bench` must install **standalone**, so it cannot import the repo's `apex_qa` at runtime — it
ships its own copy of the canonical SQuAD/HotpotQA metrics. This test is the guarantee that the copy
is not a *fork*: it fails the moment `june_bench.score` drifts from the canonical source. It is a
**repo-development** test — `skipUnless` when `apex_qa` isn't importable (e.g. the package installed
on its own), so the standalone test suite never hard-fails.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "june_bench"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "benchmarks" / "apex_qa"))

# import the metric fns directly (the package re-exports a `score()` fn that shadows the submodule
# name, so `import june_bench.score` would resolve to the function via attribute access).
from june_bench.score import (                # noqa: E402
    answer_in_context as jb_answer_in_context,
    exact_match as jb_exact_match,
    normalize_answer as jb_normalize_answer,
    score as jb_score,
    token_f1 as jb_token_f1,
)
from june_bench.ports import Record           # noqa: E402

try:
    import metrics as apex                     # benchmarks/apex_qa/metrics.py
    _HAVE_APEX = True
except Exception:
    _HAVE_APEX = False

try:
    import answer_gate as gate                 # benchmarks/apex_qa/answer_gate.py
    _HAVE_GATE = True
except Exception:
    _HAVE_GATE = False


_PAIRS = [
    ("The Beatles.", "beatles"),
    ("Ada Lovelace", "ada lovelace"),
    ("Paris, France", "paris"),
    ("", ""),
    ("nothing matches", "Mars"),
    ("a an the of", "the of a an"),
    ("Leonardo da Vinci", "da Vinci"),
    ("42", "forty two"),
]


@unittest.skipUnless(_HAVE_APEX, "benchmarks/apex_qa/metrics.py not importable (standalone install)")
class TestMetricParity(unittest.TestCase):
    def test_normalize_identical(self):
        for p, _ in _PAIRS:
            self.assertEqual(jb_normalize_answer(p), apex.normalize_answer(p))

    def test_exact_match_identical(self):
        for p, g in _PAIRS:
            self.assertEqual(jb_exact_match(p, g), apex.exact_match(p, g))

    def test_token_f1_identical(self):
        for p, g in _PAIRS:
            self.assertAlmostEqual(jb_token_f1(p, g), apex.token_f1(p, g), places=9)

    def test_answer_in_context_identical(self):
        ctx = "Ada Lovelace wrote the first algorithm in Paris."
        for golds in (["Ada Lovelace"], ["Mars"], ["paris"], []):
            self.assertEqual(jb_answer_in_context(ctx, golds), apex.answer_in_context(ctx, golds))


@unittest.skipUnless(_HAVE_GATE, "answer_gate not importable")
class TestSummaryParityWithAnswerGate(unittest.TestCase):
    def _records(self):
        return [
            Record(qid="1", question="q", golds=("Paris",), prediction="Paris",
                   context=("answer is Paris",), calls=1, cost=0.002),
            Record(qid="2", question="q", golds=("Rome",), prediction="", context=("Rome here",),
                   calls=1, cost=0.001, abstained=True),
            Record(qid="3", question="q", golds=("Mars",), prediction="Venus",
                   context=("Mars is red",), calls=3, cost=0.006),
        ]

    def test_em_f1_coverage_cost_match_answer_gate(self):
        recs = self._records()
        jb_sum = jb_score(recs)
        ag = gate.evaluate([
            gate.AnswerRecord(golds=r.golds, prediction=r.prediction,
                              context_text=" ".join(r.context), calls=r.calls, cost=r.cost,
                              abstained=r.abstained) for r in recs])
        self.assertEqual(jb_sum["em"], ag.em)
        self.assertEqual(jb_sum["f1"], ag.f1)
        self.assertEqual(jb_sum["coverage"], ag.coverage)
        self.assertEqual(jb_sum["context_recall"], ag.context_recall)
        self.assertEqual(jb_sum["cost_per_answer"], ag.cost_per_answer)
        self.assertEqual(jb_sum["calls_per_answer"], ag.calls_per_answer)


if __name__ == "__main__":
    unittest.main()
