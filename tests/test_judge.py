"""LLM-judge correctness — pure prompt/parse + selective scoring + the report column."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.judge import build_judge_prompt, judge_records, parse_verdict  # noqa: E402
from june_bench.ports import Record  # noqa: E402
from june_bench.report import to_markdown  # noqa: E402


class TestJudge(unittest.TestCase):
    def test_parse_verdict(self):
        for yes in ("yes", "Yes.", "YES", "correct", "y", "  yes  "):
            self.assertEqual(parse_verdict(yes), 1.0, yes)
        for no in ("no", "No.", "incorrect", "", "maybe", "I think yes"):  # only a leading yes counts
            self.assertEqual(parse_verdict(no), 0.0, no)

    def test_prompt_carries_q_refs_pred(self):
        system, user = build_judge_prompt("Who?", ["Alice", "A. Smith"], "Alice Smith is the one.")
        self.assertIn("yes or no", system.lower())
        self.assertIn("QUESTION: Who?", user)
        self.assertIn("Alice | A. Smith", user)            # multi-gold joined
        self.assertIn("Alice Smith is the one.", user)

    def test_judge_records_selective_and_failsoft(self):
        recs = [
            Record(qid="1", question="q", golds=("alpha",), prediction="alpha is the answer", abstained=False),
            Record(qid="2", question="q", golds=("alpha",), prediction="beta", abstained=False),
            Record(qid="3", question="q", golds=("alpha",), prediction="", abstained=True),   # excluded
        ]
        # fake judge: correct iff a gold appears as a token in the prediction
        def jf(question, golds, prediction):  # noqa: ANN001
            toks = set(prediction.lower().split())
            return 1.0 if any(g.lower() in toks for g in golds) else 0.0
        acc = judge_records(recs, jf)
        self.assertAlmostEqual(acc, 0.5)                   # 1 of the 2 answered judged correct

    def test_failed_judge_call_is_excluded_not_scored_wrong(self):
        # A judge CALL that raises (rate-limit/network) must be EXCLUDED, never scored 0.0 — else a
        # correct system looks wrong (the judge < EM impossibility). Here q2's judge raises; the score
        # must be 1.0 (the one that graded, correct), NOT 0.5 (which would count the failure as wrong).
        recs = [
            Record(qid="1", question="q", golds=("alpha",), prediction="alpha", abstained=False),
            Record(qid="2", question="q", golds=("alpha",), prediction="alpha", abstained=False),
        ]
        calls = {"n": 0}
        def flaky(question, golds, prediction):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("HTTP 429 rate limited")
            return 1.0
        self.assertAlmostEqual(judge_records(recs, flaky), 1.0)   # over the 1 that graded, not 0.5

    def test_all_judge_calls_failing_does_not_crash(self):
        recs = [Record(qid="1", question="q", golds=("a",), prediction="a", abstained=False)]
        def always_fail(question, golds, prediction):  # noqa: ANN001
            raise RuntimeError("down")
        self.assertEqual(judge_records(recs, always_fail), 0.0)   # empty mean, but a warning fired

    def test_report_adds_judge_column_when_present(self):
        s = {"n": 2, "answered": 2, "em": 0.5, "f1": 0.6, "coverage": 1.0,
             "context_recall": 1.0, "calls_per_answer": 1.0, "cost_per_answer": 0.0, "judge": 0.75}
        md = to_markdown(s, system="june-api", dataset="hotpot", split="full")
        self.assertIn("judge", md.splitlines()[2])         # header has the column
        self.assertIn("0.75", md)
        # and absent when not judged:
        s2 = dict(s)
        s2.pop("judge")
        self.assertNotIn("judge", to_markdown(s2, system="x", dataset="y", split="z"))


if __name__ == "__main__":
    unittest.main()
