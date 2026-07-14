"""Answer-dump (--dump) — confirm per-question predictions are written to JSONL.

Guards the one-flag replacement for manual curl dumps: when a strict-EM run shows EM≈0, the dump lets
you SEE whether it's a *verbose-but-correct* answer (high ``n_pred_words``) or a real miss. Runs under
plain `unittest` (in-sandbox, no pytest) AND under pytest on the dev machine — no model/network.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.cli import main as cli_main   # noqa: E402


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(ln) for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


class TestDumpAnswers(unittest.TestCase):
    def test_run_dump_writes_one_line_per_question(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(pathlib.Path(d) / "ans.jsonl")
            rc = cli_main(["run", "--system", "echo", "--dataset", "smoke", "--dump", out])
            self.assertEqual(rc, 0)
            rows = _read_jsonl(out)
            self.assertEqual(len(rows), 4)                       # the bundled smoke fixture has 4 Qs
            r0 = rows[0]
            # the schema a diagnosis needs: prediction next to gold, plus the verbosity signal
            for key in ("system", "dataset", "qid", "question", "golds", "prediction",
                        "n_pred_words", "abstained", "system_error"):
                self.assertIn(key, r0)
            self.assertEqual(r0["system"], "echo")
            # echo answers with the gold, so word count is the gold's length (the verbose-zero detector)
            self.assertEqual(r0["n_pred_words"], len(r0["prediction"].split()))

    def test_suite_dump_tags_each_system(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(pathlib.Path(d) / "suite.jsonl")
            md = str(pathlib.Path(d) / "r.md")
            rc = cli_main(["suite", "--systems", "echo,null", "--datasets", "smoke",
                           "--dump", out, "--out", md])
            self.assertEqual(rc, 0)
            rows = _read_jsonl(out)
            self.assertEqual(len(rows), 8)                       # 2 systems × 4 questions
            self.assertEqual({r["system"] for r in rows}, {"echo", "null"})

    def test_no_dump_flag_writes_nothing(self):
        # absent --dump, the run must not create a file (default "" → disabled)
        with tempfile.TemporaryDirectory() as d:
            sentinel = pathlib.Path(d) / "should_not_exist.jsonl"
            rc = cli_main(["run", "--system", "echo", "--dataset", "smoke"])
            self.assertEqual(rc, 0)
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
