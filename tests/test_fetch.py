"""SB-fetch · the fetch normalizers + I/O, proven WITHOUT a network.

For every dataset we feed a tiny synthetic payload in that source's *documented* raw schema, run the
pure normalizer, and round-trip it through the REAL loader — asserting the Example fields (golds +
context/corpus) come out right. This is the guarantee that `june-bench fetch` writes loader-correct
data; the live URLs are validated on the user's machine by the same round-trip in `fetch_one`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.datasets import registry                         # noqa: E402
from june_bench.datasets.fetch import fetch_one                  # noqa: E402
from june_bench.datasets.fetch_sources import SOURCES            # noqa: E402


def _load_via_loader(name: str, normalized, data_dir: pathlib.Path):
    """Write the normalized blob under the loader's full_file, then load the FULL split."""
    src = SOURCES[name]
    (data_dir / src.full_file).write_text(json.dumps(normalized), encoding="utf-8")
    prev = os.environ.get("JUNE_BENCH_DATA")
    os.environ["JUNE_BENCH_DATA"] = str(data_dir)
    try:
        return list(registry.get(name).load("full"))
    finally:
        if prev is None:
            os.environ.pop("JUNE_BENCH_DATA", None)
        else:
            os.environ["JUNE_BENCH_DATA"] = prev


class TestHotpotFamily(unittest.TestCase):
    def test_hotpot_passthrough_and_context(self):
        raw = [{"_id": "h1", "question": "Q?", "answer": "A",
                "context": [["T", ["s1.", "s2."]], ["U", ["s3."]]]}]
        with tempfile.TemporaryDirectory() as d:
            ex = _load_via_loader("hotpot", SOURCES["hotpot"].normalize(raw), pathlib.Path(d))
        self.assertEqual(ex[0].qid, "h1")
        self.assertEqual(ex[0].golds, ("A",))
        self.assertIn("T: s1. s2.", ex[0].context)     # title prepended to its joined sentences

    def test_2wiki_id_to_underscore_id(self):
        raw = [{"id": "w1", "question": "Q?", "answer": "A", "context": [["T", ["s."]]]}]
        with tempfile.TemporaryDirectory() as d:
            ex = _load_via_loader("2wiki", SOURCES["2wiki"].normalize(raw), pathlib.Path(d))
        self.assertEqual(ex[0].qid, "w1")               # `id` mapped to `_id`
        self.assertEqual(ex[0].golds, ("A",))

    def test_musique_paragraphs_to_context(self):
        raw = [{"id": "m1", "question": "Q?", "answer": "A",
                "paragraphs": [{"title": "T", "paragraph_text": "P body."}]}]
        with tempfile.TemporaryDirectory() as d:
            ex = _load_via_loader("musique", SOURCES["musique"].normalize(raw), pathlib.Path(d))
        self.assertEqual(ex[0].golds, ("A",))
        self.assertIn("T: P body.", ex[0].context)     # MuSiQue title prepended to its paragraph


class TestJuneFamily(unittest.TestCase):
    def test_locomo_turns_to_corpus(self):
        # real LoCoMo: evidence is a dialog-id `D<session>:<turn>` → gold SESSION; one doc per session
        # (turns joined), id `<sample>#<qa>::<session>` — must match the local converter's structure.
        raw = [{"sample_id": "c1",
                "conversation": {"session_1_date_time": "1 Jan",
                                 "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "hi bob"}]},
                "qa": [{"question": "who?", "answer": "Alice", "category": 1, "evidence": ["D1:1"]}]}]
        with tempfile.TemporaryDirectory() as d:
            ex = _load_via_loader("locomo", SOURCES["locomo"].normalize(raw), pathlib.Path(d))
        self.assertEqual(ex[0].qid, "c1#0")                                # per-question id
        self.assertEqual(ex[0].golds, ("Alice",))
        self.assertTrue(any("Alice: hi bob" in c for c in ex[0].corpus))   # session became a corpus doc
        # retrieval gold MUST be in the corpus id-space (else recall is 0 by construction — the bug)
        gold = set(ex[0].meta.get("gold_ids", []))
        corpus_ids = {i for i, _t in (ex[0].meta.get("corpus_docs") or [])}
        self.assertTrue(gold and gold <= corpus_ids, f"locomo gold {gold} not in corpus {corpus_ids}")

    def test_longmemeval_sessions_to_corpus(self):
        raw = [{"question_id": "q1", "question": "what?", "answer": "42",
                "question_type": "single-session", "answer_session_ids": ["s0"],
                "haystack_session_ids": ["s0"],
                "haystack_sessions": [[{"role": "user", "content": "the answer is 42"}]]}]
        with tempfile.TemporaryDirectory() as d:
            ex = _load_via_loader("longmemeval", SOURCES["longmemeval"].normalize(raw), pathlib.Path(d))
        self.assertEqual(ex[0].qid, "q1")
        self.assertEqual(ex[0].golds, ("42",))
        # retrieval gold MUST be in the corpus id-space (regression guard for the id-mismatch bug)
        gold = set(ex[0].meta.get("gold_ids", []))
        corpus_ids = {i for i, _t in (ex[0].meta.get("corpus_docs") or [])}
        self.assertTrue(gold and gold <= corpus_ids, f"lme gold {gold} not in corpus {corpus_ids}")
        self.assertTrue(any("the answer is 42" in c for c in ex[0].corpus))

    def test_financebench_evidence_to_corpus(self):
        raw = [{"financebench_id": "f1", "question": "revenue?", "answer": "$1B",
                "evidence": [{"evidence_text": "Revenue was $1B."}]}]
        with tempfile.TemporaryDirectory() as d:
            ex = _load_via_loader("financebench", SOURCES["financebench"].normalize(raw), pathlib.Path(d))
        self.assertEqual(ex[0].golds, ("$1B",))
        self.assertTrue(any("Revenue was $1B." in c for c in ex[0].corpus))


class TestFetchOneIO(unittest.TestCase):
    """Exercise the parse + write + round-trip path (no network) via --from local raw files."""

    def test_fetch_one_from_local_json(self):
        raw = [{"_id": "h1", "question": "Q?", "answer": "A", "context": [["T", ["s."]]]}]
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src_file = d / "raw.json"
            src_file.write_text(json.dumps(raw))
            status, msg = fetch_one(SOURCES["hotpot"], d / "data", raw_from=str(src_file))
        self.assertEqual(status, "ok", msg)
        self.assertIn("1 examples", msg)

    def test_fetch_one_from_local_jsonl(self):
        rows = [{"id": "m1", "question": "Q?", "answer": "A",
                 "paragraphs": [{"title": "T", "paragraph_text": "P."}]}]
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src_file = d / "raw.jsonl"
            src_file.write_text("\n".join(json.dumps(r) for r in rows))
            status, msg = fetch_one(SOURCES["musique"], d / "data", raw_from=str(src_file))
        self.assertEqual(status, "ok", msg)

    def test_fetch_one_skips_existing(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "data").mkdir()
            (d / "data" / SOURCES["hotpot"].full_file).write_text("[]")
            status, _ = fetch_one(SOURCES["hotpot"], d / "data")
            self.assertEqual(status, "skipped")

    def test_bad_payload_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src_file = d / "raw.json"
            src_file.write_text('{"not": "a list of qa"}')   # normalizes to 0 examples
            status, msg = fetch_one(SOURCES["hotpot"], d / "data", raw_from=str(src_file))
            self.assertEqual(status, "failed", msg)
            self.assertFalse((d / "data" / SOURCES["hotpot"].full_file).exists())  # cleaned up


if __name__ == "__main__":
    unittest.main()
