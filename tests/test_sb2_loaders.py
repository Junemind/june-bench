"""SB2 · the four benchmark loaders.

The bundled `smoke` slices load offline (no data dir) and parse into well-formed Examples for both
formats (HotpotQA context-QA + June memory-QA). The `full` splits are read from the repo data dir
when present (a dev-only check, skipped on a standalone install).
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import run, score                       # noqa: E402
from june_bench.datasets import registry                # noqa: E402
from june_bench.datasets.loaders import _resolve        # noqa: E402
from june_bench.systems.base import EchoSystem          # noqa: E402

_HOTPOT = ("hotpot", "2wiki", "musique")
_JUNE = ("locomo", "longmemeval", "financebench")


class TestRegistration(unittest.TestCase):
    def test_all_six_registered(self):
        for name in (*_HOTPOT, *_JUNE):
            self.assertIn(name, registry.names())


class TestHotpotSmoke(unittest.TestCase):
    def test_smoke_parses_context_qa(self):
        ex = list(registry.get("hotpot").load("smoke"))
        self.assertEqual(len(ex), 2)
        self.assertTrue(ex[0].question)
        self.assertTrue(ex[0].golds)
        self.assertTrue(ex[0].context)          # distractor passages present
        self.assertFalse(ex[0].corpus)

    def test_2wiki_and_musique_share_the_format(self):
        for name in ("2wiki", "musique"):
            ex = list(registry.get(name).load("smoke"))
            self.assertTrue(ex and ex[0].context)


class TestJuneSmoke(unittest.TestCase):
    def test_smoke_parses_memory_qa(self):
        ex = list(registry.get("locomo").load("smoke"))
        self.assertEqual(len(ex), 2)
        self.assertTrue(ex[0].corpus)            # the conversation's docs to ingest
        self.assertFalse(ex[0].context)
        self.assertTrue(ex[0].golds)
        self.assertIn("question_type", ex[0].meta)

    def test_corpus_is_conversation_scoped(self):
        ex = {e.qid: e for e in registry.get("locomo").load("smoke")}
        # conv-smoke#0 must only see its own two session docs, not conv-smoke#1's.
        self.assertEqual(len(ex["conv-smoke#0"].corpus), 2)
        self.assertTrue(all("LGBTQ" in c or "weather" in c for c in ex["conv-smoke#0"].corpus))

    def test_lme_and_financebench_share_the_format(self):
        for name in ("longmemeval", "financebench"):
            ex = list(registry.get(name).load("smoke"))
            self.assertTrue(ex and ex[0].corpus)


class TestEndToEndOnSmoke(unittest.TestCase):
    def test_echo_oracle_scores_em_1_on_each(self):
        for name in (*_HOTPOT, *_JUNE):
            s = score(run(EchoSystem(), registry.get(name), split="smoke"))
            self.assertEqual(s["em"], 1.0, f"{name} smoke EM")


class TestFullSplitWhenDataPresent(unittest.TestCase):
    @unittest.skipUnless(_resolve("hotpot_dev.json"), "repo hotpot data not present (standalone)")
    def test_hotpot_full_loads(self):
        ex = list(registry.get("hotpot").load("full"))
        self.assertGreater(len(ex), 10)
        self.assertTrue(ex[0].question and ex[0].context)

    @unittest.skipUnless(_resolve("locomo.june.json"), "repo locomo data not present (standalone)")
    def test_locomo_full_loads_with_scoped_corpus(self):
        ex = list(registry.get("locomo").load("full"))
        self.assertGreater(len(ex), 100)
        self.assertTrue(ex[0].question and ex[0].corpus)


if __name__ == "__main__":
    unittest.main()
