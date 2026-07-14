"""Limit-aware streaming June loader — the memory fix for the big splits (locomo 164MB / lme 253MB).

`_june_examples_streaming` (ijson) must produce EXACTLY the same Examples as the full `json.load`
path for the first N queries — it just avoids materializing the whole file. Validated against the
bundled smoke fixture (small, offline); the memory win is inherent to streaming.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.datasets import loaders  # noqa: E402

_FIX = pathlib.Path(loaders.__file__).resolve().parent / "fixtures" / "june.smoke.json"

try:
    import ijson  # noqa: F401
    _HAVE_IJSON = True
except Exception:
    _HAVE_IJSON = False


def _key(ex):
    return (ex.qid, ex.question, ex.golds, ex.corpus,
            tuple(ex.meta.get("gold_ids", [])), tuple(ex.meta.get("corpus_docs", [])))


class TestLimitInFullLoader(unittest.TestCase):
    def test_full_loader_respects_limit(self):
        blob = loaders._read_json(_FIX)
        full = loaders._june_examples(blob)
        n = max(1, len(full) - 1) if len(full) > 1 else 1
        limited = loaders._june_examples(blob, limit=n)
        self.assertEqual(len(limited), min(n, len(full)))
        self.assertEqual([_key(e) for e in limited], [_key(e) for e in full[:n]])


@unittest.skipUnless(_HAVE_IJSON, "ijson not installed ([stream] extra)")
class TestStreamingMatchesFull(unittest.TestCase):
    def test_streaming_equals_full_for_first_n(self):
        full = loaders._june_examples(loaders._read_json(_FIX))
        for n in (1, len(full)):
            streamed = loaders._june_examples_streaming(_FIX, limit=n)
            self.assertIsNotNone(streamed)
            self.assertEqual(len(streamed), min(n, len(full)))
            self.assertEqual([_key(e) for e in streamed], [_key(e) for e in full[:n]],
                             f"streaming != full at limit={n}")

    def test_dataset_load_smoke_with_limit(self):
        # the registered dataset's load(split, limit) path (smoke uses the fixture, limit honored)
        from june_bench.datasets import registry
        ds = registry.get("locomo")
        one = list(ds.load("smoke", limit=1))
        self.assertEqual(len(one), 1)
        self.assertTrue(one[0].meta.get("corpus_docs") is not None)


class TestFallbackWithoutIjson(unittest.TestCase):
    def test_streaming_returns_none_without_ijson(self):
        # simulate ijson missing → returns None so the caller falls back to the full loader
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"ijson": None}):
            self.assertIsNone(loaders._june_examples_streaming(_FIX, limit=1))


if __name__ == "__main__":
    unittest.main()
