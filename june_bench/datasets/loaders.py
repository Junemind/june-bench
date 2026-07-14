"""Dataset loaders (SB0 smoke + SB2 the four real benchmarks).

Two formats cover all six datasets, so each is one small adapter behind the `Dataset` port:

* **HotpotQA format** (`hotpot` / `2wiki` / `musique`) — `[{_id, question, answer, context:[[title,
  [sents]]]}]`. Context-QA: the distractor passages ride in `Example.context`.
* **June format** (`locomo` / `longmemeval` / `financebench`) — `{documents:[{id,text}], queries:[{query,
  answer, gold, haystack, question_type}]}` with **conversation-scoped** doc ids (`<conv>::<chunk>`).
  Memory-QA: the conversation's documents ride in `Example.corpus` (a memory system ingests them, then
  answers); `gold`/`question_type` ride in `meta`.

Reproducibility: the bundled few-KB fixtures (`fixtures/*.smoke.json`) make `--split smoke` run offline.
The full splits are read from the data dir (`JUNE_BENCH_DATA`, else the repo `data/` / `benchmarks/apex_qa/`
for in-repo dev); if absent, the loader raises with a `june-bench fetch` hint — and `fetch` (see
`fetch_sources.py` / `fetch.py`) downloads + normalizes those splits into exactly these shapes.
"""
from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Sequence

from june_bench._util import env_int
from june_bench.ports import Example

_HERE = pathlib.Path(__file__).resolve().parent
_FIXTURES = _HERE / "fixtures"


def _max_corpus() -> int:
    """Corpus docs fed per memory question, read PER LOAD (so a test/CLI setting the env after import
    takes effect) and crash-proof. 200 keeps a full run tractable (≈202 reqs/question, under the 600/60
    rate-limit bucket); raise via ``JUNE_BENCH_MAX_CORPUS`` for a full-haystack stress (mind the limiter)."""
    return env_int("JUNE_BENCH_MAX_CORPUS", 200, lo=1)
# In-repo dev locations (a standalone install relies on JUNE_BENCH_DATA / the fetch cache instead).
# loaders.py lives at <repo>/june_bench/june_bench/datasets/loaders.py → repo root is parents[2].
_REPO = _HERE.parents[2] if len(_HERE.parents) >= 3 else _HERE
_REPO_DATA = _REPO / "data"
_REPO_APEX = _REPO / "benchmarks" / "apex_qa"
# Where `june-bench fetch` writes (JUNE_BENCH_DATA else ~/.cache/june-bench/data). The loader MUST
# search here too, or `fetch` then `suite`/`retrieve` fails with "file not found" on a standalone
# install (the file is downloaded but never looked for) — the #1 reproduction snag for other users.
_FETCH_CACHE = pathlib.Path.home() / ".cache" / "june-bench" / "data"


def _read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(filename: str) -> pathlib.Path | None:
    for base in (os.environ.get("JUNE_BENCH_DATA"), _FETCH_CACHE, _REPO_DATA, _REPO_APEX):
        if not base:
            continue
        p = pathlib.Path(base) / filename
        if p.exists():
            return p
    return None


def _need(filename: str) -> pathlib.Path:
    p = _resolve(filename)
    if p is None:
        raise FileNotFoundError(
            f"dataset file {filename!r} not found. Set JUNE_BENCH_DATA to the data dir, "
            f"or run `june-bench fetch` to download the full splits (the bundled `smoke` split "
            f"needs no data).")
    return p


# ── HotpotQA format (hotpot / 2wiki / musique) ──────────────────────────────────────────────
def _hotpot_context(ctx) -> tuple[str, ...]:
    out: list[str] = []
    for entry in ctx or []:
        # entry is [title, [sentences]]. KEEP the title: on HotpotQA/2Wiki/MuSiQue the passage title
        # is the entity the passage is about — frequently the answer itself or the multi-hop bridge —
        # and the reference harness shows it to the model ("[i] {title}: {body}"). Dropping it
        # handicaps the answerer (measured ~1 Q/24 on the matched HotpotQA slice).
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[1], (list, tuple)):
            title = str(entry[0]).strip()
            body = " ".join(str(s) for s in entry[1]).strip()
            out.append(f"{title}: {body}" if title and body else (body or title))
        elif isinstance(entry, str):
            out.append(entry)
    return tuple(p for p in out if p)


def _hotpot_examples(raw: Sequence[dict]) -> list[Example]:
    out = []
    for i, r in enumerate(raw):
        ans = r.get("answer", "")
        out.append(Example(
            qid=str(r.get("_id", i)), question=str(r.get("question", "")),
            golds=tuple([ans]) if ans else (), context=_hotpot_context(r.get("context")),
            meta={"question_type": r.get("type", ""), "level": r.get("level", "")}))
    return out


class HotpotFormatDataset:
    def __init__(self, name: str, full_file: str, *, fixture: str = "hotpot.smoke.json") -> None:
        self.name = name
        self._full = full_file
        self._fixture = fixture

    def load(self, split: str = "smoke", limit: int | None = None) -> Sequence[Example]:
        path = _FIXTURES / self._fixture if split == "smoke" else _need(self._full)
        out = _hotpot_examples(_read_json(path))
        return out[:limit] if limit else out


# ── June format (locomo / longmemeval / financebench) ───────────────────────────────────────
def _conv_key(doc_or_query_id: str) -> str:
    return str(doc_or_query_id).split("::", 1)[0]


def _make_june_example(q: dict, docs: list[tuple[str, str]]) -> Example:
    """Build one Example from a query + its (id,text) corpus docs (shared by full + streaming loaders)."""
    ans = q.get("answer")
    return Example(
        qid=str(q.get("id", "")), question=str(q.get("query", "")),
        golds=tuple([str(ans)]) if ans not in (None, "") else (),
        corpus=tuple(t for _id, t in docs),                       # QA mode: text only (unchanged)
        meta={"question_type": q.get("question_type", ""),
              "gold_ids": list(q.get("gold", []) or []),
              "corpus_docs": docs})                               # retrieval mode: (id, text) pairs


_gold_trunc_warned = False


def _warn_if_gold_truncated(q: dict, docs: list, mc: int, *, truncated: bool,
                            total: int | None = None) -> None:
    """M11: the corpus cap silently drops docs beyond ``mc``. If a **gold** doc is among the dropped,
    recall is capped by construction for that query — warn once (with the fix) rather than lose it
    silently. No-op when nothing was truncated or no gold fell outside the cap."""
    global _gold_trunc_warned
    if _gold_trunc_warned or not truncated:
        return
    gold = {str(g) for g in (q.get("gold", []) or [])}
    if gold and (gold - {i for i, _t in docs}):
        import sys
        extra = f"; this conversation has {total} docs" if total else ""
        sys.stderr.write(
            f"[june-bench] a gold doc for qid={q.get('id', '?')} was dropped by the corpus cap "
            f"(JUNE_BENCH_MAX_CORPUS={mc}{extra}) — recall is capped for such queries. "
            f"Raise JUNE_BENCH_MAX_CORPUS to include them.\n")
        _gold_trunc_warned = True


def _june_examples(blob: dict, *, max_corpus: int | None = None,
                   limit: int | None = None) -> list[Example]:
    # keep (id, text) per doc so retrieval mode can ingest id-preserving docs and score rankings
    # against gold_ids; `corpus` (text-only) stays for QA mode (unchanged).
    mc = max_corpus if max_corpus is not None else _max_corpus()
    docs_by_conv: dict[str, list[tuple[str, str]]] = {}
    for d in blob.get("documents", []):
        docs_by_conv.setdefault(_conv_key(d.get("id", "")), []).append(
            (str(d.get("id", "")), str(d.get("text", ""))))
    out = []
    for q in blob.get("queries", []):
        if limit is not None and len(out) >= limit:
            break
        full = docs_by_conv.get(_conv_key(q.get("id", "")), [])
        docs = full[:mc]
        _warn_if_gold_truncated(q, docs, mc, truncated=len(full) > mc, total=len(full))
        out.append(_make_june_example(q, docs))
    return out


def _june_examples_streaming(path: pathlib.Path, *, limit: int,
                             max_corpus: int | None = None) -> list[Example] | None:
    """Memory-lean limit-aware load via **ijson** (streaming): materialize only the first ``limit``
    queries and only the documents for THOSE queries' conversations — instead of `json.load`-ing the
    whole 150-250 MB file (~1 GB of Python objects) just to take the first N. Two streaming passes:
    (1) queries → first N + their conv keys; (2) documents → keep only those conversations (capped).
    Returns None if ijson isn't installed (caller falls back to the full loader)."""
    try:
        import ijson
    except Exception:
        return None
    mc = max_corpus if max_corpus is not None else _max_corpus()
    # pass 1: first `limit` queries (small) + the conversations they need
    queries: list[dict] = []
    needed: set[str] = set()
    with open(path, "rb") as fh:
        for q in ijson.items(fh, "queries.item"):
            queries.append(q)
            needed.add(_conv_key(q.get("id", "")))
            if len(queries) >= limit:
                break
    # pass 2: only the documents for those conversations (bounded to N convs × mc). Track which convs
    # had MORE docs than the cap, so a dropped gold can be flagged (M11) instead of silently lost.
    docs_by_conv: dict[str, list[tuple[str, str]]] = {}
    capped: set[str] = set()
    with open(path, "rb") as fh:
        for d in ijson.items(fh, "documents.item"):
            ck = _conv_key(d.get("id", ""))
            if ck not in needed:
                continue
            lst = docs_by_conv.setdefault(ck, [])
            if len(lst) < mc:
                lst.append((str(d.get("id", "")), str(d.get("text", ""))))
            else:
                capped.add(ck)
    out: list[Example] = []
    for q in queries:
        ck = _conv_key(q.get("id", ""))
        docs = docs_by_conv.get(ck, [])
        _warn_if_gold_truncated(q, docs, mc, truncated=ck in capped)
        out.append(_make_june_example(q, docs))
    return out


class JuneFormatDataset:
    def __init__(self, name: str, full_file: str, *, fixture: str = "june.smoke.json") -> None:
        self.name = name
        self._full = full_file
        self._fixture = fixture

    def load(self, split: str = "smoke", limit: int | None = None) -> Sequence[Example]:
        if split == "smoke":
            return _june_examples(_read_json(_FIXTURES / self._fixture), limit=limit)
        path = _need(self._full)
        # Big full splits (locomo 164MB / lme 253MB): when a limit is set, STREAM only the needed
        # queries+docs via ijson (avoids the ~1GB whole-file parse that OOMs a memory-tight box).
        # No limit, or ijson missing ⇒ the full in-memory load (correct, just heavier).
        if limit is not None and limit > 0:
            streamed = _june_examples_streaming(path, limit=limit)
            if streamed is not None:
                return streamed
        return _june_examples(_read_json(path), limit=limit)


# ── the bundled smoke trivia set (SB0) ──────────────────────────────────────────────────────
class SmokeDataset:
    name = "smoke"

    def load(self, split: str = "smoke", limit: int | None = None) -> Sequence[Example]:
        raws = [json.loads(ln) for ln in (_FIXTURES / "smoke.jsonl").read_text(
            encoding="utf-8").splitlines() if ln.strip()]
        out = [Example(qid=str(r.get("qid", i)), question=str(r.get("question", "")),
                       golds=tuple(r.get("golds", [])), context=tuple(r.get("context", [])))
               for i, r in enumerate(raws)]
        return out[:limit] if limit else out


__all__ = ["SmokeDataset", "HotpotFormatDataset", "JuneFormatDataset"]
