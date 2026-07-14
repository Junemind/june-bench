"""JuneRetrievalSystem — score June's RETRIEVAL (recall@k/nDCG/MRR), not its answers.

Where `JuneApiSystem` asks `/v1/answer` and scores the text, this ingests a query's corpus with
**id-preserving** docs (`POST /v1/ingest/docs` → `node_id = uuid5(namespace, doc_id)`) into a
per-query canvas, runs `POST /v1/search`, and maps the ranked `node_id`s back to dataset **doc-ids**
— so `june_bench.retrieval.score_retrieval` can compute recall@k / nDCG / MRR against the gold
doc-ids, the rulers `scripts/retrieval_benchmark.py` reports. `httpx` is the only dep (the
`[june-api]` extra); the server/model are external (over HTTP), so no June source ships here.
"""
from __future__ import annotations

import os
import uuid
from collections.abc import Sequence

from june_bench._util import env_float, env_int, retry_request, try_lock_checkpoint
from june_bench.ports import Example


class JuneRetrievalSystem:
    """HTTP client scoring June's retrieval. ``retrieve(example)`` → ranked dataset doc-ids."""
    name = "june-retrieval"

    def __init__(self, base_url: str, *, api_key: str = "", docs_path: str = "/v1/ingest/docs",
                 search_path: str = "/v1/search", canvas_path: str = "/v1/canvases",
                 backfill_path: str = "/v1/embeddings/backfill", timeout: float = 120.0,
                 transport=None, k: int = 10, isolate: bool = True, backfill: bool = False,
                 cleanup: bool = True, pooled: bool = False, deep: bool = False) -> None:  # noqa: ANN001
        if not base_url:
            raise ValueError("JuneRetrievalSystem needs a base_url (e.g. http://localhost:8000). "
                             "Set JUNE_BENCH_JUNE_URL or pass base_url=…")
        import httpx
        headers = {"content-type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._docs_path, self._search_path = docs_path, search_path
        self._canvas_path, self._backfill_path = canvas_path, backfill_path
        self._k, self._isolate, self._backfill, self._cleanup = k, isolate, backfill, cleanup
        # pooled=True → retrieve over a SHARED corpus (all docs ingested once), the harder
        # cross-conversation task that matches the local diagnosis; else per-query canvas (easy).
        self._pooled = pooled
        # deep=True → ask /v1/search to run the RERANK path (it only reranks when the request sets
        # `deep` AND a reranker is wired server-side). The diagnosis showed LoCoMo is ranking-bound —
        # gold sits at ranks 11..50 — so this is the lever that promotes it into the top-10.
        self._deep = deep
        self._pool_canvas: str | None = None
        self._pool_nid_to_doc: dict[str, str] = {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                         transport=transport, headers=headers)

    async def _create_canvas(self, name: str) -> str:
        # A busy endpoint may close a keep-alive after an unrelated 5xx, or lock briefly; a retry on a
        # fresh connection succeeds. Canvas creation is a small, safe-to-retry write (a duplicate empty
        # canvas is harmless).
        r = await retry_request(
            lambda: self._client.post(self._canvas_path, json={"name": name}),
            attempts=env_int("JUNE_BENCH_CANVAS_RETRIES", 4, lo=1, hi=50), base=1.5, cap=6.0)
        r.raise_for_status()
        return str(r.json()["canvas_id"])

    async def _ingest_docs(self, docs, headers: dict, *, batch: int = 500) -> uuid.UUID:  # noqa: ANN001
        """Id-preserving ingest of (id,text) docs in batches; returns the uuid5 namespace (constant
        across batches — the engine's fixed _DOCS_NS), so node_id = uuid5(ns, doc_id).

        The write is **idempotent** (a re-sent batch upserts the same node ids, never duplicates), so a
        transient 5xx / 'database is locked' / transport error is safely retried (jittered backoff) via
        the shared retry helper. Tunable with JUNE_BENCH_INGEST_RETRIES."""
        import sys

        attempts = env_int("JUNE_BENCH_INGEST_RETRIES", 8, lo=1, hi=100)
        ns = None
        for i in range(0, len(docs), batch):
            chunk = docs[i:i + batch]
            payload = {"docs": [{"id": d, "text": t} for d, t in chunk]}
            r = await retry_request(
                lambda p=payload: self._client.post(self._docs_path, headers=headers, json=p),
                attempts=attempts,
                on_first_retry=lambda: sys.stderr.write(
                    "[june-bench] ingest hit a transient server error (likely a DB lock) — retrying…\n"))
            r.raise_for_status()
            ns = uuid.UUID(r.json()["namespace"])
        return ns

    async def ingest_pool(self, examples: Sequence[Example]) -> int:
        """POOLED setup: ingest the UNION of all examples' corpus docs ONCE into a shared canvas, then
        backfill embeddings once. Each query later retrieves over this whole pool (cross-conversation).
        Dedup by doc-id. Returns the number of pooled docs. Idempotent guard via `_pool_canvas`."""
        seen: dict[str, str] = {}
        for ex in examples:
            pairs = ex.meta.get("corpus_docs") or [(f"{ex.qid}::d{i}", t)
                                                   for i, t in enumerate(ex.corpus)]
            for doc_id, text in pairs:
                did, txt = str(doc_id), str(text)
                if txt.strip() and did not in seen:
                    seen[did] = txt
        docs = list(seen.items())
        if not docs:
            return 0
        canvas = await self._create_canvas("ret-pool") if self._isolate else ""
        headers = {"X-Canvas": canvas} if canvas else {}
        ns = await self._ingest_docs(docs, headers)
        self._pool_canvas = canvas
        self._pool_nid_to_doc = {str(uuid.uuid5(ns, did)): did for did, _ in docs}
        if self._backfill:
            try:
                await self._client.post(self._backfill_path, json={}, headers=headers)
            except Exception:  # noqa: BLE001 — model-free endpoint 4xx's; sparse lane still ranks
                pass
        return len(docs)

    async def retrieve(self, example: Example) -> list[str]:
        """POOLED: just search the shared pool (must `ingest_pool` first) → ranked doc-ids across the
        WHOLE corpus. PER-QUERY (default): ingest the example's own corpus into its own canvas, search,
        delete. Both map `node_id`→`doc_id` via the uuid5 namespace and return ranked dataset doc-ids."""
        if self._pooled:
            headers = {"X-Canvas": self._pool_canvas} if self._pool_canvas else {}
            r2 = await self._client.post(self._search_path, headers=headers,
                                         json={"query": example.question, "limit": self._k,
                                               "deep": self._deep})
            r2.raise_for_status()
            ranked = [str(it.get("node_id")) for it in r2.json().get("items", [])]
            return [self._pool_nid_to_doc[n] for n in ranked if n in self._pool_nid_to_doc]
        docs = example.meta.get("corpus_docs") or [(f"{example.qid}::d{i}", t)
                                                    for i, t in enumerate(example.corpus)]
        docs = [(str(i), str(t)) for i, t in docs if str(t).strip()]
        if not docs:
            return []
        canvas = await self._create_canvas(f"ret-{example.qid}") if self._isolate else ""
        headers = {"X-Canvas": canvas} if canvas else {}
        ns = await self._ingest_docs(docs, headers)
        # node_id → doc_id for THIS query's corpus (the only docs in its canvas)
        nid_to_doc = {str(uuid.uuid5(ns, doc_id)): doc_id for doc_id, _ in docs}
        if self._backfill:
            try:
                await self._client.post(self._backfill_path, json={}, headers=headers)
            except Exception:  # noqa: BLE001
                pass
        r2 = await self._client.post(self._search_path, headers=headers,
                                     json={"query": example.question, "limit": self._k})
        r2.raise_for_status()
        ranked = [str(it.get("node_id")) for it in r2.json().get("items", [])]
        if self._cleanup and canvas:
            try:
                await self._client.delete(f"{self._canvas_path}/{canvas}", headers=headers)
            except Exception:  # noqa: BLE001
                pass
        # keep only this corpus's docs, mapped to dataset doc-ids, order preserved
        return [nid_to_doc[n] for n in ranked if n in nid_to_doc]

    async def cleanup_pool(self) -> None:
        """Delete the shared pool canvas (pooled mode), best-effort."""
        if self._pool_canvas:
            try:
                await self._client.delete(f"{self._canvas_path}/{self._pool_canvas}",
                                          headers={"X-Canvas": self._pool_canvas})
            except Exception:  # noqa: BLE001
                pass
            self._pool_canvas = None

    async def aclose(self) -> None:
        await self._client.aclose()


def _load_checkpoint(path: str | None) -> dict[str, list[str]]:
    """Load {qid: ranking} from a JSONL checkpoint (one record per completed query; last wins).
    Missing / unreadable / partially-written → best-effort (skip bad lines). Never raises."""
    done: dict[str, list[str]] = {}
    if not path:
        return done
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[str(rec["qid"])] = list(rec["ranking"])
                except (ValueError, KeyError, TypeError):
                    continue           # a torn last line from a hard kill — ignore it
    except OSError:
        return {}
    return done


def _append_checkpoint(path: str | None, qid: str, ranking: list[str]) -> None:
    """Append one completed query's ranking to the checkpoint (crash-safe: append-only JSONL)."""
    if not path:
        return
    import json
    import os as _os
    try:
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"qid": qid, "ranking": ranking}) + "\n")
            fh.flush()
            _os.fsync(fh.fileno())     # durable: a kill/crash mid-run never loses recorded progress
    except OSError:                     # a checkpoint write must never fail the run
        pass


def run_retrieval(system: JuneRetrievalSystem, examples: Sequence[Example],
                  *, on_progress=None, checkpoint_path: str | None = None,  # noqa: ANN001
                  ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Run retrieval over examples → ({qid: ranked doc-ids}, {qid: gold doc-ids}) for `score_retrieval`.

    Fail-soft per query (a flaky query → empty ranking, scored a no-hit) but ABORTS on a CASCADE of
    consecutive errors — the endpoint is DOWN (server crash / OOM), so a partial run is not a valid
    result and must not be silently scored (same honesty rule as the QA runner).

    ``checkpoint_path`` (optional): each successful query is appended to a JSONL file; on a re-run those
    qids are loaded and SKIPPED, so a dropped run resumes instead of restarting. Only successes are
    checkpointed — a query that errored is retried on the next run.

    ``on_progress(done, total)`` (optional) is called after each query for a live progress bar."""
    import asyncio
    import sys

    max_consec = env_int("JUNE_BENCH_MAX_CONSEC_ERRORS", 8, lo=1, hi=1000)
    total = len(examples)
    resumed = _load_checkpoint(checkpoint_path)
    if resumed:
        sys.stderr.write(f"[june-bench] resuming: {len(resumed)}/{total} queries already recorded in "
                         f"{checkpoint_path} — skipping them (delete the file to start fresh).\n")
    _lock = try_lock_checkpoint(checkpoint_path)   # H3: warn on a concurrent run sharing this checkpoint

    async def _go() -> tuple[dict, dict]:
        rankings, golds = {}, {}
        errors = consec = done = 0
        try:
            # POOLED: ingest the union of all docs ONCE, then every query retrieves over the whole pool.
            # (On a resume the pool is re-ingested — only the per-query retrievals are skipped.)
            if getattr(system, "_pooled", False):
                n = await system.ingest_pool(examples)
                sys.stderr.write(f"[june-bench] pooled retrieval: ingested {n} docs into one corpus; "
                                 f"each of {len(examples)} queries retrieves over all of them.\n")
            for ex in examples:
                golds[ex.qid] = list(ex.meta.get("gold_ids", []) or [])
                if ex.qid in resumed:                 # already done on a previous run → skip the network
                    rankings[ex.qid] = resumed[ex.qid]
                    done += 1
                    if on_progress is not None:
                        try:
                            on_progress(done, total)
                        except Exception:  # noqa: BLE001
                            pass
                    continue
                try:
                    rankings[ex.qid] = await system.retrieve(ex)
                    consec = 0
                    _append_checkpoint(checkpoint_path, ex.qid, rankings[ex.qid])
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    consec += 1
                    rankings[ex.qid] = []
                    sys.stderr.write(f"\n[june-bench] june-retrieval errored on qid={ex.qid}: "
                                     f"{type(exc).__name__}: {exc} — scored a no-hit, continuing\n")
                    if consec >= max_consec:
                        resume_hint = (f" Progress is saved — just re-run the same command to resume "
                                       f"from query {done + 1}." if checkpoint_path else "")
                        raise RuntimeError(
                            f"aborting: {consec} consecutive retrieval errors — the endpoint is "
                            f"unreachable (server down, or the client lost the network). "
                            f"{len(rankings)}/{len(examples)} attempted; a partial run is NOT a valid "
                            f"result.{resume_hint}") from exc
                done += 1
                if on_progress is not None:
                    try:
                        on_progress(done, total)
                    except Exception:  # noqa: BLE001 — a progress-bar hiccup must never fail the run
                        pass
        finally:
            if getattr(system, "_pooled", False):
                await system.cleanup_pool()
            await system.aclose()
        if errors:
            sys.stderr.write(f"[june-bench] june-retrieval: {errors}/{len(examples)} queries errored "
                             f"(scored as no-hit) — the score under-states retrieval; fix the cause.\n")
        return rankings, golds
    try:
        return asyncio.run(_go())
    finally:
        if _lock is not None:
            _lock.close()               # release the H3 checkpoint lock


def from_env(*, k: int | None = None) -> JuneRetrievalSystem:
    url = os.environ.get("JUNE_BENCH_JUNE_URL", "")
    if not url:
        raise ValueError("set JUNE_BENCH_JUNE_URL to your June endpoint to use --system june-retrieval")
    _truthy = ("1", "true", "yes")
    backfill = os.environ.get("JUNE_BENCH_JUNE_BACKFILL", "").strip().lower() in _truthy
    pooled = os.environ.get("JUNE_BENCH_RETRIEVAL_POOL", "").strip().lower() in _truthy
    deep = os.environ.get("JUNE_BENCH_RETRIEVAL_DEEP", "").strip().lower() in _truthy
    kk = k if k is not None else env_int("JUNE_BENCH_RETRIEVAL_K", 10, lo=1, hi=1000)
    return JuneRetrievalSystem(url, api_key=os.environ.get("JUNE_BENCH_JUNE_KEY", ""),
                               k=kk, backfill=backfill, pooled=pooled, deep=deep,
                               timeout=env_float("JUNE_BENCH_JUNE_TIMEOUT", 120.0, lo=1.0))


__all__ = ["JuneRetrievalSystem", "run_retrieval", "from_env"]
