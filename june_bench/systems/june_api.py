"""JuneApiSystem (SB3) — the DEFAULT June adapter: a thin HTTP client to June's REST API.

This is the source-protection boundary (decided 2026-06-22): the published suite reaches June **only
over HTTP** (`POST /v1/answer`), so it ships *no* `june_ai` source — just the request/response shapes.
A benchmarker points it at any June endpoint with a base URL (+ optional key).

For memory benchmarks (LoCoMo/LongMemEval/FinanceBench) the runner calls `ingest(corpus)` first
(`POST /v1/ingest/text` per doc) so June has the conversation in its graph before the question; for
context-QA the question is asked directly. The model/server is external — this adapter is pure plumbing
and is unit-tested against an `httpx.MockTransport` (no server). `httpx` is the only dependency, pulled
by the `[june-api]` extra and imported lazily so the base package needs nothing.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

from june_bench._util import env_float
from june_bench.ports import Example, Prediction


class _FallbackToText(Exception):
    """Internal signal: the endpoint lacks /v1/ingest/docs → fall back to the per-doc /v1/ingest/text path."""


# Every open-pool run's shared canvas is created under this exact name, so a pre-run sweep can find and
# delete leftovers from earlier runs by name (see `sweep_stale_pools`). Single source of truth: the create
# call and the sweep filter both use this, so they can never drift.
_POOL_CANVAS_NAME = "bench-pool"

# An LLM synthesizer signals "can't answer" with a sentinel phrase (the answer_llm prompt says reply
# exactly "I don't know"), whereas the extractive floor signals it with a `degraded` abstain marker.
# Detecting both keeps abstention measured consistently across June's two synthesizers — otherwise an
# LLM "I don't know" is mis-scored as an answered-but-wrong item (deflating accuracy, inflating coverage).
_ABSTAIN_SENTINELS = frozenset({
    "i don't know", "i dont know", "i do not know", "unknown", "no answer",
    "not enough information", "insufficient evidence", "i cannot answer", "i can't answer",
})


def _looks_abstained(text: str) -> bool:
    norm = text.strip().lower().rstrip(".!").strip()
    return (not norm) or norm in _ABSTAIN_SENTINELS


class JuneApiSystem:
    """HTTP client to June's `/v1/answer` (+ `/v1/ingest/text`). ``transport`` lets a test inject an
    `httpx.MockTransport`; ``params`` are extra `AnswerIn` overrides (e.g. `{"mode": "llm_augmented"}`)."""
    name = "june-api"

    def __init__(self, base_url: str, *, api_key: str = "", answer_path: str = "/v1/answer",
                 ingest_path: str = "/v1/ingest/text", backfill_path: str = "/v1/embeddings/backfill",
                 docs_path: str = "/v1/ingest/docs",
                 canvas_path: str = "/v1/canvases", timeout: float = 30.0, transport=None,
                 params: dict | None = None, isolate: bool = True, backfill: bool = False,
                 cleanup: bool = False, llm_key: str = "", llm_model: str = "", record: bool = False,
                 pooled: bool = False, llm_platform: str = "") -> None:  # noqa: ANN001
        if not base_url:
            raise ValueError("JuneApiSystem needs a base_url (e.g. http://localhost:8000). "
                             "Set JUNE_BENCH_JUNE_URL or pass base_url=…")
        import httpx  # lazy — only the [june-api] extra needs it
        headers = {"content-type": "application/json"}
        # June's service authenticates on the `X-API-Key` header (june_service.auth / app.get_caller),
        # not `Authorization: Bearer`. Speak the service's documented contract so the published online
        # client authenticates against any June endpoint instead of 401-ing.
        if api_key:
            headers["X-API-Key"] = api_key
        # BYO-key: send the caller's own LLM key so the June endpoint synthesizes on the CALLER's dime
        # (the host never pays / never holds the key). Sent on every request as the X-LLM-Key header.
        if llm_key:
            headers["X-LLM-Key"] = llm_key
        # BYO-model (fully-open): pick the answer/reasoning model per request. The endpoint routes it to
        # BOTH the synthesizer and the reasoner, so the caller reproduces ANY model's number end-to-end.
        # Empty ⇒ the endpoint's default model.
        if llm_model:
            headers["X-LLM-Model"] = llm_model
        # BYO-platform: an allowlisted ENUM the endpoint maps to a serving URL (never a URL from
        # here — SSRF). "openrouter" is the endpoint default, so it is NOT sent: old endpoints that
        # predate the header then behave byte-identically, and the run_reproduce guard refuses
        # non-default platforms against endpoints that don't advertise support.
        if llm_platform and llm_platform.strip().lower() != "openrouter":
            headers["X-LLM-Platform"] = llm_platform.strip().lower()
        self._answer_path = answer_path
        self._ingest_path = ingest_path
        self._docs_path = docs_path
        self._backfill_path = backfill_path
        self._canvas_path = canvas_path
        self._params = dict(params or {})
        # isolate=True → each question runs in its own June canvas (no cross-question leakage)
        self._isolate = isolate
        # backfill=True → embed freshly-ingested content so June's DENSE lane has vectors. Required to
        # engage the semantic lane (else the endpoint answers on its lexical/graph floor only).
        self._backfill = backfill
        # cleanup=True → DELETE the question's canvas after answering (Q5: stop per-question workspaces
        # accumulating in the registry/DB over a full run). Best-effort; needs isolate=True to apply.
        self._cleanup = cleanup
        # record=True → RECORD MODE: don't ingest into June's graph; send the example's passages
        # inline on /v1/answer (the endpoint answers over them directly). Isolates the answerer +
        # reasoning layer — June's published-headline setting — vs the default ingest→retrieve path.
        self._record = record
        # pooled=True → OPEN-POOL QA: ingest the deduped UNION of ALL examples' evidence ONCE into a
        # single shared canvas (via `ingest_pool`, called by the runner before the loop), then each
        # question answers over that whole pool — so June's retrieval lanes (BM25 + dense + fusion)
        # must FIND the gold, exactly like the local `run.py --retrieval pool`. Contrast: record hands
        # passages inline (no retrieval); default per-question ingest scopes evidence to one question.
        self._pooled = pooled
        self._pool_canvas: str = ""
        self._pool_ready = False
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                         transport=transport, headers=headers)

    async def _create_canvas(self, name: str) -> str:
        """Create a fresh isolated canvas and return its server-assigned ``canvas_id``.

        June's `X-Canvas` is a server-issued canvas **id** (a UUID), resolved fail-closed against
        the canvas registry — an arbitrary name is *not* a valid handle and would 404. So a client
        that wants per-question isolation must POST `/v1/canvases` first and use the returned id as
        the `X-Canvas` value (the documented create→use→delete lifecycle). The create call itself
        carries no `X-Canvas` (it runs in the caller's home workspace). Retried on a transient blip
        (a busy endpoint's 5xx / dropped keep-alive) — a duplicate empty canvas is harmless."""
        from june_bench._util import env_int, retry_request
        r = await retry_request(
            lambda: self._client.post(self._canvas_path, json={"name": name}),
            attempts=env_int("JUNE_BENCH_CANVAS_RETRIES", 4, lo=1, hi=50), base=1.5, cap=6.0)
        r.raise_for_status()
        return str(r.json()["canvas_id"])

    async def _ingest_docs(self, docs: Sequence[str], headers: dict) -> int:
        # Resilient like the retrieval path: a busy single-writer endpoint throws a transient 5xx /
        # 'database is locked' on a contended write; retry (jittered) instead of failing the question.
        # /v1/ingest/text has no stable doc id, so a retry after a *post-commit* failure could duplicate
        # the passage — we send a content-derived Idempotency-Key so an endpoint that honours it dedupes
        # (harmless if it doesn't; the QA pool is also text-deduped upstream). Tunable via
        # JUNE_BENCH_INGEST_RETRIES (shared with june_retrieval).
        import hashlib
        import sys as _sys

        from june_bench._util import env_int, retry_request
        attempts = env_int("JUNE_BENCH_INGEST_RETRIES", 8, lo=1, hi=100)
        n = 0
        for doc in docs:
            text = str(doc)
            if not text.strip():
                continue
            idem = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
            hdr = {**headers, "Idempotency-Key": idem}
            payload = {"text": text, "format": "text", "source_app": "june-bench"}
            r = await retry_request(
                lambda h=hdr, p=payload: self._client.post(self._ingest_path, json=p, headers=h),
                attempts=attempts,
                on_first_retry=lambda: _sys.stderr.write(
                    "[june-bench] ingest hit a transient server error (likely a DB lock) — retrying…\n"))
            r.raise_for_status()                 # a non-5xx that's still an error (4xx) → surface it
            n += 1
        return n

    async def _ingest_pool_docs(self, docs: Sequence[str], headers: dict, batches: int) -> int:
        """Ingest the shared pool in a FEW BATCHED writes via ``/v1/ingest/docs`` (many docs per request =
        ONE write transaction each) instead of one write per doc. A big pool (~1.6k passages at n=100) sent
        one-at-a-time is ~1.6k separate write transactions that lock-storm the endpoint's single-writer
        SQLite ("database is locked" → 500); batching collapses that to ``batches`` transactions.

        **Output-neutral:** all docs land in the SAME pool canvas, so every question still retrieves over
        the WHOLE union — only the number of upload transactions changes, not the pool or the answers.
        Id-preserving + idempotent (uuid5 over a content-hash id → upsert), so a retried batch never
        duplicates. Fail-soft: if the batched endpoint is unavailable (older server), fall back to the
        per-doc path so the run still works."""
        import hashlib
        import math
        import sys as _sys

        from june_bench._util import env_int, retry_request
        items = [{"id": hashlib.sha256(str(d).encode("utf-8")).hexdigest()[:32], "text": str(d)}
                 for d in docs if str(d).strip()]
        if not items:
            return 0
        n_batches = max(1, min(batches, len(items)))
        size = math.ceil(len(items) / n_batches)
        attempts = env_int("JUNE_BENCH_INGEST_RETRIES", 8, lo=1, hi=100)
        _sys.stderr.write(f"[june-bench] pool ingest: {len(items)} passages in {n_batches} batched "
                          f"write(s) (≈{size}/batch) — avoids the SQLite lock storm.\n")
        sent = 0
        try:
            for i in range(0, len(items), size):
                chunk = items[i:i + size]
                r = await retry_request(
                    lambda p={"docs": chunk}: self._client.post(self._docs_path, json=p, headers=headers),
                    attempts=attempts,
                    on_first_retry=lambda: _sys.stderr.write(
                        "[june-bench] batched ingest hit a transient server error — retrying…\n"))
                if r.status_code == 404:                 # server without /v1/ingest/docs → fall back
                    raise _FallbackToText
                r.raise_for_status()
                sent += len(chunk)
            return sent
        except _FallbackToText:
            _sys.stderr.write("[june-bench] endpoint has no /v1/ingest/docs — falling back to per-doc "
                              "ingest (slower; may lock on a large pool).\n")
            return await self._ingest_docs([it["text"] for it in items], headers)

    async def _maybe_backfill(self, headers: dict) -> dict:
        """Embed the just-ingested content (within this question's canvas) so the dense lane has
        vectors — LOOPING until the endpoint reports ``done`` (Defect A, found 2026-07-26).

        This used to be a single bodiless POST. When the backfill route later grew a ``max_nodes``
        body (default 256), that one call silently embedded 256 of a 991-doc pool — dense and FTS
        then ran over ~26% of the corpus with no error anywhere, and the model honestly refused the
        rest. The desktop app already looped (main.js: ``{max_nodes:128}`` until ``r.done``); only
        this client fired once. Endpoints predating the field ignore the body and return no
        ``done`` → exactly one call, byte-identical to the old behaviour.

        Fail-soft on transport errors and 4xx (a model-free endpoint has no embedder — expected;
        the answer proceeds on graph+lexical). But a pool that will not CONVERGE is reported
        loudly: silent partial coverage is precisely how this bug hid for a day."""
        acc: dict = {"embedded": 0, "last": {}}
        if not self._backfill:
            return acc
        try:
            for _ in range(64):        # 64×1000 nodes ≫ any bench pool — a runaway bound, not a budget
                r = await self._client.post(self._backfill_path, json={"max_nodes": 1000},
                                            headers=headers)
                if r.status_code >= 400:               # 'no embedder configured' → dense lane not wired
                    return acc
                body = r.json() if r.content else {}
                acc["last"] = body
                try:
                    acc["embedded"] += int(body.get("embedded", 0) or 0)
                except (TypeError, ValueError):
                    pass
                if body.get("done", True):             # old route has no `done` → single-shot, complete
                    return acc
            import sys as _sys
            _sys.stderr.write("[june-bench] WARNING: embeddings backfill did not converge after 64 "
                              "rounds — the dense lane may cover only part of the pool.\n")
        except Exception:  # noqa: BLE001 — best-effort; the answer still proceeds on graph+lexical
            pass
        return acc

    async def ingest_pool(self, examples: Sequence["Example"]) -> int:  # noqa: F821
        """OPEN-POOL setup (the local `--retrieval pool` analogue): ingest the deduped UNION of every
        example's evidence (``context`` passages + ``corpus`` docs) ONCE into a single shared canvas,
        then backfill embeddings once so June's dense lane has vectors over the WHOLE pool. Every
        question later answers over this pool — June must RETRIEVE the gold, its retrieval lanes are
        exercised (recall < 1.0), not handed the answer. Idempotent via ``_pool_ready``. Dedup by
        normalized text (QA passages carry no stable id). Returns the number of pooled docs."""
        if self._pool_ready:
            return 0
        seen: dict[str, str] = {}
        for ex in examples:
            for passage in (*ex.context, *ex.corpus):
                t = str(passage).strip()
                if t and t not in seen:
                    seen[t] = t
        docs = list(seen.values())
        canvas = await self._create_canvas(_POOL_CANVAS_NAME) if self._isolate else ""
        # Register the handle the MOMENT the canvas exists — before the ingest/backfill that can 500 — so
        # `cleanup_pool` (runner's finally) always deletes it. Registering only after a successful ingest
        # leaked the canvas on every failed run, and leftover pools are what bloat the endpoint's SQLite.
        self._pool_canvas = canvas
        headers = {"X-Canvas": canvas} if canvas else {}
        # BATCHED upload of the shared pool → avoids the 'database is locked' storm a big per-doc ingest
        # causes on the endpoint's single-writer SQLite. Batch count: JUNE_BENCH_POOL_INGEST_BATCHES if the
        # user chose one (interactive prompt), else default ~500 docs/batch. Output-neutral (same pool).
        from june_bench._util import env_int
        batches = env_int("JUNE_BENCH_POOL_INGEST_BATCHES", 0, lo=0, hi=10000)
        if batches <= 0:
            import math
            batches = max(1, math.ceil(len(docs) / 500))
        n = await self._ingest_pool_docs(docs, headers, batches)
        bf = await self._maybe_backfill(headers)       # embed the whole pool → dense lane has vectors
        # INGEST VERIFICATION (July 2026): the client's "N passages" print is a claim about what it
        # SENT, not what LANDED. Runs on a bloated endpoint silently landed ~680 of 991 (fossil
        # record in the residue DB) and nothing anywhere said so. The backfill counters are a
        # server-side census of the same freshly-created canvas: fully embedded ⇒ embedded ≈ docs.
        # A shortfall means the questions will run over a pool the harness did not upload — say so
        # loudly BEFORE money is spent answering over it. (Endpoints without counters report 0 →
        # check skipped, behaviour unchanged.)
        server_n = int(bf.get("embedded") or 0)
        if server_n and n and server_n < n * 0.98:
            import sys as _sys
            _sys.stderr.write(
                f"[june-bench] ⚠ SILENT PARTIAL INGEST: sent {n} pool docs but the endpoint "
                f"embedded only {server_n} — retrieval will run over an incomplete pool and "
                f"refusals/misses will NOT be June's fault. Wipe/compact the endpoint DB and "
                f"re-run before trusting this run's numbers.\n")
        self._pool_ready = True                        # (_pool_canvas already registered above)
        return n

    async def _maybe_cleanup(self, canvas: str, headers: dict) -> None:
        """Q5: delete the question's canvas after answering so workspaces don't accumulate over a run.
        Best-effort — a failure here never affects the (already-returned) answer."""
        if not (self._cleanup and canvas):
            return
        try:
            await self._client.delete(f"{self._canvas_path}/{canvas}", headers=headers)
        except Exception:  # noqa: BLE001
            pass

    async def _probe_latency(self, samples: int = 3):  # noqa: ANN201 — median seconds, or None
        """Round-trip latency of a lightweight, DB-touching health probe (`GET /v1/canvases/health`, which
        does a small registry read) as a proxy for endpoint capacity: if a trivial call is slow, the box's
        CPU is contended and our writes will be too. Returns the MEDIAN of a few samples (robust to a single
        blip), or ``None`` if the probe can't be measured (unreachable / 5xx) → the caller fails OPEN."""
        import statistics
        import time
        got: list[float] = []
        for _ in range(max(1, samples)):
            t0 = time.monotonic()
            try:
                r = await self._client.get(f"{self._canvas_path}/health")
            except Exception:  # noqa: BLE001 — unreachable → can't measure → fail open
                return None
            if r.status_code >= 500:                     # server erroring → can't tell → fail open
                return None
            got.append(time.monotonic() - t0)
        return statistics.median(got) if got else None

    async def await_ready(self) -> None:
        """PRE-RUN READINESS GATE: before the heavy pool ingest, make sure the endpoint has capacity, so we
        never start a big write onto an already-saturated box and fail after retries (the 'box already hot'
        case — e.g. a prior run's embed pass still churning). Signal is `_probe_latency`. If it's over the
        threshold, print a notice and POLL until it clears or `max_wait` elapses, then proceed regardless
        (batching + the write-retry path absorb residual slowness). Never hangs (bounded by max_wait) and
        never hard-fails (fail-OPEN: an unmeasurable probe or a still-busy box both proceed). Fully env-
        tunable; `JUNE_BENCH_READY_GATE=0` disables it. A future direct load endpoint could replace the
        latency proxy without changing this control flow."""
        import asyncio
        import os
        import sys as _sys
        if os.environ.get("JUNE_BENCH_READY_GATE", "1").strip() == "0":
            return
        thresh = env_float("JUNE_BENCH_READY_PROBE_S", 1.5, lo=0.1, hi=60.0)
        max_wait = env_float("JUNE_BENCH_READY_MAX_WAIT_S", 120.0, lo=0.0, hi=3600.0)
        interval = env_float("JUNE_BENCH_READY_INTERVAL_S", 5.0, lo=1.0, hi=300.0)

        lat = await self._probe_latency()
        if lat is None or lat <= thresh:                 # can't measure (fail open) or healthy → go now
            return
        _sys.stderr.write(f"[june-bench] endpoint busy (health probe {lat:.1f}s > {thresh:.1f}s) — waiting "
                          f"up to {int(max_wait)}s for load to settle before ingest…\n")
        waited = 0.0
        while waited < max_wait:
            await asyncio.sleep(interval)
            waited += interval
            lat = await self._probe_latency()
            if lat is None or lat <= thresh:
                tail = "" if lat is None else f"({lat:.1f}s) "
                _sys.stderr.write(f"[june-bench] endpoint ready {tail}— continuing after {int(waited)}s.\n")
                return
        _sys.stderr.write(f"[june-bench] endpoint still busy after {int(max_wait)}s — proceeding anyway "
                          f"(batching + write-retries will absorb slow writes).\n")

    async def sweep_stale_pools(self) -> int:
        """PRE-RUN self-heal: delete leftover `bench-pool` canvases from earlier runs before this one starts.

        `cleanup_pool` tears down the current run's canvas on exit, but a run that was hard-killed (SIGKILL,
        power loss) or ingested against an OLD build that leaked on failure can leave an orphan behind — and
        orphaned pools accumulate in the endpoint's single-writer SQLite, contending later writes into
        'database is locked'. This sweeps them so a fresh run starts against a clean DB, with no manual DB
        reset needed. `GET /v1/canvases` is user-fenced (returns only THIS access key's canvases), and we
        delete ONLY those named exactly `_POOL_CANVAS_NAME` — a user's real canvases are never touched.
        Fail-soft + best-effort: an older endpoint without the list route (404), a transport blip, or a
        per-delete error is ignored and the run proceeds. Returns the number of stale pools swept."""
        if not self._isolate:
            return 0
        try:
            r = await self._client.get(self._canvas_path)
            if r.status_code != 200:
                return 0
            items = r.json()
        except Exception:  # noqa: BLE001 — no list route / transport blip / bad JSON → nothing to sweep
            return 0
        if not isinstance(items, list):
            return 0
        swept = 0
        for it in items:
            if not (isinstance(it, dict) and it.get("name") == _POOL_CANVAS_NAME):
                continue
            cid = str(it.get("canvas_id") or "").strip()
            if not cid:
                continue
            try:
                await self._client.delete(f"{self._canvas_path}/{cid}")
                swept += 1
            except Exception:  # noqa: BLE001 — one stuck delete must not block the run
                pass
        if swept:
            import sys as _sys
            _sys.stderr.write(f"[june-bench] swept {swept} leftover pool canvas(es) from earlier runs "
                              f"(kept the endpoint DB clean).\n")
        return swept

    async def cleanup_pool(self) -> None:
        """DELETE this run's shared open-pool canvas so a benchmark run leaves nothing behind in the
        endpoint DB (the runner calls this in its ``finally`` — success, error, or abort). Unconditional
        (not gated on ``_cleanup``): a pooled benchmark canvas is inherently ephemeral, and leftovers
        bloat the endpoint's single-writer SQLite → the 'database is locked' contention on later runs.
        Best-effort + idempotent — clears the handle first so it never double-deletes, and a failure is
        harmless (a leftover canvas is dead weight, never a correctness issue: isolation means no run ever
        reads another run's canvas)."""
        canvas, self._pool_canvas = self._pool_canvas, ""
        if not canvas:
            return
        try:
            await self._client.delete(f"{self._canvas_path}/{canvas}",
                                      headers={"X-Canvas": canvas})
        except Exception:  # noqa: BLE001
            pass

    async def answer(self, example: Example) -> Prediction:
        """Run one example end-to-end against June, isolated to its own canvas.

        June answers over its *graph*, so the evidence must be in it first. We feed June BOTH
        modalities into a per-question **canvas** (an isolated workspace, via the ``X-Canvas``
        header): the in-prompt ``context`` passages (HotpotQA/2Wiki/MuSiQue) and the ``corpus``
        documents a memory system ingests (LoCoMo/LongMemEval). The canvas is what keeps one
        question's evidence from leaking into the next. (Intentionally no public ``ingest``
        method: the runner would call it *outside* the canvas — doing it here keeps ingest and
        answer on the same isolated scope.)"""
        # RECORD MODE: skip ingest + the graph entirely — hand June the passages inline on /v1/answer
        # (isolates answerer+reasoning, the published-headline setting). No canvas/backfill/cleanup.
        if self._record:
            passages = [str(p) for p in (*example.context, *example.corpus) if str(p).strip()]
            payload = {"query": example.question, "passages": passages, **self._params}
            r = await self._client.post(self._answer_path, json=payload)
            r.raise_for_status()
            data = r.json()
            text = str(data.get("answer", ""))
            degraded = list(data.get("degraded", []) or [])
            abstained = _looks_abstained(text) or any(str(m).startswith("abstain") for m in degraded)
            return Prediction(text=text, meta={"degraded": degraded, "mode": data.get("mode", ""),
                                               "calls": 1, "ingest_calls": 0, "record": True,
                                               "abstained": abstained})
        # POOLED MODE: the shared pool was ingested ONCE by `ingest_pool` (runner hook). Answer over it
        # WITHOUT any per-question ingest/canvas/cleanup — June retrieves the gold from the whole pool.
        if self._pooled:
            headers = {"X-Canvas": self._pool_canvas} if self._pool_canvas else {}
            payload = {"query": example.question, **self._params}
            r = await self._client.post(self._answer_path, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            text = str(data.get("answer", ""))
            degraded = list(data.get("degraded", []) or [])
            abstained = _looks_abstained(text) or any(str(m).startswith("abstain") for m in degraded)
            return Prediction(text=text, meta={"degraded": degraded, "mode": data.get("mode", ""),
                                               "calls": 1, "ingest_calls": 0, "pooled": True,
                                               "backfilled": bool(self._backfill),
                                               "abstained": abstained, "canvas": self._pool_canvas})
        canvas = await self._create_canvas(f"bench-{example.qid}") if self._isolate else ""
        headers = {"X-Canvas": canvas} if canvas else {}
        ingest_calls = await self._ingest_docs(example.corpus, headers)
        ingest_calls += await self._ingest_docs(example.context, headers)
        await self._maybe_backfill(headers)        # embed → dense lane has vectors (if configured)
        payload = {"query": example.question, **self._params}
        r = await self._client.post(self._answer_path, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        await self._maybe_cleanup(canvas, headers)  # Q5: drop the per-question workspace
        text = str(data.get("answer", ""))
        degraded = list(data.get("degraded", []) or [])
        abstained = _looks_abstained(text) or any(str(m).startswith("abstain") for m in degraded)
        # `calls` = the answer-LLM call (the cost axis). `ingest_calls` records ingest-side work
        # separately (Q4) so cost stays transparent rather than apples-to-oranges with cognee.
        return Prediction(text=text, meta={"degraded": degraded, "mode": data.get("mode", ""),
                                           "calls": 1, "ingest_calls": ingest_calls,
                                           "backfilled": bool(self._backfill),
                                           "abstained": abstained, "canvas": canvas})

    async def aclose(self) -> None:
        await self._client.aclose()


def from_env() -> JuneApiSystem:
    """Build a JuneApiSystem from `JUNE_BENCH_JUNE_URL` (+ `JUNE_BENCH_JUNE_KEY`). The registry's
    zero-arg factory; raises a clear error if the endpoint isn't configured."""
    url = os.environ.get("JUNE_BENCH_JUNE_URL", "")
    if not url:
        raise ValueError("set JUNE_BENCH_JUNE_URL to your June endpoint "
                         "(e.g. http://localhost:8000) to use --system june-api")
    _truthy = ("1", "true", "yes")
    backfill = os.environ.get("JUNE_BENCH_JUNE_BACKFILL", "").strip().lower() in _truthy
    cleanup = os.environ.get("JUNE_BENCH_JUNE_CLEANUP", "").strip().lower() in _truthy
    # JUNE_BENCH_JUNE_RECORD=1 → record mode: answer over supplied passages, no ingest/retrieval.
    record = os.environ.get("JUNE_BENCH_JUNE_RECORD", "").strip().lower() in _truthy
    # JUNE_BENCH_JUNE_POOL=1 → OPEN-POOL QA: ingest the deduped union of ALL items' evidence once, then
    # every question RETRIEVES its gold out of the whole pool (the local `run.py --retrieval pool`).
    # Mutually exclusive with record (pool retrieves; record hands passages in) — pool wins if both set.
    pool = os.environ.get("JUNE_BENCH_JUNE_POOL", "").strip().lower() in _truthy
    if pool:
        record = False
    # JUNE_BENCH_JUNE_MULTIHOP=1 → ask the endpoint to decompose→per-subquery→merge (Apex A3, the
    # PlannedGraphAnswer path behind June's multi-hop/open-pool number). Carried as an AnswerIn
    # override so it rides through BOTH record and ingest payloads. NOTE: the route short-circuits to
    # the reasoner when the SERVER has JUNE_REASON set — to exercise multihop, run the server with
    # reason OFF (the `multihop` mode in scripts/bench_june.sh does exactly this).
    params: dict = {}
    if os.environ.get("JUNE_BENCH_JUNE_MULTIHOP", "").strip().lower() in _truthy:
        params["multihop"] = True
        _subq = os.environ.get("JUNE_BENCH_JUNE_MAX_SUBQ", "").strip()
        if _subq.isdigit():
            params["max_subqueries"] = max(1, min(8, int(_subq)))
    # Memory datasets ingest up to 200 docs/question + backfill embeddings + an LLM answer, so 30s is
    # too tight; default 120s and let JUNE_BENCH_JUNE_TIMEOUT override.
    timeout = env_float("JUNE_BENCH_JUNE_TIMEOUT", 120.0, lo=1.0)
    # BYO-key: your own LLM key for June's answer synthesis (so you pay, not the endpoint host).
    # Falls back to OPENROUTER_API_KEY (the same key the bench/cognee use) for convenience.
    llm_key = os.environ.get("JUNE_BENCH_LLM_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
    # BYO-model: the answer/reasoning model the endpoint should use for THIS run (X-LLM-Model). Empty ⇒
    # the endpoint's default. Lets a benchmarker reproduce any model's number against one endpoint.
    llm_model = os.environ.get("JUNE_BENCH_LLM_MODEL", "")
    # BYO-platform enum (openrouter/openai/anthropic/google) — see __init__; default = endpoint's.
    llm_platform = os.environ.get("JUNE_BENCH_LLM_PLATFORM", "")
    return JuneApiSystem(url, api_key=os.environ.get("JUNE_BENCH_JUNE_KEY", ""),
                         backfill=backfill, cleanup=cleanup, timeout=timeout, llm_key=llm_key,
                         llm_model=llm_model, llm_platform=llm_platform,
                         record=record, pooled=pool, params=params or None)


__all__ = ["JuneApiSystem", "from_env"]
