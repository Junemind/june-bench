"""june_bench runner — the fixed orchestration: ask the System each Example → Records (SB0).

Pure: depends only on the two ports, so it's unit-tested with a fake System + fixture Dataset
(no model, no network). If a System exposes ``ingest`` (a memory system), the runner calls it with
each example's ``corpus`` before asking — so context-QA and memory systems share one path.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import Sequence

from june_bench.ports import Dataset, Example, Record, System


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


async def _answer_one(system: System, ex: Example) -> Record:
    ingest = getattr(system, "ingest", None)
    if ingest is not None and ex.corpus:
        await _maybe_await(ingest(ex.corpus))
    pred = await system.answer(ex)
    m = pred.meta or {}
    return Record(
        qid=ex.qid, question=ex.question, golds=ex.golds, prediction=pred.text,
        context=ex.context, calls=int(m.get("calls", 0)), cost=float(m.get("cost", 0.0)),
        abstained=bool(m.get("abstained", False)), meta=dict(m))


def _error_record(ex: Example, exc: BaseException) -> Record:
    """An example a System couldn't answer (its ingest/answer raised) becomes an *abstained miss* with
    the error in ``meta`` — never a crashed run. Scored as no-answer (honest), but distinguishable from
    a real abstention via ``meta['system_error']`` so a config outage isn't silently read as '0.0 = bad'."""
    return Record(
        qid=ex.qid, question=ex.question, golds=ex.golds, prediction="",
        context=ex.context, calls=0, cost=0.0, abstained=True,
        meta={"error": f"{type(exc).__name__}: {exc}", "system_error": True})


def _fail_fast() -> bool:
    return os.environ.get("JUNE_BENCH_FAIL_FAST", "").strip().lower() in ("1", "true", "yes")


def _load_record_checkpoint(path: str | None) -> dict[str, Record]:
    """Load {qid: Record} from a JSONL checkpoint (one answered example per line; last wins).
    Missing / unreadable / a torn last line from a hard kill → best-effort. Never raises."""
    done: dict[str, Record] = {}
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
                    d = json.loads(line)
                    d["golds"] = tuple(d.get("golds", []))
                    d["context"] = tuple(d.get("context", []))
                    done[str(d["qid"])] = Record(**d)
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        return {}
    return done


def _append_record_checkpoint(path: str | None, rec: Record) -> None:
    """Append one answered example to the checkpoint (crash-safe append-only JSONL). Predictions cost
    money/time to produce, so saving them means a resumed run never re-pays for a completed question."""
    if not path:
        return
    import dataclasses
    import json
    import os as _os
    try:
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")
            fh.flush()
            _os.fsync(fh.fileno())     # durable: survive a kill/crash so resume never loses a paid answer
    except OSError:                     # a checkpoint write must never fail the run
        pass


async def run_async(system: System, examples: Sequence[Example], *,
                    fail_fast: bool | None = None, on_progress=None,  # noqa: ANN001
                    checkpoint_path: str | None = None) -> list[Record]:
    """Ask the system each example. **Fail-soft per example** (the default): if one example raises
    (a flaky provider, a rate limit, one system's outage in a multi-system suite), it is recorded as
    an errored miss and the run continues — so a failure on question N never loses questions 1..N-1,
    and one competitor's outage can't void the whole head-to-head. Set ``JUNE_BENCH_FAIL_FAST=1``
    (or pass ``fail_fast=True``) to re-raise instead, for debugging an adapter.

    ``checkpoint_path`` (optional): each answered example is appended to a JSONL file; on a re-run those
    qids are loaded and SKIPPED (no re-answer, no re-spend), so a dropped run resumes instead of
    restarting. Only successes are checkpointed — an errored example is retried on the next run."""
    ff = _fail_fast() if fail_fast is None else fail_fast
    resumed = _load_record_checkpoint(checkpoint_path)
    if resumed:
        sys.stderr.write(f"[june-bench] resuming: {len(resumed)}/{len(examples)} examples already "
                         f"answered in {checkpoint_path} — skipping them (delete the file to start "
                         f"fresh).\n")
    # Fail-soft is right for ONE flaky question, but WRONG when the whole endpoint dies (server crash /
    # OOM): then EVERY remaining call fails and we'd score a cascade of phantom misses → a misleading
    # "completed" run. So a run of CONSECUTIVE errors means the system is DOWN — abort honestly.
    from june_bench._util import env_int, try_lock_checkpoint
    max_consec = env_int("JUNE_BENCH_MAX_CONSEC_ERRORS", 8, lo=1, hi=1000)
    _lock = try_lock_checkpoint(checkpoint_path)   # H3: warn on a concurrent run sharing this checkpoint
    out: list[Record] = []
    errors = 0
    consec = 0
    total = len(examples)
    try:
        # POOLED (open-pool QA): a System that builds a SHARED corpus (JuneApiSystem with pooled=True)
        # ingests the UNION of all examples' evidence ONCE, before any question — so each question
        # retrieves its gold out of the whole pool. Additive + guarded; non-pooled systems are unaffected.
        pool_setup = getattr(system, "ingest_pool", None)
        if pool_setup is not None and getattr(system, "_pooled", False):
            # PRE-RUN READINESS GATE: don't start the heavy ingest onto an already-saturated endpoint (the
            # 'box already hot' case) — wait for capacity first so we never dive into a big write that would
            # fail after retries. Best-effort/optional; bounded + fail-open (never hangs, never blocks).
            pool_ready = getattr(system, "await_ready", None)
            if pool_ready is not None:
                try:
                    await _maybe_await(pool_ready())
                except Exception:  # noqa: BLE001 — a readiness hiccup must never block the run
                    pass
            # PRE-RUN SWEEP: clear orphaned pool canvases from earlier (hard-killed / old-build) runs before
            # ingesting, so leftovers can't bloat the endpoint's SQLite into 'database is locked'. Best-effort
            # and optional — systems without the hook (Cognee, oracles) are skipped; a failure never blocks.
            pool_sweep = getattr(system, "sweep_stale_pools", None)
            if pool_sweep is not None:
                try:
                    await _maybe_await(pool_sweep())
                except Exception:  # noqa: BLE001 — a sweep hiccup must never block the real run
                    pass
            n_pool = await _maybe_await(pool_setup(examples))
            sys.stderr.write(f"[june-bench] pooled QA: ingested {n_pool} deduped docs into one shared "
                             f"corpus; each of {len(examples)} questions answers over the whole pool.\n")
        for ex in examples:
            if ex.qid in resumed:                       # already answered on a previous run → skip
                out.append(resumed[ex.qid])
                # A checkpoint skip is NOT a live success — do NOT reset `consec`, or a dead endpoint
                # interspersed with resumed hits would never trip the cascade guard (audit H7).
                if on_progress is not None:
                    on_progress(len(out), total)
                continue
            try:
                rec = await _answer_one(system, ex)
                out.append(rec)
                consec = 0                              # a live answer means the endpoint is alive
                _append_record_checkpoint(checkpoint_path, rec)
                if on_progress is not None:             # live progress for the `reproduce` UX
                    on_progress(len(out), total)
            except Exception as exc:  # noqa: BLE001 — isolate one example; honesty preserved via meta+warn
                if ff:
                    raise
                errors += 1
                consec += 1
                if on_progress is not None:
                    on_progress(len(out) + 1, total)
                sys.stderr.write(
                    f"[june-bench] {getattr(system, 'name', '?')} errored on qid={ex.qid}: "
                    f"{type(exc).__name__}: {exc} — scoring it a miss and continuing "
                    f"(set JUNE_BENCH_FAIL_FAST=1 to stop on first error)\n")
                out.append(_error_record(ex, exc))
                if consec >= max_consec:
                    resume_hint = (" Progress is saved — re-run the same command to resume (answered "
                                   "questions are skipped, not re-paid)." if checkpoint_path else "")
                    raise RuntimeError(
                        f"aborting: {consec} consecutive errors from "
                        f"'{getattr(system, 'name', '?')}' — the endpoint is unreachable (server down / "
                        f"OOM / connection refused, or the client lost the network). {len(out)}/"
                        f"{len(examples)} attempted; a partial run is NOT a valid result and must not be "
                        f"reported.{resume_hint} Tune with JUNE_BENCH_MAX_CONSEC_ERRORS.") from exc
        if errors:
            sys.stderr.write(
                f"[june-bench] {getattr(system, 'name', '?')}: {errors}/{len(examples)} examples errored "
                f"(scored as misses). A non-zero error count means the score UNDER-states the system — "
                f"fix the cause (e.g. embedder/key) before trusting it as a real result.\n")
        return out
    finally:
        if _lock is not None:
            _lock.close()                               # H3: release the checkpoint lock
        # Delete the run's shared open-pool canvas so a benchmark never leaves data in the endpoint DB
        # (leftover pools bloat the single-writer SQLite → 'database is locked' contention on later runs).
        # BEFORE aclose (the client must still be alive to send the DELETE); best-effort on every exit
        # path — success, per-example error, or cascade-abort. Systems without a pool (Cognee, oracles) or
        # a shared corpus simply don't expose `cleanup_pool` and are skipped.
        pool_cleanup = getattr(system, "cleanup_pool", None)
        if pool_cleanup is not None:
            try:
                await _maybe_await(pool_cleanup())
            except Exception:  # noqa: BLE001 — cleanup must never mask the real result or error
                pass
        closer = getattr(system, "aclose", None)
        if closer is not None:
            try:
                await _maybe_await(closer())            # C1: release the HTTP client/pool on every path
            except Exception:  # noqa: BLE001 — cleanup must never mask the real result or error
                pass


def run(system: System, dataset: Dataset, *, split: str = "smoke",
        limit: int | None = None, on_progress=None,  # noqa: ANN001
        checkpoint_path: str | None = None) -> list[Record]:
    """Load the split, ask the system each example, return scored-ready Records. Sync entry point
    (wraps the async path) so the CLI and tests call one function. ``on_progress(done, total)`` is
    an optional callback for a live progress display (used by the `reproduce` command).
    ``checkpoint_path`` (optional) makes the run resumable — see :func:`run_async`."""
    examples = list(dataset.load(split, limit=limit))   # limit-aware: lets big June splits stream
    if limit is not None:
        examples = examples[:limit]
    return asyncio.run(run_async(system, examples, on_progress=on_progress,
                                 checkpoint_path=checkpoint_path))


__all__ = ["run", "run_async"]
