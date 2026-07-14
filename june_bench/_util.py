"""Shared low-level utilities: hardened env parsing + one async retry policy.

Extracted so the whole harness uses **one** retry curve (jittered, to avoid thundering-herd on a shared
endpoint) and **one** crash-proof env parse — instead of the five hand-rolled copies the audit found
(H5/H6). Pure stdlib; ``httpx`` is imported lazily inside the retry helper so the base package needs
nothing.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable, Iterable


def env_int(name: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    """Parse an int env var, NEVER raising: a bad value warns and falls back to ``default``; the result
    is clamped to ``[lo, hi]``. Replaces naked ``int(os.environ[...])`` that crashed a run on a typo."""
    raw = os.environ.get(name)
    val = default
    if raw is not None and raw.strip():
        try:
            val = int(raw)
        except (TypeError, ValueError):
            sys.stderr.write(f"[june-bench] {name}={raw!r} is not an integer — using {default}.\n")
            val = default
    if lo is not None and val < lo:
        val = lo
    if hi is not None and val > hi:
        val = hi
    return val


def env_float(name: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    """Parse a float env var, NEVER raising (see :func:`env_int`)."""
    raw = os.environ.get(name)
    val = default
    if raw is not None and raw.strip():
        try:
            val = float(raw)
        except (TypeError, ValueError):
            sys.stderr.write(f"[june-bench] {name}={raw!r} is not a number — using {default}.\n")
            val = default
    if lo is not None and val < lo:
        val = lo
    if hi is not None and val > hi:
        val = hi
    return val


def env_flag(name: str) -> bool:
    """True iff the env var is set to a truthy token (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


async def retry_request(
    send: Callable[[], Awaitable],  # noqa: ANN401 — returns an httpx.Response
    *,
    attempts: int,
    retryable_statuses: Iterable[int] = (),
    base: float = 2.0,
    cap: float = 10.0,
    on_first_retry: Callable[[], None] | None = None,
):  # noqa: ANN201 — returns an httpx.Response
    """Call ``send()`` (an idempotent coroutine returning an ``httpx.Response``) up to ``attempts`` times,
    retrying on a **retryable status** (any 5xx, plus the explicit ``retryable_statuses`` e.g. 429) or a
    transport/timeout error, with **jittered** linear backoff (``min(base*(n+1), cap) + rand(0,0.5)``).

    Returns the first non-retryable ``Response`` (the caller does ``raise_for_status()`` / parsing);
    raises the last error if every attempt was retryable.

    **Idempotency is the caller's contract** — only wrap operations that are safe to run more than once
    (id-preserving writes, reads, or requests carrying an idempotency key). A transport error may occur
    *after* the server committed, so a non-idempotent write must not be retried through here.
    """
    import asyncio
    import random

    import httpx

    attempts = max(1, attempts)
    retryable = frozenset(retryable_statuses)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            r = await send()
            if r.status_code >= 500 or r.status_code in retryable:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {r.status_code}", request=r.request, response=r)
            else:
                return r
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
        if attempt < attempts - 1:
            if attempt == 0 and on_first_retry is not None:
                try:
                    on_first_retry()
                except Exception:  # noqa: BLE001 — a notice must never break the retry
                    pass
            delay = min(base * (attempt + 1), cap) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def try_lock_checkpoint(path: str | None):  # noqa: ANN201 — returns an open file handle or None
    """Best-effort advisory lock (Unix `flock`) so two runs with the SAME checkpoint don't interleave
    appends and corrupt it (audit H3). Returns an open handle holding a non-blocking exclusive lock —
    **keep it for the run's lifetime** (closing/GC releases it) — or ``None`` if the lock is held by
    another run (with a warning), or if locking is unavailable (Windows / no ``fcntl`` / no path). Never
    raises; a failure to lock never blocks the run."""
    if not path:
        return None
    try:
        import fcntl
    except Exception:  # noqa: BLE001 — Windows / restricted: skip locking, run unguarded
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = open(path + ".lock", "w")  # noqa: SIM115 — handle is intentionally kept open for the run
    except OSError:
        return None
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except OSError:
        fh.close()
        sys.stderr.write(
            f"[june-bench] another run appears to be using {path} — concurrent runs with the same "
            f"config can corrupt the checkpoint. Use --fresh, a different size, or wait for it to finish.\n")
        return None


def probe_config(url: str, key: str, *, timeout: float = 5.0) -> dict:
    """GET the endpoint's answer + embeddings health → its effective config (**capabilities only** — no
    model ids, moat rule). One shared implementation for the CLI result footnote and the reproduce
    ``--show-config`` auditor path (was duplicated). **Narrow** fail-soft: only network/decode errors are
    swallowed (→ ``{}``); an unexpected code error (KeyError/AttributeError) propagates rather than being
    misread as 'endpoint down'."""
    if not url:
        return {}
    import json as _json

    import httpx
    hdr = {"X-API-Key": key} if key else {}
    try:
        with httpx.Client(base_url=url.rstrip("/"), timeout=timeout, headers=hdr) as c:
            a = c.get("/v1/answer/health").json()
            try:
                e = c.get("/v1/embeddings/health").json()
            except (httpx.HTTPError, httpx.InvalidURL, _json.JSONDecodeError):
                e = {}
        return {**a, "_dense": "on" if e.get("enabled") else "off"}
    except (httpx.HTTPError, httpx.InvalidURL, _json.JSONDecodeError):
        return {}


__all__ = ["env_int", "env_float", "env_flag", "retry_request", "probe_config", "try_lock_checkpoint"]
