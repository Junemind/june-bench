"""LLM-judge correctness — the verbosity-agnostic ruler (Gap #2, mirrors `score_deepeval.py`).

Strict EM/F1 punishes a *correct* answer for being phrased differently — fine when both systems emit
bare spans, unfair when one returns a paragraph (e.g. Cognee's graph-completion vs June's terse
synthesizer). This judges **semantic correctness**: given the QUESTION, the reference ANSWER(s), and
a PREDICTION, a single fixed LLM decides yes/no, *ignoring surface formatting* — DeepEval's GEval
"Correctness" criterion. Held identical across systems, it's an apples-to-apples accuracy where the
only thing that differs is the engine. The model lives in an injected callable (httpx adapter); the
scorer logic is pure.

Env (separate namespace, falls back to the answer model so one key serves both):
    JUNE_JUDGE_LLM_URL / _MODEL / _KEY / _TIMEOUT   (else JUNE_ANSWER_LLM_*, then OPENROUTER_API_KEY)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Sequence

from june_bench._util import env_float

_LOG = logging.getLogger(__name__)

# Transient statuses worth retrying — a multi-system judged suite fires hundreds of sequential calls
# (esp. after a graph-building competitor like Cognee), so one 429/5xx must not zero a correct answer.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class _JudgeError(Exception):
    """A judge CALL failed (network/HTTP/parse) — distinct from the judge saying 'no'. Surfaced so a
    rate-limited or misconfigured judge is never silently scored as 'incorrect' (which would make a
    correct system look wrong, e.g. judge < EM)."""

JUDGE_SYSTEM = (
    "You grade a question-answering system. Given a QUESTION, the reference correct ANSWER(s), and "
    "the system's PREDICTION, decide whether the prediction is a correct answer to the question. "
    "Ignore surface differences — phrasing, extra words, word order, capitalization, punctuation, and "
    "whether it's a full sentence; judge ONLY whether it conveys a correct answer that matches the "
    "reference. A prediction that states the correct fact is correct even if verbose; an empty, "
    "evasive, or 'I don't know' prediction is incorrect. Reply with exactly one word: yes or no."
)


def build_judge_prompt(question: str, golds: Sequence[str], prediction: str) -> tuple[str, str]:
    """Pure (system, user) for the judge."""
    refs = " | ".join(g for g in golds if g) or "(none provided)"
    user = (f"QUESTION: {(question or '').strip()}\n"
            f"REFERENCE ANSWER(S): {refs}\n"
            f"PREDICTION: {(prediction or '').strip() or '(empty)'}\n\n"
            "Is the PREDICTION correct? Reply yes or no.")
    return JUDGE_SYSTEM, user


def parse_verdict(text: str) -> float:
    """1.0 if the judge said yes/correct/true, else 0.0 (fail-soft: unparseable → 0)."""
    t = (text or "").strip().lower().lstrip("-*•").strip()
    first = t.split()[0].rstrip(".,!:") if t.split() else ""
    return 1.0 if first in ("yes", "correct", "true", "y") else 0.0


def _env_retries(default: int = 5) -> int:
    try:
        return max(1, int(os.environ.get("JUNE_JUDGE_LLM_RETRIES", str(default))))
    except ValueError:
        return default


def _chat_judge(url: str, model: str, key: str, timeout: float):
    """Build a ``judge(question, golds, prediction) -> float`` over an OpenAI-compatible endpoint.

    Retries transient errors (429/5xx/transport) with backoff — a judged H2H fires hundreds of
    sequential calls and one rate-limit blip must not score a correct answer as wrong. On EXHAUSTED
    failure it raises ``_JudgeError`` (NOT a silent 0.0), so the caller can exclude-and-warn rather
    than fabricate an 'incorrect'."""
    def judge(question: str, golds, prediction: str) -> float:  # noqa: ANN001
        import httpx
        system, user = build_judge_prompt(question, golds, prediction)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model, "temperature": 0, "max_tokens": 8,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        retries = _env_retries()
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = httpx.post(url, headers=headers, timeout=timeout, json=payload)
                if r.status_code in _RETRYABLE_STATUS:
                    last = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                else:
                    r.raise_for_status()
                    return parse_verdict(r.json()["choices"][0]["message"]["content"])
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last = exc
            if attempt < retries - 1:
                time.sleep(min(2.0 ** attempt, 30.0) + 0.5)
        raise _JudgeError(f"judge call failed after {retries} attempts: {last}") from last
    return judge


def judge_from_env():
    """A judge callable from `JUNE_JUDGE_LLM_*` (falling back to `JUNE_ANSWER_LLM_*` / OpenRouter), or
    **None** when no judge LLM is configured (so `--judge` degrades to a clear message, never a crash)."""
    url = (os.environ.get("JUNE_JUDGE_LLM_URL") or os.environ.get("JUNE_ANSWER_LLM_URL", "")).strip()
    model = (os.environ.get("JUNE_JUDGE_LLM_MODEL") or os.environ.get("JUNE_ANSWER_LLM_MODEL", "")).strip()
    key = (os.environ.get("JUNE_JUDGE_LLM_KEY") or os.environ.get("JUNE_ANSWER_LLM_KEY")
           or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not url or not model or not key:
        return None
    timeout = env_float("JUNE_JUDGE_LLM_TIMEOUT", 60.0, lo=1.0)
    return _chat_judge(url, model, key, timeout)


def _mean(xs) -> float:  # noqa: ANN001
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def judge_records(records, judge_fn) -> float:
    """Selective judged accuracy over **answered** records (parallels the EM/F1 axes; coverage already
    captures abstention).

    A **failed** judge CALL (rate-limit/network/parse — raised as ``_JudgeError`` after retries) is
    EXCLUDED from the mean, never scored 0.0. Scoring a failed call as 'incorrect' is what made a
    correct system look wrong (e.g. judge 0.0 < EM 0.478 when a graph-building competitor exhausted
    the rate budget before its judge phase). Excluded failures are logged and a loud warning fires if
    any occurred, so the judge number is honest about how many items it actually graded."""
    graded: list[float] = []
    failed = 0
    for r in records:
        if getattr(r, "abstained", False):
            continue
        try:
            graded.append(judge_fn(r.question, r.golds, r.prediction))
        except _JudgeError as exc:
            failed += 1
            _LOG.warning("judge call failed (excluded, not scored 0): %s", exc)
        except Exception as exc:  # noqa: BLE001 — any other error: exclude too, never fabricate a 0
            failed += 1
            _LOG.warning("judge call errored (excluded): %s: %s", type(exc).__name__, exc)
    if failed:
        sys.stderr.write(
            f"[june-bench] JUDGE: {failed} judge call(s) FAILED and were EXCLUDED (not scored wrong). "
            f"The judge score is over the {len(graded)} item(s) that graded; a high failure count "
            f"means the judge was rate-limited/misconfigured — raise JUNE_JUDGE_LLM_RETRIES, use a "
            f"separate JUNE_JUDGE_LLM_KEY, or re-run. Do NOT read this as the system being wrong.\n")
    return _mean(graded)


__all__ = ["JUDGE_SYSTEM", "build_judge_prompt", "parse_verdict", "judge_from_env", "judge_records"]
