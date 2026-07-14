"""CogneeSystem (SB4) — the first pluggable competitor, behind the `[cognee]` extra.

Cognee is heavy (its own LLM + embedding + graph stack), so the adapter follows the same discipline
as `JuneApiSystem`: the cognee machinery lives in **injected callables**, and the adapter is pure
plumbing that is unit-tested with fakes (no cognee, no keys). A `from_cognee()` factory builds those
callables from the cognee library when it's installed — carrying the Anthropic-mode patch the
in-repo head-to-head driver needed (`head_to_head/cognee_h2h_driver.py`).

Flow (mirrors the head-to-head): `ingest(corpus)` builds cognee's KG from the conversation/passages
(`prune → add → cognify`); `answer(question)` runs cognee's graph-completion search. For context-QA
(HotpotQA) the per-question passages are ingested in `answer` before searching; for memory-QA the
runner ingests the conversation once via `ingest(corpus)`.

The real cognee wiring is dev-only (FL-SB4: needs `pip install cognee[evals]` + an LLM/embedding
provider; not runnable in-sandbox and not validated here against a cognee version). What ships and is
tested is the adapter contract.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from june_bench.ports import Example, Prediction
from june_bench.runner import _maybe_await


class CogneeSystem:
    """Wraps cognee behind two injected callables.

    * ``answer_fn(question) -> str`` — cognee graph-completion search.
    * ``ingest_fn(docs) -> None`` (optional) — build/extend cognee's KG from documents.
    """
    name = "cognee"

    def __init__(self, answer_fn, *, ingest_fn=None,  # noqa: ANN001
                 qa_engine: str = "cognee_graph_completion", pooled: bool = False) -> None:
        self._answer_fn = answer_fn
        self._ingest_fn = ingest_fn
        self.qa_engine = qa_engine
        # pooled=True → OPEN-POOL QA: build the KG ONCE over the deduped union of ALL evidence (via
        # `ingest_pool`, the runner hook), then every question retrieves over that whole graph — WITHOUT
        # a per-question prune+rebuild. This mirrors JuneApiSystem's pooled mode AND the LOCAL
        # head-to-head driver (`run_corpus_builder` builds the corpus once, then answers all). The
        # per-question `answer`-time ingest (below) scoped Cognee to one question's passages online,
        # making a "June-pool vs Cognee" run an unmatched task; pooled mode restores parity.
        self._pooled = pooled
        self._pool_ready = False

    async def ingest(self, corpus: Sequence[str]) -> None:
        # POOLED: the shared pool is built ONCE by `ingest_pool`; per-example ingest is a no-op (so the
        # runner's per-question `ingest(ex.corpus)` can't prune+rebuild the pooled graph).
        if self._pooled:
            return
        if self._ingest_fn is not None and corpus:
            await _maybe_await(self._ingest_fn(list(corpus)))

    async def ingest_pool(self, examples: Sequence[Example]) -> int:
        """OPEN-POOL setup (runner hook, mirrors `JuneApiSystem.ingest_pool` + the local driver): build
        Cognee's graph ONCE over the deduped union of every example's evidence (``context`` passages +
        ``corpus`` docs), then each question answers over that whole graph. A SINGLE reset+add+cognify —
        not a per-question teardown. Idempotent via ``_pool_ready``; dedup by normalized text (QA
        passages carry no stable id). Returns the number of pooled docs."""
        if self._pool_ready or self._ingest_fn is None:
            self._pool_ready = True
            return 0
        seen: set[str] = set()
        docs: list[str] = []
        for ex in examples:
            for passage in (*ex.context, *ex.corpus):
                t = str(passage).strip()
                key = " ".join(t.lower().split())            # normalized-text dedup
                if t and key not in seen:
                    seen.add(key)
                    docs.append(t)
        await _maybe_await(self._ingest_fn(docs))            # one reset+build+cognify over the whole pool
        self._pool_ready = True
        return len(docs)

    async def answer(self, example: Example) -> Prediction:
        # POOLED: retrieve over the shared pool built by `ingest_pool` — NO per-question ingest/prune.
        # NON-POOLED context-QA (HotpotQA): ingest this question's passages before searching (record-like).
        if not self._pooled and example.context and self._ingest_fn is not None:
            await _maybe_await(self._ingest_fn(list(example.context)))
        out = await _maybe_await(self._answer_fn(example.question))
        text = str(out or "").strip()
        return Prediction(text=text, meta={"calls": 1, "engine": self.qa_engine,
                                           "pooled": self._pooled, "abstained": not text})


async def terse_graph_completion(cognee, search_type, question: str, *,  # noqa: ANN001
                                 prompt_path: str, inline_prompt: str):
    """Run Cognee GRAPH_COMPLETION asking for a BARE SPAN, the way Cognee benchmarks itself.

    Tries the shipped benchmark prompt FILE first (``system_prompt_path`` — the documented, version-
    stable knob its eval_framework uses → the local-proven terse 0.417), then an inline terse
    ``system_prompt`` (older cognee), then a bare call (oldest). Split out as a module-level helper so
    the prompt wiring is unit-testable with a fake ``cognee`` (no install, no keys)."""
    try:
        return await cognee.search(query_type=search_type, query_text=question,
                                   system_prompt_path=prompt_path)
    except TypeError:
        pass            # an older cognee without the system_prompt_path kwarg
    try:
        return await cognee.search(query_type=search_type, query_text=question,
                                   system_prompt=inline_prompt)
    except TypeError:
        return await cognee.search(query_type=search_type, query_text=question)


def from_cognee(*, qa_engine: str = "cognee_graph_completion", reset: bool = True,
                pooled: bool | None = None) -> CogneeSystem:
    """Build a CogneeSystem from the installed `cognee` library (the `[cognee]` extra). Raises a clear
    error if cognee isn't installed. **Dev-only / not validated in-sandbox** (FL-SB4) — verify against
    your cognee version + provider keys before trusting numbers.

    ``pooled`` (default: read env) → OPEN-POOL QA, so Cognee faces the SAME retrieval task as June-pool
    (build the KG once over the whole pool, retrieve per question) instead of being handed each
    question's passages. Unset ⇒ ``COGNEE_POOL`` else ``JUNE_BENCH_JUNE_POOL`` — so a single pool flag on
    an ``--systems june-api,cognee`` H2H pools BOTH sides. Set ``COGNEE_POOL=0`` to opt Cognee out."""
    if pooled is None:
        pooled = _pool_from_env()
    try:
        import cognee  # noqa: F401
        from cognee.api.v1.search import SearchType
    except Exception as exc:  # not installed
        raise RuntimeError(
            "CogneeSystem needs cognee — `pip install june-bench[cognee]` (cognee[evals]) and an "
            "LLM/embedding provider in the env (see the head-to-head .env.cognee.example).") from exc

    _patch_cognee_anthropic_mode()

    async def ingest_fn(docs):
        if reset:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        for d in docs:
            await cognee.add(str(d))
        await cognee.cognify()

    # Fairness — make Cognee answer in bare spans, the SAME way Cognee benchmarks ITSELF. June's
    # synthesizer emits a bare span (strict span EM is meaningful); Cognee's GRAPH_COMPLETION DEFAULTS
    # to the conversational prompt `answer_simple_question.txt` ("Answer ... Be as brief as possible")
    # → full-sentence paragraphs → strict EM ~0 even when correct (exactly what we measured: judge
    # 0.875 right, EM 0.0). Cognee SHIPS a HotpotQA-tuned terse prompt, `answer_simple_question_
    # benchmark.txt` ("Minimize words; what/who → single word/phrase, no full sentences; no punctuation;
    # dry concise lowercase"), and its OWN eval_framework points `system_prompt_path` at exactly that
    # file (run_question_answering(system_prompt="answer_simple_question_benchmark.txt")) — that is the
    # config behind the local head-to-head's terse 0.417. We use the SAME shipped file (resolved inside
    # cognee's prompt dir by read_query_prompt), so this is Cognee's own benchmark answerer, not a patch.
    # Override via COGNEE_SYSTEM_PROMPT_PATH; the LLM `--judge` remains the verbosity-agnostic ruler.
    _BENCH_PROMPT = os.environ.get("COGNEE_SYSTEM_PROMPT_PATH", "answer_simple_question_benchmark.txt")
    _TERSE_INLINE = ("Output ONLY the answer — the fewest words (a name, date, number, or short noun "
                     "phrase), no full sentence, no punctuation. Yes/no questions → exactly 'yes'/'no'.")

    # Fairness vs June's grounded multi-hop REASONING: let Cognee use its CHAIN-OF-THOUGHT graph
    # completion too (`GRAPH_COMPLETION_COT`), not only one-shot `GRAPH_COMPLETION` — otherwise June's
    # `reason=llm·hops=4` is an unmatched structural edge on multi-hop QA. Pick via COGNEE_SEARCH_TYPE
    # (e.g. GRAPH_COMPLETION | GRAPH_COMPLETION_COT | GRAPH_COMPLETION_CONTEXT_EXTENSION). Version-safe:
    # if the installed cognee lacks the requested member (older builds have no COT) we WARN and fall
    # back to GRAPH_COMPLETION rather than crash. The chosen type is recorded on the engine label.
    # Show Cognee's REASONING menu so the H2H matches the right tier (fairness): list every
    # GRAPH_COMPLETION* search type the installed cognee exposes. As of 1.1.2 these are GRAPH_COMPLETION
    # (one-shot), GRAPH_COMPLETION_COT (chain-of-thought), and GRAPH_COMPLETION_CONTEXT_EXTENSION
    # (iterative context-gather) — Cognee has NO verify/revise rung (its iteration IS the CoT/extension),
    # so June's R4 verify is a June-only capability; report June with R4 OFF as the apples-to-apples CoT row.
    _reasoning = sorted(s for s in dir(SearchType) if s.startswith("GRAPH_COMPLETION"))
    sys.stderr.write(f"[june-bench] cognee reasoning search types available: {_reasoning}\n")

    _ST_NAME = os.environ.get("COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION").strip().upper()
    _SEARCH_TYPE = getattr(SearchType, _ST_NAME, None)
    if _SEARCH_TYPE is None:
        sys.stderr.write(
            f"[june-bench] COGNEE_SEARCH_TYPE={_ST_NAME!r} is not a SearchType in this cognee version — "
            f"falling back to GRAPH_COMPLETION (upgrade cognee for CoT/context-extension).\n")
        _SEARCH_TYPE = SearchType.GRAPH_COMPLETION
        _ST_NAME = "GRAPH_COMPLETION"

    async def _search_terse(question):
        return await terse_graph_completion(
            cognee, _SEARCH_TYPE, question,
            prompt_path=_BENCH_PROMPT, inline_prompt=_TERSE_INLINE)

    def _extract_answer(item) -> str:
        """Pull the answer TEXT out of whatever cognee's search returns — never str() the wrapper.

        cognee's newer GRAPH_COMPLETION returns a per-dataset dict
        ``{'dataset_id': UUID(...), 'dataset_name': ..., 'search_result': [answer, ...]}`` (not a bare
        string), so the old ``str(first)`` scored the WHOLE dict repr (UUIDs + metadata) and buried the
        real answer → an unfair EM≈0 even when ``search_result`` was exactly right. Handle the dict
        (join ``search_result``), the legacy bare string, and an object with ``.text``."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            sr = item.get("search_result")
            if isinstance(sr, (list, tuple)):
                return " ".join(_extract_answer(x) for x in sr).strip()
            if sr is not None:
                return str(sr).strip()
            for k in ("text", "answer", "completion", "content"):
                if item.get(k):
                    return str(item[k]).strip()
            return ""
        return str(getattr(item, "text", item)).strip()

    async def answer_fn(question):
        results = await _search_terse(question)
        if not results:
            return ""
        return _extract_answer(results[0])

    if pooled:
        sys.stderr.write("[june-bench] cognee OPEN-POOL: building the KG once over the whole pool "
                         "(matched to June-pool; no per-question rebuild).\n")
    return CogneeSystem(answer_fn, ingest_fn=ingest_fn, qa_engine=f"cognee_{_ST_NAME.lower()}",
                        pooled=pooled)


def _pool_from_env() -> bool:
    """Whether Cognee runs OPEN-POOL. Anti-drift contract: a single ``JUNE_BENCH_JUNE_POOL=1`` on an
    ``--systems june-api,cognee`` H2H must pool BOTH sides onto the identical retrieval task — so this
    reads the SAME flag ``june_api.from_env`` reads. ``COGNEE_POOL`` (1/0) is an explicit per-side
    override (e.g. ``COGNEE_POOL=0`` to keep June pooled but opt Cognee out). Kept as a named helper so a
    test locks the June↔Cognee flag parity and it can't silently drift."""
    _t = ("1", "true", "yes")
    _cog = os.environ.get("COGNEE_POOL", "").strip().lower()
    if _cog:
        return _cog in _t
    return os.environ.get("JUNE_BENCH_JUNE_POOL", "").strip().lower() in _t


def _patch_cognee_anthropic_mode() -> None:
    """Carry-over of the head-to-head fix so Claude works as cognee's LLM (cognee ≤1.1.2 bug)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (  # noqa: E501
            AnthropicAdapter,
        )
        AnthropicAdapter.default_instructor_mode = "tool_call"
    except Exception:
        pass  # best-effort; newer cognee may not need it


__all__ = ["CogneeSystem", "from_cognee", "terse_graph_completion", "_pool_from_env"]
