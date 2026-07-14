"""Fetch sources + PURE normalizers for the full dataset splits (SB-fetch).

`june-bench fetch` turns each public benchmark into the exact on-disk JSON the loaders in
``loaders.py`` already parse — so a third party with no June source can `pip install june-bench`,
fetch, and run the full suite. Each dataset is **one `Source` adapter** (§2 modularity): canonical
mirror URLs + a *pure* `normalize(payload) -> obj` that emits the loader's shape. The network I/O
and round-trip validation live in the CLI; everything here is pure and unit-tested without a network.

Two target shapes (see loaders.py):
  * **hotpot**  `[{_id, question, answer, context:[[title,[sents]]]}]`  (hotpot / 2wiki / musique)
  * **june**    `{documents:[{id,text}], queries:[{id,query,answer,gold,question_type}]}`  with
                conversation-scoped ids ``<conv>::<chunk>``        (locomo / longmemeval / financebench)

A fetched file is only written if it round-trips through the real loader into ≥1 Example, so a wrong
URL or format drift fails loudly rather than producing silent garbage. When a mirror is unreachable,
the CLI prints the source's ``manual`` hint (canonical page + filename) — the normalizer still applies
once the raw file is in place, so manual download is always a clean fallback.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    name: str
    full_file: str                       # the filename loaders.py expects in the data dir
    fmt: str                             # "hotpot" | "june" (the loader format it must satisfy)
    urls: tuple[str, ...]                # mirrors, tried in order
    normalize: Callable[[object], object]  # pure: parsed payload -> the object to json.dump
    member: str | None = None            # if the download is a .zip, the member to read
    jsonl: bool = False                  # payload is JSON-lines (one object per line)
    manual: str = ""                     # human fallback when every mirror fails
    headers: dict = field(default_factory=dict)


# ── hotpot family ───────────────────────────────────────────────────────────────────────────
def _norm_hotpot_passthrough(payload):
    """HotpotQA's own dev file is already the target shape — just guarantee a list + an `_id`."""
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r.setdefault("_id", r.get("id", i))      # 2wiki uses `id`; hotpot uses `_id`
        out.append(r)
    return out


def _norm_musique(payload):
    """MuSiQue jsonl → hotpot shape: paragraphs[{title,paragraph_text}] → context[[title,[text]]]."""
    out = []
    for i, r in enumerate(payload or []):
        paras = r.get("paragraphs", []) or []
        context = [[p.get("title", ""), [p.get("paragraph_text", "")]] for p in paras]
        out.append({"_id": r.get("id", i), "question": r.get("question", ""),
                    "answer": r.get("answer", ""), "context": context,
                    "type": "musique"})
    return out


# ── june (memory-QA) family ─────────────────────────────────────────────────────────────────
def _june_blob(documents, queries):
    return {"documents": documents, "queries": queries}


def _turns_text(session) -> str:
    """Session turns → one text block (mirrors scripts/convert_dataset.py:_turns_text)."""
    parts = []
    for t in session or []:
        if isinstance(t, dict):
            role = t.get("role") or t.get("speaker") or ""
            content = t.get("content") or t.get("text") or t.get("clean_text") or ""
            parts.append(f"{role}: {content}".strip(": ").strip())
        else:
            parts.append(str(t))
    return "\n".join(p for p in parts if p)


def _turn_is_gold(t) -> bool:
    return isinstance(t, dict) and bool(t.get("has_answer") or t.get("is_answer") or t.get("evidence"))


def _locomo_evidence_sessions(evidence) -> set:
    """LoCoMo evidence dialog-ids ('D1:3', sometimes packed 'D8:6; D9:17') → gold SESSION ids
    ({'session_1', …}) — the gold grain the local converter uses."""
    import re
    sids: set = set()
    for ev in evidence or []:
        for piece in re.split(r"[;,]", str(ev)):
            m = re.search(r"[Dd](\d+):", piece.strip())
            if m:
                sids.add(f"session_{int(m.group(1))}")
    return sids


def _norm_locomo(payload):
    """LoCoMo locomo10.json → june format, structurally identical to the LOCAL converter that produced
    the published headline (scripts/convert_dataset.py): ONE query per question (id
    ``<sample_id>#<qa_index>``); each query carries its conversation's SESSIONS as documents
    (``<qid>::<session_key>``, turns joined); gold = the session(s) holding the QA's evidence dialog-ids
    (``D<n>:<t>`` → ``session_<n>``). Per-query, session-grain — so the fetched full set reproduces the
    local numbers, not a divergent per-turn reimplementation."""
    documents, queries = [], []
    seen: set = set()
    for i, conv in enumerate(payload if isinstance(payload, list) else []):
        sample_id = str(conv.get("sample_id") or conv.get("id") or f"conv-{i}")
        sessions = [(str(sid), _turns_text(val))
                    for sid, val in (conv.get("conversation") or {}).items() if isinstance(val, list)]
        for j, qa in enumerate(conv.get("qa") or []):
            qid = f"{sample_id}#{j}"
            gold_sids = _locomo_evidence_sessions(qa.get("evidence"))
            gold = []
            for sid, text in sessions:
                did = f"{qid}::{sid}"
                if did not in seen:
                    documents.append({"id": did, "text": text})
                    seen.add(did)
                if sid in gold_sids:
                    gold.append(did)
            ans = qa.get("answer", qa.get("adversarial_answer"))
            q = {"id": qid, "query": str(qa.get("question", "")), "gold": list(dict.fromkeys(gold))}
            if qa.get("category") is not None:
                q["question_type"] = f"cat{qa.get('category')}"
            if ans is not None:
                q["answer"] = str(ans)
            queries.append(q)
    return _june_blob(documents, queries)


def _norm_longmemeval(payload):
    """LongMemEval *_s.json → june format, structurally identical to the LOCAL converter
    (scripts/convert_dataset.py:_longmemeval_units): each haystack SESSION becomes one document
    ``<question_id>::<session_id>`` (turns joined); gold = the answer session(s) in the SAME id space,
    plus any session carrying a gold-marked turn. Per-query, session-grain → reproduces the local number."""
    documents, queries = [], []
    seen: set = set()
    for item in payload or []:
        qid = str(item.get("question_id") or item.get("id") or "")
        sessions = item.get("haystack_sessions") or item.get("sessions") or []
        sids = item.get("haystack_session_ids") or [f"s{k}" for k in range(len(sessions))]
        answer_ids = {str(x) for x in (item.get("answer_session_ids") or item.get("answer_sessions") or [])}
        gold = []
        for idx, session in enumerate(sessions):
            sid = str(sids[idx]) if idx < len(sids) else f"s{idx}"
            did = f"{qid}::{sid}"
            if did not in seen:
                documents.append({"id": did, "text": _turns_text(session)})
                seen.add(did)
            if sid in answer_ids or any(_turn_is_gold(t) for t in (session or [])):
                gold.append(did)
        q = {"id": qid, "query": str(item.get("question", "")), "gold": list(dict.fromkeys(gold))}
        qt = str(item.get("question_type") or "").strip()
        if qt:
            q["question_type"] = qt
        if item.get("answer") is not None:
            q["answer"] = str(item.get("answer"))
        queries.append(q)
    return _june_blob(documents, queries)


def _fb_page(ev: dict) -> tuple[str, str]:
    """One FinanceBench evidence entry → (page_id, page_text). Page id ``{doc_name}#p{page}`` dedupes
    the same page globally; full-page text is preferred so distractors carry realistic context."""
    doc = str(ev.get("doc_name") or "doc")
    page = ev.get("evidence_page_num", ev.get("page_number"))
    pid = f"{doc}#p{page}" if page is not None else doc
    text = ev.get("evidence_text_full_page") or ev.get("evidence_text") or ""
    return pid, str(text)


def _norm_financebench(payload):
    """FinanceBench → june RETRIEVAL format (real gold, not evidence-grounded).

    The jsonl ships only evidence pages (not full PDFs), so the self-contained corpus is the GLOBAL
    pool of every question's evidence pages: each question must find ITS gold page among all of them
    (distractors = other questions' pages). june_bench's loader scores each query only within docs
    sharing its ``_conv_key``, so the pool is **namespaced under each question's id** (`{fid}::{page}`)
    and gold pages are emitted FIRST so they survive the loader's per-conversation corpus cap. This
    mirrors `scripts/convert_dataset.py --profile financebench` (an easier proxy for full-10-K
    retrieval; the faithful task needs the PDF corpus). BM25 is weak here — run with the dense lane."""
    import hashlib

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    pool: dict[str, str] = {}                                    # global page_id → full-page text
    for r in rows or []:
        for ev in (r.get("evidence") or []):
            if isinstance(ev, dict):
                pid, text = _fb_page(ev)
                if pid and text:
                    pool.setdefault(pid, text)
    pool_ids = list(pool.keys())
    from june_bench.datasets.loaders import _max_corpus
    max_distractors = max(1, _max_corpus() - 10)                # derive from the loader cap (was a magic 150)

    documents, queries = [], []
    for i, r in enumerate(rows or []):
        fid = str(r.get("financebench_id") or r.get("id") or i).strip()
        gold_pids: list[str] = []
        for ev in (r.get("evidence") or []):
            if isinstance(ev, dict):
                pid, text = _fb_page(ev)
                if pid and text and pid not in gold_pids:
                    gold_pids.append(pid)
        if not gold_pids:                                       # no resolvable gold page → skip (can't score)
            continue
        # Deterministic distractor sample (stable per fid) from the rest of the global pool.
        others = [p for p in pool_ids if p not in gold_pids]
        others.sort(key=lambda p: hashlib.md5(f"{fid}:{p}".encode(), usedforsecurity=False).hexdigest())
        chosen = gold_pids + others[:max_distractors]          # gold FIRST → survives the corpus cap
        hay = []
        for pid in chosen:
            did = f"{fid}::{pid}"
            documents.append({"id": did, "text": pool[pid]})
            hay.append(did)
        ans = r.get("answer")
        queries.append({"id": fid, "query": str(r.get("question", "")),
                        "answer": "" if ans is None else str(ans),
                        "gold": [f"{fid}::{p}" for p in gold_pids], "haystack": hay,
                        "question_type": str(r.get("question_type") or "financebench")})
    return _june_blob(documents, queries)


# ── the source registry (canonical mirrors; tried in order) ──────────────────────────────────
SOURCES: dict[str, Source] = {
    "hotpot": Source(
        "hotpot", "hotpot_dev.json", "hotpot",
        ("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",),
        _norm_hotpot_passthrough,
        manual="HotpotQA dev (distractor): https://hotpotqa.github.io → save "
               "hotpot_dev_distractor_v1.json as hotpot_dev.json in your data dir."),
    "2wiki": Source(
        "2wiki", "2wiki_dev.json", "hotpot",
        ("https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",),
        _norm_hotpot_passthrough, member="dev.json",
        manual="2WikiMultihopQA: https://github.com/Alab-NII/2wikimultihop → unzip, save "
               "dev.json as 2wiki_dev.json in your data dir."),
    "musique": Source(
        "musique", "musique_dev.json", "hotpot",
        ("https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",),
        _norm_musique, jsonl=True,
        manual="MuSiQue: https://github.com/StonyBrookNLP/musique (data is on Google Drive) → save "
               "musique_ans_v1.0_dev.jsonl, then `june-bench fetch --datasets musique --from <path>`."),
    "locomo": Source(
        "locomo", "locomo.june.json", "june",
        ("https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",),
        _norm_locomo,
        manual="LoCoMo: https://github.com/snap-research/locomo → data/locomo10.json."),
    "longmemeval": Source(
        "longmemeval", "lme.june.json", "june",
        # The original `xiaowu0162/longmemeval` HF repo is DEPRECATED (its `longmemeval_s.json` 404s);
        # the maintainer replaced it with `longmemeval-cleaned` (noisy sessions removed). The published
        # numbers use the cleaned `longmemeval_s_cleaned.json` (277 MB, LFS).
        ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",),
        _norm_longmemeval,
        manual="LongMemEval (cleaned): https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned "
               "→ download longmemeval_s_cleaned.json, then "
               "`june-bench fetch --datasets longmemeval --from <path>`."),
    "financebench": Source(
        "financebench", "financebench.june.json", "june",
        # PatronusAI renamed `financebench_open_source.jsonl` → `financebench_merged.jsonl`.
        # NOTE: `_norm_financebench` yields evidence-grounded QA rows (gold empty) — fine for the QA
        # signal, but the RETRIEVAL full set with proper gold pages is built by
        # `scripts/convert_dataset.py --profile financebench`, not this quick normalizer.
        ("https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl",),
        _norm_financebench, jsonl=True,
        manual="FinanceBench: https://huggingface.co/datasets/PatronusAI/financebench → "
               "financebench_merged.jsonl."),
}

__all__ = ["Source", "SOURCES"]
