"""june-bench CLI (SB0) — `run`, `list`. SB5 adds `suite`/`score`/`report`; SB-fetch adds `fetch`.

    june-bench list
    june-bench run --system echo --dataset smoke --split smoke
    june-bench run --system null --dataset smoke --json
    june-bench fetch --datasets all            # download the full splits → ~/.cache/june-bench/data

The `echo`/`null` systems + the bundled `smoke` dataset need no deps, key, or network, so this runs
offline immediately after `pip install june-bench`. `fetch` lands the full splits for `--split full`.
"""
from __future__ import annotations

import argparse
import os
import sys

import pathlib

from june_bench import datasets, systems
from june_bench.datasets.fetch import default_data_dir, fetch
from june_bench.datasets.fetch_sources import SOURCES
from june_bench.report import suite_json, suite_markdown, to_json, to_markdown
from june_bench.retrieval import score_retrieval
from june_bench.runner import run
from june_bench.score import score


def _cmd_list(_args) -> int:  # noqa: ANN001
    print("systems: ", ", ".join(systems.names()))
    print("datasets:", ", ".join(datasets.registry.names()))
    return 0


def _maybe_judge(args, records, summary) -> None:  # noqa: ANN001
    """Add a verbosity-agnostic LLM-judged correctness score to ``summary`` when --judge is set.
    A no-op (with a clear warning) if no judge LLM is configured — never crashes the run."""
    if not getattr(args, "judge", False):
        return
    from june_bench.judge import judge_from_env, judge_records
    jf = judge_from_env()
    if jf is None:
        print("warning: --judge set but no judge LLM configured "
              "(set JUNE_JUDGE_LLM_* or JUNE_ANSWER_LLM_* / OPENROUTER_API_KEY)", file=sys.stderr)
        return
    summary["judge"] = round(judge_records(records, jf), 4)


def _probe_server() -> dict:
    """GET the June endpoint's answer + embeddings health → its effective config (reason, max_sources,
    synthesizer, dense-lane on/off), so a result records what ACTUALLY ran. Capabilities only — the
    endpoint never returns model ids (moat rule). Fail-soft → {} (no URL/unreachable). Shared helper."""
    import os

    from june_bench._util import probe_config
    return probe_config(os.environ.get("JUNE_BENCH_JUNE_URL", ""),
                        os.environ.get("JUNE_BENCH_JUNE_KEY", ""))


def _run_config_line(args) -> str:
    """A one-line, self-describing config string for the result footnote: the knobs that determine
    the numbers (mode · answerer · reason · max_sources · embedder · multihop · judge). Server-side
    values are probed from /v1/answer/health; client-side ones come from the bench env."""
    import os
    truthy = ("1", "true", "yes")
    rec = os.environ.get("JUNE_BENCH_JUNE_RECORD", "").strip().lower() in truthy
    pool = os.environ.get("JUNE_BENCH_JUNE_POOL", "").strip().lower() in truthy
    if pool:
        rec = False
    _mode = "pool(open-pool·retrieves)" if pool else ("record(supplied-passages)" if rec
                                                      else "ingest→retrieve")
    parts = [f"mode={_mode}"]
    if not rec and os.environ.get("JUNE_BENCH_JUNE_BACKFILL", "").strip().lower() in truthy:
        parts.append("backfill=on")
    h = _probe_server()
    if h:
        if h.get("synthesizer"):
            parts.append(f"answerer={h['synthesizer']}")           # e.g. llm:openai/gpt-4o
        if h.get("max_sources") is not None:
            parts.append(f"max_sources={h['max_sources']}")
        # The ANSWER token cap GOVERNS strict span EM/F1: the default 64 forces the bare span; a high
        # cap lets elaborative models write sentences and EM/F1 collapse even when right. Record it so
        # a brevity-defeating run (e.g. MAX_TOKENS=512 → EM≈0) is self-labeled, never a silent "0.0".
        if h.get("max_tokens") is not None:
            parts.append(f"max_tokens={h['max_tokens']}")
        if h.get("reasoner"):
            step = h.get("reason_step") or h.get("reason_strategy") or "on"
            parts.append(f"reason={step}·hops={h.get('reason_max_hops')}")
            if h.get("reason_verify"):                              # R4 self-correction rung, if on
                parts.append(f"verify={h.get('verifier') or 'on'}·rev={h.get('reason_max_revisions', 0)}"
                             + ("·adaptive" if h.get("reason_adaptive") else ""))
        else:
            parts.append("reason=off")
        if not rec:                                                # dense lane only matters for retrieval/ingest
            parts.append(f"dense={h.get('_dense') or 'off'}")
    if getattr(args, "judge", False):
        jm = os.environ.get("JUNE_JUDGE_LLM_MODEL") or os.environ.get("JUNE_ANSWER_LLM_MODEL", "?")
        parts.append(f"judge={jm}")
    return " · ".join(parts)


def _dump_records(path: str, tagged) -> None:  # noqa: ANN001
    """Write each scored Record as one JSONL line (the one-flag replacement for manual curl dumps).

    Strict span EM/F1 scores a *correct-but-verbose* answer as 0 — so a sudden ``EM≈0`` could be a
    real miss OR just a model writing sentences. This dumps the actual prediction next to its gold so
    you can tell which: ``n_pred_words`` makes verbosity obvious at a glance (a 40-word answer to a
    2-word gold is the verbose-zero signature, not a wrong answer). ``tagged`` is ``(system, dataset,
    Record)`` tuples; truncates the file once per run."""
    import json as _json
    with open(path, "w", encoding="utf-8") as fh:
        for system, dataset, r in tagged:
            m = r.meta or {}
            fh.write(_json.dumps({
                "system": system, "dataset": dataset, "qid": r.qid, "question": r.question,
                "golds": list(r.golds), "prediction": r.prediction,
                "n_pred_words": len((r.prediction or "").split()),
                "abstained": bool(getattr(r, "abstained", False)),
                "system_error": bool(m.get("system_error", False)), "mode": m.get("mode", ""),
                # ``degraded`` names any fail-soft fallback that fired. A ``synth:<name>`` marker means
                # the LLM synthesizer RAISED and June degraded to the extractive floor (mode=local) —
                # i.e. the "answer" is a passage dump, not the model. This is how you spot a silently
                # mislabeled floor result (EM≈0 + judge≈ok + mode=local = degraded, not a model loss).
                "degraded": list(m.get("degraded", []) or []),
            }, ensure_ascii=False) + "\n")
    print(f"dumped {len(tagged)} answer(s) → {path}", file=sys.stderr)


def _cmd_run(args) -> int:  # noqa: ANN001
    try:
        system = systems.get(args.system)
        dataset = datasets.registry.get(args.dataset)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records = run(system, dataset, split=args.split, limit=args.limit)
    if getattr(args, "dump", ""):
        _dump_records(args.dump, [(args.system, args.dataset, r) for r in records])
    summary = score(records)
    _maybe_judge(args, records, summary)
    if args.json:
        print(to_json(summary, system=args.system, dataset=args.dataset,
                      split=args.split, model=args.model))
    else:
        md = to_markdown(summary, system=args.system, dataset=args.dataset,
                         split=args.split, model=args.model)
        cfg = _run_config_line(args)
        print(md + (f"\n_config: {cfg}_\n" if cfg else ""))
    return 0


def _cmd_suite(args) -> int:  # noqa: ANN001
    sysnames = [s.strip() for s in args.systems.split(",") if s.strip()]
    dsnames = (datasets.registry.names() if args.datasets == "all"
               else [d.strip() for d in args.datasets.split(",") if d.strip()])
    rows: list[dict] = []
    dump_rows: list = []                           # (system, dataset, Record) across cells, if --dump
    for sn in sysnames:
        for dn in dsnames:
            try:                                   # fail-soft per cell — one bad cell never kills the matrix
                recs = run(systems.get(sn), datasets.registry.get(dn),
                           split=args.split, limit=args.limit)
                if getattr(args, "dump", ""):
                    dump_rows += [(sn, dn, r) for r in recs]
                summ = score(recs)
                _maybe_judge(args, recs, summ)
                rows.append({"system": sn, "dataset": dn, "summary": summ})
            except Exception as exc:               # noqa: BLE001
                rows.append({"system": sn, "dataset": dn, "error": str(exc)})
    if getattr(args, "dump", "") and dump_rows:
        _dump_records(args.dump, dump_rows)
    if args.json:
        out = suite_json(rows, split=args.split, model=args.model)
    else:
        out = suite_markdown(rows, split=args.split, model=args.model)
        cfg = _run_config_line(args)
        if cfg:
            out = out.rstrip("\n") + f"\n\n_config: {cfg}_\n"
    if args.out:
        pathlib.Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} cells)")
    else:
        print(out)
    return 0


def _cmd_fetch(args) -> int:  # noqa: ANN001
    names = (list(SOURCES) if args.datasets == "all"
             else [d.strip() for d in args.datasets.split(",") if d.strip()])
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else default_data_dir()
    print(f"fetching {len(names)} dataset(s) → {data_dir}")
    if args.from_file and len(names) != 1:
        print("error: --from takes exactly one --datasets entry", file=sys.stderr)
        return 2
    results = fetch(names, data_dir, raw_from=args.from_file, force=args.force)
    failed = 0
    for _name, status, msg in results:
        mark = {"ok": "✓", "skipped": "•", "failed": "✗"}.get(status, "?")
        print(f"  {mark} {msg}")
        failed += status == "failed"
    if any(s == "ok" for _, s, _ in results):
        print(f"\nfull splits ready. Run with:  JUNE_BENCH_DATA={data_dir} june-bench suite "
              f"--systems june-api --datasets all --split full")
    return 1 if failed else 0


def _cmd_retrieve(args) -> int:  # noqa: ANN001
    """Score June's RETRIEVAL (recall@k/nDCG/MRR) over a dataset, not its answers."""
    from june_bench.systems import june_retrieval
    try:
        dataset = datasets.registry.get(args.dataset)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # pass the limit INTO load() so the June loader streams only the needed queries+docs (avoids the
    # whole-file parse that OOMs on the 150-250MB splits); keep the slice as a belt-and-suspenders cap.
    examples = list(dataset.load(args.split, limit=args.limit))
    if args.limit:
        examples = examples[: args.limit]
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if getattr(args, "pool", False):
        os.environ["JUNE_BENCH_RETRIEVAL_POOL"] = "1"   # from_env reads this → pooled-corpus mode
    system = june_retrieval.from_env(k=max(ks))
    rankings, golds = june_retrieval.run_retrieval(system, examples)
    summary = score_retrieval(rankings, golds, ks=ks, turn_grain=args.turn_grain)
    if args.json:
        import json as _json
        out = _json.dumps({"dataset": args.dataset, "split": args.split, "model": args.model, **summary})
    else:
        rec_vals = [summary["recall"][k] for k in ks]
        ndcg_vals = [summary["ndcg"][k] for k in ks]
        header = "| n | " + " | ".join(f"R@{k}" for k in ks) + " | " \
                 + " | ".join(f"nDCG@{k}" for k in ks) + " | MRR |"
        divider = "|---|" + "---|" * (2 * len(ks) + 1)
        rowcells = ([str(summary["n_queries"])]
                    + [f"{v:.4f}" for v in rec_vals]
                    + [f"{v:.4f}" for v in ndcg_vals]
                    + [f"{summary['mrr']:.4f}"])
        row = "| " + " | ".join(rowcells) + " |"
        tg = " · turn-grain" if args.turn_grain else ""
        cfg = _run_config_line(args)
        out = (f"### june-retrieval · {args.dataset}/{args.split}{tg}\n\n"
               f"{header}\n{divider}\n{row}\n\n"
               f"_recall/nDCG over {summary['n_queries']} judged queries; MRR@10._"
               + (f"\n_config: {cfg}_" if cfg else ""))
    if args.out:
        pathlib.Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="june-bench",
                                 description="Reproducible benchmark suite (June + pluggable competitors)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # The friendly, no-env-vars path for a non-expert to reproduce June's open-pool number.
    rp = sub.add_parser("reproduce",
                        help="one-command, plain-language reproduction of June's open-pool HotpotQA number")
    rp.add_argument("--key", default="", help="access key (else prompted / $JUNE_BENCH_JUNE_KEY)")
    rp.add_argument("--llm-key", dest="llm_key", default="",
                    help="your OpenRouter key (else prompted / $OPENROUTER_API_KEY); you pay for answers")
    rp.add_argument("--model", default="",
                    help="answer model, any OpenRouter id (else asked); e.g. openai/gpt-4o, "
                         "anthropic/claude-opus-4-8 — June's pipeline runs end-to-end with YOUR model")
    rp.add_argument("--questions", type=int, default=0, help="how many (e.g. 5, 24, 100); else asked")
    rp.add_argument("--no-judge", dest="no_judge", action="store_true", help="skip the judged-correct column")
    rp.add_argument("--fresh", action="store_true",
                    help="ignore any saved checkpoint and start this run from scratch (it still saves a "
                         "new one, so a drop is resumable)")
    rp.add_argument("--show-config", dest="show_config", action="store_true",
                    help="also print the endpoint's effective config (lanes/limits) for auditors")
    rp.set_defaults(func=lambda args: __import__("june_bench.reproduce", fromlist=["run_reproduce"]).run_reproduce(args))

    # Plain-language June-vs-Cognee head-to-head — no env vars, no bash. Asks 2 keys + run size, bakes the
    # matched stack (same embedder discovered from June's probe, same model, same open-pool task, same judge).
    rh = sub.add_parser("reproduce-h2h",
                        help="one-command, plain-language June-vs-Cognee head-to-head (matched open-pool)")
    rh.add_argument("--key", default="", help="access key (else prompted / $JUNE_BENCH_JUNE_KEY)")
    rh.add_argument("--llm-key", dest="llm_key", default="",
                    help="your OpenRouter key (else prompted / $OPENROUTER_API_KEY); pays for both systems")
    rh.add_argument("--model", default="", help="answer model on BOTH sides (default openai/gpt-4o; Opus "
                                                "is blocked for Cognee — $90+, aborted)")
    rh.add_argument("--questions", type=int, default=0, help="how many (e.g. 5, 100); else asked")
    rh.add_argument("--cot", dest="cot", action="store_true", default=None,
                    help="Cognee chain-of-thought tier (matches June's multi-hop); else asked")
    rh.add_argument("--one-shot", dest="cot", action="store_false",
                    help="Cognee one-shot graph completion instead of chain-of-thought")
    rh.add_argument("--no-judge", dest="no_judge", action="store_true", help="skip the judged-correct column")
    # Embedder for the fair comparison — three modes (else asked): default (Cognee's own), specify, discover.
    rh.add_argument("--embedder", default="",
                    help="mode 2: the embedder id June uses, matched to a local fastembed model "
                         "(exact-embedder run). Omit → Cognee's own default (moat-safe).")
    rh.add_argument("--admin-key", dest="admin_key", default="",
                    help="mode 3: your ADMIN key to auto-discover June's embedder from the authenticated "
                         "config route (the public endpoint never discloses it).")
    rh.set_defaults(func=lambda args: __import__(
        "june_bench.reproduce_h2h", fromlist=["run_reproduce_h2h"]).run_reproduce_h2h(args))

    # Retrieval reproduction — recall@k/nDCG/MRR (how well June FINDS evidence). No LLM key needed.
    rpr = sub.add_parser("reproduce-retrieval",
                         help="plain-language reproduction of June's RETRIEVAL numbers (recall@k/nDCG/MRR)")
    rpr.add_argument("--key", default="", help="access key (else prompted / $JUNE_BENCH_JUNE_KEY)")
    rpr.add_argument("--dataset", default="", help="locomo / longmemeval / financebench (else asked)")
    rpr.add_argument("--questions", type=int, default=0, help="how many queries (default 20)")
    rpr.add_argument("--diagnose", action="store_true",
                     help="deep (k=50) breakdown: is a miss a RETRIEVAL loss (lane not surfacing gold) "
                          "or a RANKING loss (gold below 10 → a reranker fixes it)?")
    rpr.add_argument("--pooled", action="store_true",
                     help="HARDER cross-conversation mode: ingest EVERY conversation's docs into one "
                          "shared corpus and retrieve each query's gold out of all of them. Default is "
                          "per-query haystack (retrieve within the query's own conversation) — the same "
                          "scope June's local numbers were measured at.")
    rpr.add_argument("--deep", action="store_true",
                     help="ask the endpoint to rerank (deep=on). Off by default: the local diagnosis "
                          "measured reranking NET-NEGATIVE on this data (it demotes an "
                          "already-healthy top-5), so the champion config does not rerank.")
    rpr.add_argument("--dense", action="store_true", default=None,
                     help="engage the DENSE lane (backfill embeddings before searching). Interactive runs "
                          "ask; off in non-interactive use — on conversational data sparse BM25 carries "
                          "recall and embedding is the slow, lock-prone step. Adds the semantic lane.")
    rpr.add_argument("--turns", action="store_true", default=None,
                     help="FAITHFUL turn-grain reproduction (locomo only): split each session into turns, "
                          "retrieve over the conversation's ~380 turn-chunks, score session-MAX — the exact "
                          "task June's local 0.856/0.925 was measured at (the default session-grain slice is "
                          "an easier task). Interactive runs ask; ingests once per conversation to limit "
                          "SQLite lock pressure.")
    rpr.add_argument("--fresh", action="store_true",
                     help="ignore any saved checkpoint and start this run from scratch (it still saves a "
                          "new one, so a drop is resumable)")
    rpr.add_argument("--show-config", dest="show_config", action="store_true",
                     help="also print the retrieval configuration")
    rpr.set_defaults(func=lambda args: __import__(
        "june_bench.reproduce", fromlist=["run_reproduce_retrieval"]).run_reproduce_retrieval(args))

    sub.add_parser("list", help="list registered systems + datasets").set_defaults(func=_cmd_list)
    r = sub.add_parser("run", help="run one system on one dataset split")
    r.add_argument("--system", default="echo")
    r.add_argument("--dataset", default="smoke")
    r.add_argument("--split", default="smoke")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--model", default="", help="model id, recorded in the result header")
    r.add_argument("--judge", action="store_true",
                   help="add a verbosity-agnostic LLM-judged correctness column (JUNE_JUDGE_LLM_*)")
    r.add_argument("--dump", default="",
                   help="write per-question answers (prediction vs gold + word count) to this JSONL "
                        "path — confirms whether an EM≈0 is verbosity or a real miss")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_run)

    su = sub.add_parser("suite", help="run a systems × datasets matrix → a RESULTS.md table")
    su.add_argument("--systems", default="echo", help="comma-separated system names")
    su.add_argument("--datasets", default="smoke", help="comma-separated names, or 'all'")
    su.add_argument("--split", default="smoke")
    su.add_argument("--limit", type=int, default=None)
    su.add_argument("--model", default="")
    su.add_argument("--out", default="", help="write the table to this path (else stdout)")
    su.add_argument("--judge", action="store_true",
                    help="add a verbosity-agnostic LLM-judged correctness column (JUNE_JUDGE_LLM_*)")
    su.add_argument("--dump", default="",
                    help="write per-question answers (prediction vs gold + word count) to this JSONL "
                         "path — confirms whether an EM≈0 is verbosity or a real miss")
    su.add_argument("--json", action="store_true")
    su.set_defaults(func=_cmd_suite)

    f = sub.add_parser("fetch", help="download the full dataset splits into the data dir")
    f.add_argument("--datasets", default="all", help="comma-separated names, or 'all'")
    f.add_argument("--data-dir", default="", help="where to write (default: $JUNE_BENCH_DATA or ~/.cache/june-bench/data)")
    f.add_argument("--from", dest="from_file", default=None,
                   help="normalize a LOCAL raw file instead of downloading (one --datasets entry)")
    f.add_argument("--force", action="store_true", help="refetch even if the file already exists")
    f.set_defaults(func=_cmd_fetch)

    rt = sub.add_parser("retrieve", help="score June's RETRIEVAL (recall@k/nDCG/MRR) over a dataset")
    rt.add_argument("--dataset", default="locomo", help="locomo / longmemeval / financebench / 2wiki / …")
    rt.add_argument("--split", default="smoke")
    rt.add_argument("--limit", type=int, default=None)
    rt.add_argument("--ks", default="1,5,10", help="comma-separated cutoffs for recall/nDCG")
    rt.add_argument("--turn-grain", action="store_true",
                    help="collapse turn-grain chunk ids to parent (session) ids before scoring")
    rt.add_argument("--pool", action="store_true",
                    help="POOLED corpus: ingest ALL docs once + retrieve each query over the whole pool "
                         "(cross-conversation, comparable to the local diagnosis) instead of per-query "
                         "canvases (within-conversation, easy → saturates at 1.0)")
    rt.add_argument("--model", default="", help="retrieval stack id, recorded in the header")
    rt.add_argument("--out", default="")
    rt.add_argument("--json", action="store_true")
    rt.set_defaults(func=_cmd_retrieve)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
