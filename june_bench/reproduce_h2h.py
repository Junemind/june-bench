"""`june-bench reproduce-h2h` — the one-command, plain-language June-vs-Cognee head-to-head.

Same philosophy as `reproduce`: the user is asked only for the two keys nobody else can supply and how
much to run — no `JUNE_BENCH_*` / `EMBEDDING_*` / `LLM_*` incantations, no bash script. Behind the
scenes this bakes the *matched* stack (same embedder, same answer model, same open-pool task, same
judge) so June and Cognee face an identical benchmark, then prints a side-by-side.

Moat-safe: the embedder is a *published* benchmark parameter (bge-large-en-v1.5 — disclosed in README +
the fairness methodology), NOT the moat (the cost-gated extraction/graph pipeline is), so it defaults in
source and is exempt from the model-name guard. The internal extraction models stay banned + tested
(tests/test_reproduce.py::test_no_model_name_anywhere_in_shipped_package). The embedder Cognee uses is
chosen by `_choose_embedder`:
  · default (recommended) — `_DEFAULT_EMBEDDER`, June's dense-lane embedder → the SAME-EMBEDDER
                            reproduction out of the box, one command, nothing to export.
  · specify              — `--embedder <id>` overrides the default with any embedder, matched to a local
                           fastembed model (also settable via `JUNE_BENCH_EMBEDDER`).
  · discover             — `--admin-key` reads June's live embedder from the authenticated
                           `/v1/embeddings/config` route (the PUBLIC health route stays redacted).

Fairness baked in: Cognee runs its own graph-RAG **locally** (it never touches June's box); it is given
the SAME embedder June's dense lane uses and the SAME answer model (gpt-4o — Cognee on Opus was $90+ and
never finished, so it's blocked). June-Opus is a SEPARATE efficiency row via `reproduce`, not this
matched one.
"""
from __future__ import annotations

import os
import sys

# Reuse the friendly primitives so the two commands look and feel identical.
from june_bench.reproduce import PRESET, _ask, _is_tty, _progress

_DEFAULT_MODEL = "openai/gpt-4o"                 # the matched row; Cognee can't affordably run Opus
# June's dense-lane embedder = the default for the matched run, so `reproduce-h2h` is one command with no
# env var to export. This is a PUBLISHED benchmark parameter (README + fairness methodology), not the moat
# — the cost-gated extraction/graph pipeline is — so naming it here is deliberate and moat-safe (bge is
# exempt from tests/test_reproduce.py::_BANNED for exactly this reason).
_DEFAULT_EMBEDDER = "bge-large-en-v1.5"
# Published open-pool targets, same 100-slice (EM, F1) — from the local head-to-head.
_TARGETS = {"june-api": (0.63, 0.80), "cognee": (0.53, 0.66)}
_LABELS = {"june-api": "June (open-pool)", "cognee": "Cognee (graph-RAG)"}
# Canonical metered API cost per 100 questions on gpt-4o, from the published head-to-head. Cognee's
# chain-of-thought fires many LLM calls per question PLUS a one-time graph build (cognify), so it costs
# ~8x June. Used as the labeled cost basis when a live per-run figure isn't captured.
_COST_PER_100Q_GPT4O = {"june-api": 1.30, "cognee": 10.64}
# Cognee's chain-of-thought tier fires ~4-5 LLM rounds per question (retrieve → follow-up → retrieve …),
# so it costs ~2x the base graph-completion figure — MEASURED ~$20/100Q on a live run (vs the $10.64 base
# above). The h2h defaults to CoT (to match June's multi-hop), so the pre-run estimate must use this or it
# undershoots the bill ~2x.
_COGNEE_COT_PER_100Q = 20.0


def _cognee_per_100q(cot: bool) -> float:
    return _COGNEE_COT_PER_100Q if cot else _COST_PER_100Q_GPT4O["cognee"]


def _cost_estimate(name: str, n: int) -> float:
    """Metered-reference cost for `n` questions of a system on gpt-4o base config (0 if unknown)."""
    return _COST_PER_100Q_GPT4O.get(name, 0.0) * max(0, n) / 100.0


def _openrouter_usage(key: str) -> float | None:
    """Authoritative cumulative $ spent on this OpenRouter key (`GET /api/v1/credits` → ``total_usage``).
    ``None`` on any failure. Both systems answer on the caller's SAME BYO key — June via the ``X-LLM-Key``
    header, Cognee via litellm locally — so a **per-phase delta** of this value is each system's REAL
    billed cost, straight from the provider (no token math, no endpoint change, no estimation). Never
    raises: a metering hiccup must not affect a paid run."""
    if not key:
        return None
    try:
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/credits",
                      headers={"Authorization": f"Bearer {key}"}, timeout=15.0)
        if r.status_code == 200:
            u = (r.json().get("data") or {}).get("total_usage")
            return float(u) if u is not None else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _norm_embedder_name(raw: str) -> str:
    """`"http:<model>+pfx"` → `"<model>"` (strip provider prefix + asymmetric-prefix suffix)."""
    return raw.split(":", 1)[-1].split("+", 1)[0].strip()


def _match_fastembed(name: str):  # noqa: ANN201 — (model_id, dims) or None
    """Match an embedder NAME to a local fastembed model by suffix — so the id is never a source literal
    (moat rule). Returns (fastembed_model_id, dims) or None."""
    name = (name or "").strip().lower()
    if not name:
        return None
    try:
        from fastembed import TextEmbedding
    except Exception:  # noqa: BLE001 — fastembed not installed; the caller's pre-flight reports it
        return None
    for e in TextEmbedding.list_supported_models():
        mid = str(e.get("model", ""))
        if mid.lower().endswith(name):            # match by suffix → never hardcode the id
            return mid, int(e.get("dim") or 1024)
    return None


def _default_fastembed():  # noqa: ANN201 — (model_id, dims) or None
    """A local fastembed model chosen AT RUNTIME (no hardcoded id → moat-safe) for mode 1, where no
    embedder was specified. The provider is forced to keyless-local fastembed, so mode 1 must still get a
    fastembed-SUPPORTED model — otherwise Cognee falls back to its OpenAI default and crashes on the local
    provider. Picks the smallest-dim supported model (cheapest local download). Returns (id, dims)/None."""
    try:
        from fastembed import TextEmbedding
        models = list(TextEmbedding.list_supported_models())
        if not models:
            return None
        best = min(models, key=lambda e: int(e.get("dim") or 10**9))
        return str(best.get("model", "")), int(best.get("dim") or 384)
    except Exception:  # noqa: BLE001 — fastembed absent → pre-flight already reports it
        return None


def _admin_embedder(url: str, admin_key: str) -> str:
    """Read June's embedder id from the AUTHENTICATED admin route `/v1/embeddings/config` (the PUBLIC
    `/v1/embeddings/health` stays redacted per the moat rule — only an admin key sees the id). Returns the
    raw id (e.g. ``"http:<model>+pfx"``) or ``""`` on any failure/denied."""
    if not url or not admin_key:
        return ""
    try:
        import httpx
        with httpx.Client(base_url=url.rstrip("/"), timeout=8.0,
                          headers={"X-API-Key": admin_key}) as c:
            r = c.get("/v1/embeddings/config")
            if r.status_code == 200:
                return str(r.json().get("embedder") or "")
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _match_or_raw(mid: str):  # noqa: ANN201 — (id, dims)
    """A given embedder id → a local fastembed model (matched by suffix), or the id at a default dim."""
    return _match_fastembed(_norm_embedder_name(mid)) or (mid, 1024)


def _discover_via_admin(url: str, ak: str):  # noqa: ANN201 — (id, dims) or None
    raw = _admin_embedder(url, ak) if ak else ""
    if not raw:
        print("   (couldn't read the embedder via admin — using Cognee's keyless local default)")
        return None
    matched = _match_fastembed(_norm_embedder_name(raw))
    if not matched:
        print("   (June's embedder has no local fastembed match — using the keyless local default)")
    return matched


def _choose_embedder(args, url: str):  # noqa: ANN001, ANN201 — (id, dims) or None
    """Decide which embedder Cognee uses for the fair comparison.

    Precedence: ``--embedder`` flag → ``--admin-key`` discover → the published DEFAULT
    (``_DEFAULT_EMBEDDER`` = June's dense-lane embedder; overridable via ``JUNE_BENCH_EMBEDDER``). The
    default IS June's embedder, so the out-of-the-box run is the *same-embedder* reproduction with nothing
    to export — the embedder is a disclosed benchmark parameter, not the moat, so defaulting to it is safe
    (see ``_DEFAULT_EMBEDDER``). Interactive runs get a one-key a/b confirm; non-interactive runs take the
    default silently. Returns (id, dims); never None (the matched run always has an embedder)."""
    explicit = (getattr(args, "embedder", "") or "").strip()
    admin_key = (getattr(args, "admin_key", "") or "").strip()
    default = os.environ.get("JUNE_BENCH_EMBEDDER", "").strip() or _DEFAULT_EMBEDDER

    if explicit:                                   # --embedder flag → intentional, use silently
        return _match_or_raw(explicit)
    if admin_key:                                  # --admin-key → discover from June's admin route
        return _discover_via_admin(url, admin_key) or _match_or_raw(default)
    if not _is_tty():                              # non-interactive → the matched default, no prompt
        return _match_or_raw(default)

    # Interactive: the default already matches June — just confirm, no id to type.
    print(f"\n3) Embedder for Cognee — I'll use '{default}' so Cognee embeds with the SAME model June's "
          f"dense lane uses (the same-embedder reproduction), unless you want to change it.")
    c = _ask("   [a] use it (recommended) · [b] use a different embedder id  > ", default="a").strip().lower()
    if c == "b":
        mid = _ask("   embedder model id: > ").strip()
        if mid:
            return _match_or_raw(mid)
        print("   (none given — keeping the matched default)")
    return _match_or_raw(default)


def _preflight(url: str, key: str, llm_key: str, model: str) -> list[str]:
    """Cheap checks that catch the whole failure class BEFORE any money is spent. Returns a list of
    human-readable problems (empty = good)."""
    problems: list[str] = []
    if "opus" in model.lower():                   # the $90+, aborted path
        problems.append("Cognee on Opus was offered first, cost $90+, and never finished — this matched "
                        "run uses gpt-4o on both sides. (June-Opus is a separate `reproduce` run.)")
    try:
        import cognee  # noqa: F401
    except Exception:  # noqa: BLE001
        problems.append("Cognee isn't installed — run: pip install \"june-bench[cognee]\"")
    try:
        import fastembed  # noqa: F401
    except Exception:  # noqa: BLE001
        problems.append("fastembed isn't installed (Cognee's local embedder) — reinstall with the extra: "
                        "pip install \"june-bench[cognee]\" (bundles it)")
    # OpenRouter reachable with THIS key + model — fails cheap, before any ingest.
    if llm_key:
        try:
            import httpx
            probe = model.split("/", 1)[-1] if model.startswith("openrouter/") else model
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                           headers={"Authorization": f"Bearer {llm_key}"},
                           json={"model": probe, "max_tokens": 5,
                                 "messages": [{"role": "user", "content": "ping"}]}, timeout=30.0)
            if "choices" not in r.text:
                problems.append(f"OpenRouter didn't accept {probe!r} with that key (HTTP {r.status_code}) "
                                "— check the key/model before running.")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"couldn't reach OpenRouter ({exc}) — check your connection/key.")
    return problems


def _apply_env(access: str, llm: str, model: str, embed: tuple[str, int] | None,
               cot: bool) -> None:
    """Bake the matched stack into the env both sides read — the user types none of this. June's knobs
    mirror `reproduce._apply_env`; Cognee's mirror the local head-to-head's `.env.cognee`."""
    # ── June side (open-pool, BYO answer model + key) ──
    os.environ["JUNE_BENCH_JUNE_URL"] = os.environ.get("JUNE_BENCH_JUNE_URL") or PRESET["url"]
    os.environ["JUNE_BENCH_JUNE_KEY"] = access
    os.environ["JUNE_BENCH_JUNE_POOL"] = "1"        # pools BOTH sides (Cognee reads this too) → matched task
    os.environ["JUNE_BENCH_JUNE_BACKFILL"] = "1"
    os.environ["JUNE_BENCH_LLM_KEY"] = llm
    os.environ["JUNE_BENCH_LLM_MODEL"] = model
    # ── Cognee side (runs locally; same embedder + same model) ──
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"  # local, keyless, free
    emb = embed if embed is not None else _default_fastembed()
    if emb is not None:
        # `emb` is the chosen embedder (default = June's dense-lane model, or an --embedder/--admin-key
        # override). We MUST set a fastembed-SUPPORTED model, because the provider above is forced to
        # fastembed. Cognee's OWN default is an OpenAI embedder (openai/text-embedding-3-large) —
        # incompatible with fastembed AND the keyed/paid model behind the original online failure — so
        # leaving it unset would crash; the defaulted/matched id avoids that.
        os.environ["EMBEDDING_MODEL"] = emb[0]
        os.environ["HUGGINGFACE_TOKENIZER"] = emb[0]
        os.environ["EMBEDDING_DIMENSIONS"] = str(emb[1])
    os.environ["LLM_PROVIDER"] = "custom"            # OpenRouter-compatible path (native Anthropic is broken)
    os.environ["LLM_MODEL"] = model if model.startswith("openrouter/") else f"openrouter/{model}"
    os.environ["LLM_ENDPOINT"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = llm
    os.environ["LLM_INSTRUCTOR_MODE"] = "tool_call"
    os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
    os.environ["COGNEE_SEARCH_TYPE"] = "GRAPH_COMPLETION_COT" if cot else "GRAPH_COMPLETION"
    # ── shared judge (fixed model, verbosity-agnostic) ──
    os.environ.setdefault("JUNE_JUDGE_LLM_URL", PRESET["judge_url"])
    os.environ.setdefault("JUNE_JUDGE_LLM_MODEL", PRESET["judge_model"])
    os.environ["JUNE_JUDGE_LLM_KEY"] = llm


def _resolve_inputs(args) -> tuple[str, str, bool, int]:  # noqa: ANN001
    print("\nJune vs Cognee — reproduce the head-to-head")
    print("Both systems get the SAME evidence pool, the SAME answer model, the SAME judge — only the")
    print("retrieval + reasoning engine differs. Cognee runs locally; June runs on its endpoint.\n")
    access = (getattr(args, "key", "") or os.environ.get("JUNE_BENCH_JUNE_KEY", "")
              or _ask("1) Access key from Junemind:\n   > "))
    llm = (getattr(args, "llm_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
           or _ask("2) Your OpenRouter API key  (pays for BOTH systems' answers, ~$10 for a full run;\n"
                   "   get one at https://openrouter.ai/keys):\n   > ", secret=True))
    # reasoning tier — match June's grounded multi-hop with Cognee's chain-of-thought (fair default)
    cot = getattr(args, "cot", None)
    if cot is None:
        if _is_tty():
            print("\n3) Cognee reasoning tier?  (June answers with grounded multi-hop)")
            print("   [1] chain-of-thought  — matches June's multi-hop (recommended, fair)")
            print("   [2] one-shot          — Cognee's base graph completion")
            cot = _ask("   > ", default="1") != "2"
        else:
            cot = True
    limit = getattr(args, "questions", 0) or 0
    if not limit:
        if _is_tty():
            _full = round(100 * (_COST_PER_100Q_GPT4O["june-api"] + _cognee_per_100q(cot)) / 100.0)
            print("\n4) How much to run?")
            print("   [1] Quick check   — 5 questions  (~1 min, a few cents)")
            print(f"   [2] Full headline — 100 questions (~15-20 min, ~${_full}"
                  f"{' · CoT' if cot else ''})")
            print("   [3] Custom        — enter a number")
            c = _ask("   > ", default="2") or "2"
            limit = (5 if c == "1" else 100 if c == "2"
                     else max(1, int(_ask("   how many? > ", default="100") or "100")))
        else:
            limit = 100
    if not access or not llm:
        print("\nNeed both an access key and an OpenRouter key. Re-run and provide them, or pass "
              "--key / --llm-key (or set JUNE_BENCH_JUNE_KEY / OPENROUTER_API_KEY).", file=sys.stderr)
        raise SystemExit(2)
    return access, llm, bool(cot), int(limit)


def _confirm_cost(limit: int, cot: bool) -> bool:
    if os.environ.get("JUNE_BENCH_YES") == "1" or not _is_tty():
        return True
    # Cognee dominates the bill; use the reasoning tier the user actually picked (CoT ≈ 2x base) so the
    # gate can't undershoot. Phrased as "up to" — the real, provider-metered figure is printed at the end.
    per_q = (_COST_PER_100Q_GPT4O["june-api"] + _cognee_per_100q(cot)) / 100.0
    est = ("a few cents" if limit <= 5
           else f"up to ~${limit * per_q:.2f} (both systems, gpt-4o{' · CoT' if cot else ''})")
    return _ask(f"\n→ Running {limit} questions on BOTH systems spends money: {est}. Proceed? [y/N] ",
                default="n").lower() == "y"


def run_reproduce_h2h(args) -> int:  # noqa: ANN001
    from june_bench import systems
    from june_bench.judge import judge_from_env, judge_records
    from june_bench.reproduce import _BundledHotpot
    from june_bench.runner import run as run_bench
    from june_bench.score import score

    access, llm, cot, limit = _resolve_inputs(args)
    model = getattr(args, "model", "") or _DEFAULT_MODEL
    from june_bench.reproduce import _ask_ingest_batches
    _ask_ingest_batches(limit)   # big pool? offer to batch the upload (avoids the SQLite lock storm)

    url = os.environ.get("JUNE_BENCH_JUNE_URL") or PRESET["url"]
    embed = _choose_embedder(args, url)   # default = June's embedder (matched) · --embedder · --admin-key

    # Apply the env BEFORE anything imports cognee. Cognee reads its LLM/embedder config from the
    # environment at import time and caches it — so if the key is set later (e.g. after `_preflight`,
    # which imports cognee to check it's installed), cognee's graph build raises "LLM API key is not set"
    # even though the key is present. `_apply_env` is pure (only sets os.environ), so applying it here —
    # ahead of pre-flight and the cost gate — is side-effect-free and removes the ordering hazard entirely.
    _apply_env(access, llm, model, embed, cot)

    problems = _preflight(url, access, llm, model)
    if problems:
        print("\n✗ Pre-flight found issues (fix these — no money spent yet):", file=sys.stderr)
        for p in problems:
            print(f"    • {p}", file=sys.stderr)
        return 2
    print("✓ pre-flight passed"
          + (f" · embedder matched to June's: {embed[0]}" if embed else "")
          + f" · Cognee tier: {'chain-of-thought' if cot else 'one-shot'}\n")

    if not _confirm_cost(limit, cot):
        print("aborted (no run).", file=sys.stderr)
        return 1

    dataset = _BundledHotpot()
    avail = len(dataset.load())
    if limit > avail:
        print(f"• note: the bundled reproduction set is {avail} questions — running {avail}.")
        limit = avail

    results: dict[str, dict] = {}
    actual_cost: dict[str, float] = {}        # per-system REAL $ from the OpenRouter credits delta
    prev_usage = _openrouter_usage(llm)       # baseline before the first system (None ⇒ metering off)
    jf = None if getattr(args, "no_judge", False) else judge_from_env()
    for name in ("june-api", "cognee"):
        print(f"\n• {_LABELS[name]} — {limit} questions, open-pool, {model}…")
        try:
            system = systems.get(name)
            recs = run_bench(system, dataset, split="full", limit=limit, on_progress=_progress)
        except Exception as exc:  # noqa: BLE001 — one system failing shouldn't lose the other's number
            print(f"  ! {name} failed: {exc}", file=sys.stderr)
            continue
        cur_usage = _openrouter_usage(llm)     # spend AFTER this system → delta = its real billed cost
        if prev_usage is not None and cur_usage is not None:
            actual_cost[name] = max(0.0, cur_usage - prev_usage)
        if cur_usage is not None:
            prev_usage = cur_usage             # advance the baseline only on a good reading
        summary = score(recs)
        judged = None
        if jf is not None:
            try:
                judged = judge_records(recs, jf)
            except Exception:  # noqa: BLE001
                judged = None
        results[name] = {"summary": summary, "judged": judged, "n": len(recs)}

    _print_h2h(results, model, actual_cost, cot)
    return 0


def _print_cost(results: dict, model: str, actual: dict, cot: bool = False) -> None:
    """Cost of the LLM API used, per system. Both systems bill the caller's SAME OpenRouter key (June via
    ``X-LLM-Key``, Cognee via litellm), so ``actual[name]`` — the per-phase credits delta — is the REAL
    amount OpenRouter charged this run. Falls back to the canonical metered $/100Q basis only when the
    credits API was unreachable; the Cognee basis is tier-aware (CoT ≈ 2x base). Cognee's chain-of-thought
    + one-time graph build make it far costlier per answer than June — the point of showing this."""
    print("\n  Cost — LLM API used (billed to your OpenRouter key):")
    have_ref = "gpt-4o" in model.lower()
    cog100 = _cognee_per_100q(cot)
    for name in ("june-api", "cognee"):
        r = results.get(name)
        if not r:
            continue
        n = int(r.get("n") or 0)
        ref100 = cog100 if name == "cognee" else _COST_PER_100Q_GPT4O.get(name, 0.0)
        if name in actual:                          # authoritative: OpenRouter's own number
            c = actual[name]
            note = ("server-side — not billed to your key" if (name == "june-api" and c < 0.01)
                    else "metered — OpenRouter billed this run")
            print(f"    {_LABELS[name]:22} ${c:.2f}   ({note})")
        elif have_ref:                              # credits API down → labeled, tier-aware estimate
            print(f"    {_LABELS[name]:22} ~${ref100 * n / 100.0:.2f}   "
                  f"(est. from ${ref100:.2f}/100Q metered — credits API unavailable)")
        else:
            print(f"    {_LABELS[name]:22} (cost unavailable — credits API down, no gpt-4o reference)")
    if actual.get("june-api", 0.0) > 0.01 and actual.get("cognee", 0.0) > 0.0:
        print(f"    → Cognee ≈ {actual['cognee'] / actual['june-api']:.0f}× June's API cost "
              f"(measured this run).")
    elif have_ref:
        jc = _COST_PER_100Q_GPT4O["june-api"]
        tier = " · CoT" if cot else ""
        print(f"    → Cognee ≈ {cog100 / jc:.0f}× June's API cost "
              f"(metered gpt-4o reference{tier}: ${cog100:.2f} vs ${jc:.2f} per 100 questions).")


def _print_h2h(results: dict, model: str, actual: dict | None = None, cot: bool = False) -> None:
    line = "─" * 64
    print("\n" + line)
    print(f"  June vs Cognee — matched open-pool · {model}")
    print(line)
    print(f"  {'System':22} {'EM':>6} {'F1':>6} {'judged':>8}   target")
    for name in ("june-api", "cognee"):
        r = results.get(name)
        if not r:
            print(f"  {_LABELS[name]:22} {'—':>6} {'—':>6} {'—':>8}   (did not complete)")
            continue
        s = r["summary"]
        em, f1 = s.get("em", 0.0), s.get("f1", 0.0)
        j = r["judged"]
        t = _TARGETS.get(name)
        tgt = f"~{t[0]:.2f}/{t[1]:.2f}" if t else "—"
        print(f"  {_LABELS[name]:22} {em:>6.2f} {f1:>6.2f} "
              f"{(f'{j:.0%}' if j is not None else '—'):>8}   {tgt}")
    print(line)
    ja = results.get("june-api", {}).get("summary", {}).get("em")
    ca = results.get("cognee", {}).get("summary", {}).get("em")
    if ja is not None and ca is not None:
        print(f"  Δ EM (June − Cognee): {ja - ca:+.2f}   "
              f"({'June leads' if ja > ca else 'Cognee leads' if ca > ja else 'tie'})")
    _print_cost(results, model, actual or {}, cot)
    print(line)
    print("  Both: same evidence pool · same answer model · same judge. Cognee ran locally.")


__all__ = ["run_reproduce_h2h"]
