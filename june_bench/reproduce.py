"""`june-bench reproduce` — the one-command, plain-language path for a non-expert to reproduce June's
open-pool HotpotQA number against a hosted endpoint.

Design goals:
* **No env-var incantations.** The eight `JUNE_BENCH_*` knobs are baked into a preset; the user is
  asked only for the two things nobody else can supply (an access key + their own LLM key) and how
  much to run — in plain English.
* **Moat-safe by default.** The friendly output describes *capabilities*, never model names
  (`_HOW_LINE` is asserted model-free by tests/test_no_moat_leak.py). `--show-config` surfaces the
  endpoint's effective settings as *capabilities* (lanes on/off, limits) — never model ids.
* **No secrets persisted.** Keys come from flags → env → prompt; only the non-secret run-size choice
  is remembered.
"""
from __future__ import annotations

import os
import sys

from june_bench._util import env_float, env_int

# The reproduction preset — everything a partner would otherwise export by hand. The answer MODEL is
# server-side (the endpoint decides + the caller's key pays), so it isn't set here; this is the client
# side only.
PRESET = {
    "url": "https://bench.june.januraine.ai",
    "system": "june-api",
    "dataset": "hotpot",
    "split": "full",
    "judge_url": "https://openrouter.ai/api/v1/chat/completions",
    "judge_model": "openai/gpt-4o",
    "published_em": 0.63,
    "published_f1": 0.80,
}

# Plain-language description of HOW June answers — capabilities, NOT model names (moat rule
# `moat-no-model-names`). Even `--show-config` reports capabilities (lanes, limits), never model ids.
_HOW_LINE = ("open-pool retrieval (June finds its own evidence — no answers handed in) · "
             "fused dense + lexical search · grounded multi-hop reasoning")

_RUN_SIZES = {"1": ("quick check", 5), "2": ("full headline", 100)}

# Published open-pool targets per answer model: (EM, F1, COVERAGE, serving-context).
#
# COVERAGE is part of the target, not a footnote. Both published numbers were measured over ALL
# 100 questions (abstain_rate 0.0 in the source files, chat-2026-06-16-beat-cognee/head_to_head/
# june_pool_100_*.json) — "0.63" means 63 correct of 100 ASKED. A later run that answers 72 and
# gets 62% of those right is NOT a reproduction of it, and until 2026-07-27 this file said it was:
# `_verdict` compared selective EM against an all-asked target, so a 27% real regression printed
# "✓ reproduced" six separate times during the bench-abstention investigation.
#
# SERVING-CONTEXT is part of the target too. The July 2026 investigation traced that regression to
# the serving of "openai/gpt-4o" itself drifting (provider mix + behavior — see
# THEORY_05_the-model-id-is-not-the-model.md): the model id alone is an unpinned pointer, so a
# target without "as served via WHOM, WHEN" is not a reproducibility claim. The string prints with
# the verdict so every reader sees what the baseline actually was.
_MODEL_TARGETS = {
    "openai/gpt-4o": (0.63, 0.80, 1.00, "as served via OpenRouter (OpenAI/Azure mix), 2026-06-16"),
    "anthropic/claude-opus-4-8": (0.76, 0.89, 1.00, "as served via OpenRouter, 2026-06-16"),
    # The post-drift re-baseline (2026-07-27): the new stack (uncapped FTS5+hnsw lanes) on the
    # DIRECT Anthropic API at temp 1.0 — measured twice at n=100 (second run uninterrupted):
    # EM 0.73/0.72 · F1 0.86/0.85 · coverage 0.98 both · right-per-asked 0.72/0.71. The
    # conservative member of the band is recorded. Platform-NATIVE id (Anthropic direct); the
    # OpenRouter-prefixed entry above is the pre-drift aggregator-era number, kept for history.
    "claude-opus-4-8": (0.72, 0.85, 0.98,
                        "as served by Anthropic DIRECT, 2026-07-27 · new stack · temp 1.0 · n=100 ×2"),
    # Same day, same engine, OpenAI direct: the alias that refuses 21-28/100 through the
    # aggregator refuses 5/100 served direct — and lands back at the June-16-era number
    # (0.62 right-per-asked vs the original 0.63 all-asked target). Single run; provisional
    # until repeated.
    "gpt-4o": (0.65, 0.83, 0.95,
               "as served by OpenAI DIRECT, 2026-07-27 · new stack · temp 0 · n=100 ×1 (provisional)"),
}
_DEFAULT_MODEL = "openai/gpt-4o"

# ── BYO-PLATFORM (July 2026): the serving platform is part of the experiment ─────────────────
# The serving-drift incident (THEORY_05_the-model-id-is-not-the-model.md) proved a model id alone
# is an unpinned pointer: "openai/gpt-4o" via OpenRouter was silently served by two providers whose
# behavior drifted. Letting the caller CHOOSE the platform — and stamping it on the result — turns
# that hidden variable into a controlled one. The client sends an ENUM (X-LLM-Platform), never a
# URL; the endpoint maps it to an allowlist. "openrouter" = the endpoint default (no header sent →
# byte-identical legacy behaviour). Non-default platforms are GUARDED: refused against endpoints
# that don't advertise `llm_platforms` in /v1/answer/health, because an Anthropic key fired at the
# default OpenRouter URL is 100 paid auth-error rows.

# The OpenRouter caveat — printed whenever the aggregator is chosen, because the July 2026
# investigation measured exactly what it costs June. An aggregator routes each request to
# whichever provider it picks; the mix and the serving builds change without notice, and none of
# it is pinnable or visible in the result. June answers only what its evidence supports and
# refuses the rest — so when aggregator serving drifts conservative, June's refusals rise while
# guess-style systems (no refusal channel) hide the same drift inside silently-changed guesses.
# Measured on identical engine + questions (2026-07-27): 45-55/100 right-per-asked through
# OpenRouter vs 62-72/100 served direct, both model families.
_OPENROUTER_CAVEAT = (
    "  ⚠ OpenRouter note: an aggregator serves each request from an unpinned, changing provider\n"
    "    mix. June answers only what its evidence supports — it does not gamble — so aggregator\n"
    "    serving drift shows up as honest refusals (measured 2026-07: 45-55/100 via OpenRouter vs\n"
    "    62-72/100 served direct, same engine, same questions, both model families). Fine for\n"
    "    one-key convenience + real-time cost metering; for accuracy-representative or publishable\n"
    "    numbers, pick a DIRECT platform — the result stamps whichever you choose.")

_PLATFORM_MENU = {
    "1": ("openrouter", "OpenRouter (default — aggregator; provider mix is OpenRouter's choice)",
          "https://openrouter.ai/keys"),
    "2": ("openai", "OpenAI direct", "https://platform.openai.com/api-keys"),
    "3": ("anthropic", "Anthropic direct", "https://console.anthropic.com"),
    "4": ("google", "Google AI direct", "https://aistudio.google.com/apikey"),
}
# Platform-NATIVE model ids (OpenRouter ids carry a vendor prefix; direct APIs do not). Targets in
# _MODEL_TARGETS are keyed by the id as sent, so only ids listed there show a published baseline.
_PLATFORM_MODELS = {
    "openrouter": [("openai/gpt-4o", "published ~0.63 EM"),
                   ("anthropic/claude-opus-4-8", "published ~0.76 EM")],
    "openai":     [("gpt-4o", "alias — FLOATS with vendor serving"),
                   ("gpt-4o-2024-05-13", "pinned snapshot")],
    "anthropic":  [("claude-opus-4-8", "the matched-Opus H2H design"),
                   ("claude-sonnet-4-5", "cheaper tier")],
    "google":     [("gemini-2.5-flash", "fast/cheap tier")],
}
# Judge follows the platform (same key, that platform's cheap tier) so a non-OpenRouter run never
# fires its key at OpenRouter. Judged scores stay comparable only WITHIN a platform — printed.
# Platform → native key env(s): consulted flag → JUNE_BENCH_LLM_KEY → these — so a
# non-interactive `--platform anthropic` run with ANTHROPIC_API_KEY set just works
# (2026-07-29 box re-baseline stalled 2h on this: the guard only knew OPENROUTER_API_KEY).
_PLATFORM_KEY_ENVS = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def _platform_env_key(platform):
    for e in _PLATFORM_KEY_ENVS.get(platform, ()):
        v = os.environ.get(e, "")
        if v:
            return v
    return ""


_PLATFORM_JUDGE = {
    "openai": ("https://api.openai.com/v1/chat/completions", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com/v1/chat/completions", "claude-sonnet-4-5"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "gemini-2.5-flash"),
}


class _BundledHotpot:
    """The exact 100-question HotpotQA slice behind the published number, shipped INSIDE the package
    (`datasets/fixtures/hotpot_reproduce.json`). So `reproduce` needs NO download — it works offline
    and behind restrictive networks, and is byte-identical to the parity slice. (HotpotQA is CC BY-SA
    4.0; this subset is redistributed under the same terms — see the fixture header / README.)"""
    name = "hotpot-reproduce"

    def load(self, split: str | None = None, limit: int | None = None):  # noqa: ANN001
        import json
        import pathlib

        from june_bench.datasets import loaders
        path = pathlib.Path(loaders._FIXTURES) / "hotpot_reproduce.json"
        exs = loaders._hotpot_examples(json.loads(path.read_text(encoding="utf-8")))
        return exs[:limit] if limit else exs


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(prompt: str, *, secret: bool = False, default: str = "") -> str:
    """Plain-language prompt. Falls back to a clear error in non-interactive use."""
    if not _is_tty():
        return default
    if secret:
        import getpass
        val = getpass.getpass(prompt)
    else:
        val = input(prompt)
    return val.strip() or default


# HotpotQA pools ~16 passages per question (2 gold + distractors). ~30 questions' worth of passages is a
# comfortable single write-batch for the endpoint's single-writer SQLite; more than that in one write can
# lock ('database is locked' → 500). Empirically 50q (≈491 deduped passages) in ONE write hit the lock,
# so the batch is capped at ~30 questions. The recommended batch COUNT scales with the run:
# ceil(N_questions / 30). Bigger set → more batches. Exposed here so the terminal number is computed.
_PASSAGES_PER_Q = 16
_QUESTIONS_PER_BATCH = 30


def _recommended_batches(n_questions: int) -> int:
    """dataset length ÷ (what ~30 questions take in one batch). ≥1, scales with N."""
    import math
    return max(1, math.ceil(max(0, n_questions) / _QUESTIONS_PER_BATCH))


def _ask_ingest_batches(n_questions: int) -> None:
    """For a large open-pool run, offer to split June's shared-pool UPLOAD into batches so the endpoint's
    single-writer SQLite doesn't lock ('database is locked' → 500) on a ~1.6k-passage ingest. Sets
    `JUNE_BENCH_POOL_INGEST_BATCHES`. **Output-neutral** — same pool, same answers; only the number of
    upload transactions changes. The recommendation is COMPUTED and shown: dataset size ÷ a ~30-question
    batch, so it differs by dataset size (30q→1, 50q→2, 100q→4, 300q→10…)."""
    if os.environ.get("JUNE_BENCH_POOL_INGEST_BATCHES"):   # already chosen via flag/env → respect it
        return
    rec = _recommended_batches(n_questions)
    if rec <= 1 or not _is_tty():                          # small run (≤30q) → one batch is fine, no prompt
        return
    est = n_questions * _PASSAGES_PER_Q
    print(f"\n• The shared pool for {n_questions} questions is large (~{est} passages). Uploading it in "
          f"one stream can lock the endpoint's SQLite ('database is locked').")
    print("  Splitting the upload into batches does NOT change the result (same pool, same answers) — "
          "only how the passages are uploaded.")
    ans = _ask(f"  How many batches?  [recommended {rec}  (≈{est} passages ÷ ~{_QUESTIONS_PER_BATCH}-question "
               f"batch); Enter to accept · 1 = no batching] > ", default=str(rec)).strip()
    try:
        n = max(1, int(ans))
    except ValueError:
        n = rec
    os.environ["JUNE_BENCH_POOL_INGEST_BATCHES"] = str(n)
    print(f"  → uploading the pool in {n} batch(es).\n")


def _resolve_inputs(args) -> tuple[str, str, str, int, str]:  # noqa: ANN001
    """Return (access_key, llm_key, model, limit, size_label) from flags → env → plain prompts."""
    print("\nJune × HotpotQA — reproduce the open-pool benchmark")
    print("June is given NO answers; it has to retrieve its own evidence, then answer.\n")

    access = (getattr(args, "key", "") or os.environ.get("JUNE_BENCH_JUNE_KEY", "")
              or _ask("1) Access key from Junemind:\n   > "))

    # PLATFORM before key: the key prompt depends on where the answers will be served.
    platform = ((getattr(args, "platform", "") or "")
                or os.environ.get("JUNE_BENCH_LLM_PLATFORM", "")).strip().lower()
    if not platform and _is_tty():
        print("\n2) Which platform serves the answer model?  (the platform is part of the "
              "experiment — it is stamped on the result)")
        for k, (_pid, label, _keys) in sorted(_PLATFORM_MENU.items()):
            print(f"   [{k}] {label}")
        pc = _ask("   > ", default="1") or "1"
        platform = _PLATFORM_MENU.get(pc, _PLATFORM_MENU["1"])[0]
    platform = platform or "openrouter"
    if platform not in {p for p, _l, _k in _PLATFORM_MENU.values()}:
        print(f"\nUnknown platform {platform!r} — supported: "
              f"{sorted(p for p, _l, _k in _PLATFORM_MENU.values())}", file=sys.stderr)
        raise SystemExit(2)
    os.environ["JUNE_BENCH_LLM_PLATFORM"] = platform
    if platform == "openrouter":
        print(_OPENROUTER_CAVEAT)
    _plabel = next(l for p, l, _k in _PLATFORM_MENU.values() if p == platform)
    _pkeys = next(k for p, _l, k in _PLATFORM_MENU.values() if p == platform)

    llm = (getattr(args, "llm_key", "") or os.environ.get("JUNE_BENCH_LLM_KEY", "")
           or _platform_env_key(platform)
           or _ask(f"   Your {_plabel.split(' (')[0]} API key  (generates the answers — you pay; "
                   f"get one at {_pkeys}):\n   > ", secret=True))

    # 3) MODEL — fully-open BYO: June's pipeline runs with the model you pick (any OpenRouter id).
    model = getattr(args, "model", "") or os.environ.get("JUNE_BENCH_LLM_MODEL", "")
    if not model:
        menu = _PLATFORM_MODELS.get(platform, _PLATFORM_MODELS["openrouter"])
        if _is_tty():
            print("\n3) Which answer model?  (June's retrieval + reasoning · YOUR model, end-to-end)")
            for i, (mid, note) in enumerate(menu, 1):
                print(f"   [{i}] {mid:29} ({note})")
            print(f"   [{len(menu) + 1}] other — any {_plabel.split(' (')[0]} model id")
            mc = _ask("   > ", default="1") or "1"
            if mc.isdigit() and 1 <= int(mc) <= len(menu):
                model = menu[int(mc) - 1][0]
            else:
                model = _ask("   model id: ") or menu[0][0]
        else:
            model = menu[0][0]
    model = model or _DEFAULT_MODEL

    limit = getattr(args, "questions", 0) or 0
    size_label = "custom"
    if not limit:
        if _is_tty():
            print("\n4) How much to run?")
            print("   [1] Quick check   — 5 questions (~1 min, a few cents)")
            print("   [2] Full headline — 100 questions (~10 min, ~$1–2)")
            print("   [3] Custom        — enter a number")
            choice = _ask("   > ", default="2") or "2"
            if choice == "3":
                try:
                    limit = max(1, int(_ask("   how many? > ", default="100") or "100"))
                    size_label = "custom"
                except ValueError:
                    size_label, limit = _RUN_SIZES["2"]
            else:
                size_label, limit = _RUN_SIZES.get(choice, ("full headline", 100))
        else:
            size_label, limit = _RUN_SIZES["2"]   # non-interactive default = full headline (100)

    if not access or not llm:
        _envs = " / ".join(_PLATFORM_KEY_ENVS.get(platform, ("OPENROUTER_API_KEY",)))
        print(f"\nNeed both an access key and a {platform} API key. Re-run and provide them, or pass "
              f"--key / --llm-key (or set JUNE_BENCH_JUNE_KEY / {_envs}).", file=sys.stderr)
        raise SystemExit(2)
    return access, llm, model, int(limit), size_label


def _apply_env(access: str, llm: str, model: str) -> None:
    """Bake the preset + the user's secrets/choice into the env the harness reads — so the user never
    types a single JUNE_BENCH_* variable themselves."""
    os.environ["JUNE_BENCH_JUNE_URL"] = os.environ.get("JUNE_BENCH_JUNE_URL") or PRESET["url"]
    os.environ["JUNE_BENCH_JUNE_KEY"] = access
    os.environ["JUNE_BENCH_JUNE_POOL"] = "1"        # open-pool: retrieve out of the whole corpus
    os.environ["JUNE_BENCH_JUNE_BACKFILL"] = "1"    # embed the pool so the dense lane has vectors
    os.environ["JUNE_BENCH_LLM_KEY"] = llm          # BYO — the caller pays for answer synthesis
    os.environ["JUNE_BENCH_LLM_MODEL"] = model      # BYO model → X-LLM-Model (synth + reasoner)
    # Judge follows the platform: a non-OpenRouter key must never be fired at OpenRouter's judge
    # URL (that was 100 paid auth-errors waiting to happen). Same key, that platform's cheap tier.
    platform = (os.environ.get("JUNE_BENCH_LLM_PLATFORM", "") or "openrouter").strip().lower()
    if platform in _PLATFORM_JUDGE:
        j_url, j_model = _PLATFORM_JUDGE[platform]
        os.environ.setdefault("JUNE_JUDGE_LLM_URL", j_url)
        os.environ.setdefault("JUNE_JUDGE_LLM_MODEL", j_model)
        print(f"  · judge: {j_model} on {platform} — judged-correct is comparable WITHIN a "
              "platform, not across platforms (the judge model differs)")
        print("  · cost note: metered cost is OpenRouter-specific; off-OpenRouter runs are "
              "not metered (use token counts × the vendor's price sheet)")
    os.environ.setdefault("JUNE_JUDGE_LLM_URL", PRESET["judge_url"])
    # Judge stays a FIXED model (not the answer model) so judged scores are comparable ACROSS answer
    # models and there's no self-judging bias.
    os.environ.setdefault("JUNE_JUDGE_LLM_MODEL", PRESET["judge_model"])
    os.environ["JUNE_JUDGE_LLM_KEY"] = llm


def _progress(done: int, total: int) -> None:
    width = 24
    filled = int(width * done / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  answering [{bar}] {done}/{total}")
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")


def _health(url: str, key: str) -> dict:
    """Probe the endpoint's effective config (the `--show-config` auditor detail: capabilities only).
    Thin wrapper over the shared `probe_config` (H4: one implementation, narrow except)."""
    from june_bench._util import probe_config
    return probe_config(url, key, timeout=8.0)


def run_reproduce(args) -> int:  # noqa: ANN001
    from june_bench import systems
    from june_bench.judge import judge_from_env, judge_records
    from june_bench.runner import run as run_bench
    from june_bench.score import score

    access, llm, model, limit, size_label = _resolve_inputs(args)
    _ask_ingest_batches(limit)   # big open-pool? offer to batch the upload (avoids the SQLite lock storm)
    _apply_env(access, llm, model)

    # PLATFORM CAPABILITY GUARD: a non-default platform against an endpoint that predates
    # X-LLM-Platform means the endpoint fires this key at its DEFAULT URL — e.g. an Anthropic key
    # at OpenRouter: ~100 paid auth-error rows and a run that measured nothing. The endpoint
    # advertises support via `llm_platforms` in /v1/answer/health; absent ⇒ refuse, loudly, BEFORE
    # any money moves. JUNE_BENCH_ASSUME_PLATFORM_OK=1 overrides (e.g. a dev endpoint without the
    # health field) — an explicit, recorded risk, never a silent one.
    _platform = (os.environ.get("JUNE_BENCH_LLM_PLATFORM", "") or "openrouter").strip().lower()
    if _platform != "openrouter" and os.environ.get(
            "JUNE_BENCH_ASSUME_PLATFORM_OK", "").strip() != "1":
        _cfg = _health(os.environ.get("JUNE_BENCH_JUNE_URL", ""), access)
        _plats = _cfg.get("llm_platforms") or []
        if _platform not in [str(p).lower() for p in _plats]:
            print(f"\n✗ This endpoint does not support platform {_platform!r} "
                  f"(it advertises: {_plats or 'none — it predates platform selection'}).\n"
                  "  Choose OpenRouter, point at an upgraded endpoint, or set "
                  "JUNE_BENCH_ASSUME_PLATFORM_OK=1 to override at your own cost.", file=sys.stderr)
            raise SystemExit(2)

    # No download: the exact 100-question slice ships INSIDE the package (offline-safe, byte-identical
    # to the parity run). This is what makes reproduction work for anyone, on any network.
    system = systems.get(PRESET["system"])
    dataset = _BundledHotpot()
    _avail = len(dataset.load())                # the published reproduction set (100), bundled offline
    if limit > _avail:                          # never silently cap — warn (custom-cap parity with retrieval)
        print(f"• note: HotpotQA's reproduction set is the published {_avail} questions — running {_avail}, "
              f"not {limit}. That {_avail} IS the headline number; there's no larger published reproduction "
              f"slice (the full HotpotQA dev set is a different, costlier benchmark, not this number).")
        limit, size_label = _avail, "full headline"
    print(f"\n• running the {size_label} ({limit} questions) with {model} — "
          f"June retrieves + answers each…\n")
    # Resumable: each answered question is checkpointed (predictions cost money), so a dropped run
    # resumes without re-paying. Keyed by dataset + answer model + size. Disable: JUNE_BENCH_NO_CHECKPOINT.
    import pathlib
    _slug = "".join(c if c.isalnum() else "-" for c in model) or "model"
    # PLATFORM is part of the checkpoint key (July 2026): the same native model id served by two
    # platforms is two different experiments, and a resume that silently mixed them would corrupt
    # the run. OpenRouter keeps the historical name (old checkpoints keep resuming); direct
    # platforms get their own files.
    _plat_ck = (os.environ.get("JUNE_BENCH_LLM_PLATFORM", "") or "openrouter").strip().lower()
    if _plat_ck != "openrouter":
        _slug = f"{_plat_ck}-{_slug}"
    _ckpt = None if os.environ.get("JUNE_BENCH_NO_CHECKPOINT") else str(
        pathlib.Path(os.environ.get("JUNE_BENCH_CHECKPOINT_DIR",
                     str(pathlib.Path.home() / ".cache" / "june-bench" / "checkpoints")))
        / f"qa-hotpot-{_slug}-{limit}.jsonl")
    if _ckpt and getattr(args, "fresh", False):
        try:
            os.remove(_ckpt)                                # --fresh: discard any prior progress
        except OSError:
            pass
    if _ckpt:
        print(f"• checkpoint: {_ckpt}\n"
              f"  (a dropped run resumes here on re-run; --fresh or delete the file to restart)\n")
    records = run_bench(system, dataset, split=PRESET["split"], limit=limit,
                        on_progress=_progress, checkpoint_path=_ckpt)
    summary = score(records)

    judged = None
    jf = judge_from_env()
    if jf is not None and not getattr(args, "no_judge", False):
        try:
            judged = judge_records(records, jf)
        except Exception:  # noqa: BLE001
            judged = None

    _print_result(summary, judged, model, args)
    return 0


def _verdict(em: float, cov: float, target: tuple | None) -> str:
    """Pass/fail against the published baseline, on a COMMON denominator.

    Compares right-per-asked (selective EM × coverage) against the target's right-per-asked
    (target EM × target coverage). This is the single change that would have caught the July 2026
    regression on its first run: selective EM is mathematically insensitive to abstention — a run
    that refuses half the set and nails the rest scores HIGHER selectively while answering fewer
    questions correctly. Gating on em×coverage makes abstention cost what it costs.

    A materially-below-target coverage is called out even when the product is close: the same
    right-per-asked at 70% coverage is a different system than at 100%, and the reader deciding
    whether to trust June should know which one they measured."""
    if target is None:                          # no published baseline for this model — still valid
        return "(no published baseline for this model — June's pipeline + your model)"
    t_em, _t_f1, t_cov = target[0], target[1], (target[2] if len(target) > 2 else 1.0)
    real, t_real = em * cov, t_em * t_cov
    if real >= t_real - 0.05:
        if cov < t_cov - 0.10:
            return (f"~ accuracy holds but coverage is {cov:.0%} vs the baseline's {t_cov:.0%} — "
                    "same score, different system; investigate the abstentions")
        return "✓ reproduced"
    if real >= 0.40:
        return (f"✗ NOT reproduced on a common denominator: {real:.2f} right-per-asked vs the "
                f"baseline's {t_real:.2f} (selective EM {em:.2f} at {cov:.0%} coverage)")
    return "✗ off — see the troubleshooting note below"


def _print_result(summary: dict, judged, model: str, args) -> None:  # noqa: ANN001
    em, f1 = summary.get("em", 0.0), summary.get("f1", 0.0)
    n = summary.get("answered", summary.get("n", 0))
    target = _MODEL_TARGETS.get(model)
    line = "─" * 58
    print("\n" + line)
    print(f"  Model:   {model}")
    # EM/F1 are SELECTIVE — averaged over answered items only (see score.score's docstring:
    # "paired with coverage, so abstaining trades coverage for accuracy in a measured way").
    # That pairing was computed and then not printed, so a run answering 51 of 100 and one
    # answering 100 of 100 both showed a bare "EM 0.61" and looked comparable. Measured
    # 2026-07-27: June abstained on 49/100 of the open-pool HotpotQA slice while the headline
    # moved five points. Coverage is not a footnote to these numbers, it is half of them.
    total = summary.get("n", n)
    abst = summary.get("abstained", max(0, total - n))
    cov = summary.get("coverage", (n / total) if total else 0.0)
    print(f"  Result:  EM {em:.2f} · F1 {f1:.2f}"
          + (f" · judged-correct {judged:.0%}" if judged is not None else "")
          + f"   (over {n} ANSWERED of {total})")
    print(f"  Coverage: {cov:.0%} — abstained on {abst}/{total}"
          + (f"   ⚠ EM/F1 are over the {n} it chose to answer" if abst else ""))
    if abst:
        print("           abstentions are honest refusals — June answers only what its evidence "
              "supports; it does not gamble")
    if total:
        print(f"  Answered-and-correct: {em * n / total:.2f} of every question asked")
    if target:
        t_cov = target[2] if len(target) > 2 else 1.0
        serving = target[3] if len(target) > 3 else "serving context unrecorded"
        print(f"  Target:  ~{target[0]:.2f} / ~{target[1]:.2f} at {t_cov:.0%} coverage ({serving})")
        print(f"  Verdict: {_verdict(em, cov, target)}")
    else:
        print(f"  Target:  {_verdict(em, cov, None)}")
    # The model id is a pointer, not a model (THEORY_05): stamp what this run actually used, so the
    # row stays interpretable after the serving world moves again — AND persist the same stamp as a
    # machine-readable ledger row (~/.cache/june-bench/results.jsonl): stdout scrolls away, but the
    # re-baseline, the canary, and the site all need these rows later. Append-only, best-effort.
    import datetime as _dt
    try:
        import json as _json
        import pathlib as _pl
        _row = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "model": model,
                "platform": (os.environ.get("JUNE_BENCH_LLM_PLATFORM", "") or "openrouter").lower(),
                "endpoint": os.environ.get("JUNE_BENCH_JUNE_URL", ""), "n": total,
                "answered": n, "abstained": abst, "coverage": cov,
                "em_selective": round(em, 4), "f1_selective": round(f1, 4),
                "right_per_asked": round(em * n / total, 4) if total else 0.0,
                "judged": (round(judged, 4) if judged is not None else None)}
        _led = _pl.Path(os.environ.get("JUNE_BENCH_RESULTS_LEDGER",
                        str(_pl.Path.home() / ".cache" / "june-bench" / "results.jsonl")))
        _led.parent.mkdir(parents=True, exist_ok=True)
        with open(_led, "a") as _f:
            _f.write(_json.dumps(_row) + "\n")
    except Exception:  # noqa: BLE001 — the ledger is an accessory, never load-bearing
        pass
    _plat = (os.environ.get("JUNE_BENCH_LLM_PLATFORM", "") or "openrouter").strip().lower()
    _prov = ("provider per OpenRouter routing (UNPINNED — direct-served runs measured "
             "10-17 pts higher, 2026-07)" if _plat == "openrouter"
             else f"served by {_plat} (direct)")
    print(f"  As-run:  {model} · platform={_plat} · endpoint "
          f"{os.environ.get('JUNE_BENCH_JUNE_URL', '?')} · {_dt.date.today().isoformat()} · {_prov}")
    print(f"  How:     {_HOW_LINE}")
    print(line)

    if em < 0.45 and n:
        print("\n  Troubleshooting: a very low score usually means the answer model didn't run "
              "(bad OpenRouter key) — the answers fall back to a text-extraction floor. Re-check your "
              "key. Run --show-config to see the endpoint's effective settings.")

    if getattr(args, "show_config", False):
        h = _health(os.environ.get("JUNE_BENCH_JUNE_URL", ""), os.environ.get("JUNE_BENCH_JUNE_KEY", ""))
        print("\n  [--show-config] endpoint configuration (for auditors):")
        for k in ("synthesizer", "_dense", "reason_step", "max_sources", "max_tokens", "temperature"):
            if k in h:
                print(f"    {k.lstrip('_'):12} = {h[k]}")
        if not h:
            print("    (health probe failed — endpoint unset or unreachable)")


# ── retrieval reproduction (recall@k / nDCG / MRR — how well June FINDS evidence, no answer LLM) ──────
_RETRIEVAL_MENU = {"1": "locomo", "2": "longmemeval", "3": "financebench"}
# 'full' sentinel: run the whole bundled slice. Both retrieval runners cap the effective count to the
# fixture's actual size (min(limit, total)), so this needs no per-dataset upper bound.
_FULL_SLICE = 1_000_000_000
_RETRIEVAL_HOW_HAYSTACK = ("per-query haystack — for each query June retrieves the gold within that "
                           "query's own conversation (fused graph + lexical + dense); the same scope "
                           "June's local numbers were measured at. No answers, no answer model")
_RETRIEVAL_HOW_POOL = ("cross-conversation pool — June ingests EVERY conversation's docs into one "
                       "corpus, then finds each query's evidence out of all of them (harder). "
                       "No answers, no answer model")


def _load_full_retrieval(dataset: str, want: int = _FULL_SLICE):
    """Load up to ``want`` queries from the published dataset via the streaming registry loader (reads
    `JUNE_BENCH_DATA` / the fetch cache / repo `data/`) — the loader **streams only the needed queries**
    (no whole-file parse), so a custom 100 doesn't pull all 1,986. If the data isn't present locally,
    **auto-fetch it once** (announced), then load. Returns the examples, or ``None`` if it couldn't be
    obtained (offline / fetch failed / auto-fetch disabled). Disable with ``JUNE_BENCH_NO_AUTOFETCH=1``."""
    from june_bench.datasets import registry

    def _load():
        # streams only `want` queries+docs memory-safely (ijson); _FULL_SLICE ⇒ everything.
        return registry.get(dataset).load(split="full", limit=want)

    try:
        return _load()
    except SystemExit:
        pass                       # not present locally → fall through to auto-fetch
    except Exception as exc:  # noqa: BLE001 — e.g. ijson missing; report and give up on full
        sys.stderr.write(f"[june-bench] couldn't read the full {dataset} "
                         f"({type(exc).__name__}: {exc}).\n")
        return None

    if os.environ.get("JUNE_BENCH_NO_AUTOFETCH"):
        sys.stderr.write(f"[june-bench] full {dataset} not present and auto-fetch is off — run "
                         f"`june-bench fetch --datasets {dataset}`.\n")
        return None
    try:
        from june_bench.datasets.fetch import default_data_dir, fetch
        data_dir = default_data_dir()
        sys.stderr.write(f"[june-bench] fetching the full {dataset} dataset once → {data_dir} "
                         f"(one-time download; needs network)…\n")
        results = fetch([dataset], data_dir)
        if not any(s in ("ok", "skipped") for _n, s, _m in results):
            sys.stderr.write("[june-bench] fetch failed: "
                             + "; ".join(m for _n, _s, m in results) + "\n")
            return None
        return _load()
    except Exception as exc:  # noqa: BLE001 — network / bit-rot; caller falls back to the bundled sample
        sys.stderr.write(f"[june-bench] couldn't fetch {dataset} ({type(exc).__name__}: {exc}).\n")
        return None


def _load_retrieval_examples(dataset: str, limit: int | None):
    """Load retrieval examples, choosing the source by size:

    * `full` (the menu's 'full' = MAX) **or a custom count larger than the bundled sample** → the ENTIRE
      published dataset (locomo 1,986 / longmemeval 500 / financebench 150), **auto-fetched once** if not
      already local, then sliced to the requested count.
    * `quick` / custom ≤ the bundled sample → the small BUNDLED offline slice (no download; gold
      guaranteed present)."""
    import json
    import pathlib

    from june_bench.datasets import loaders

    def _bundled():
        path = pathlib.Path(loaders._FIXTURES) / f"{dataset}_reproduce.june.json"
        if not path.exists():
            raise SystemExit(f"no bundled retrieval slice for {dataset!r} "
                             f"(expected {path.name}); choose locomo / longmemeval / financebench.")
        return loaders._june_examples(json.loads(path.read_text(encoding="utf-8")), limit=None)

    full = limit is None or limit >= _FULL_SLICE
    bundled = _bundled()
    want_full = full or (limit is not None and limit > len(bundled))

    if want_full:
        exs = _load_full_retrieval(dataset, _FULL_SLICE if full else limit)
        if exs is not None:
            if full:
                sys.stderr.write(f"[june-bench] {dataset}: FULL dataset — {len(exs)} queries "
                                 f"(the published set). Large; use quick/custom for a fast sample.\n")
                return exs
            n = min(limit, len(exs))
            sys.stderr.write(f"[june-bench] {dataset}: {n} queries from the full published set.\n")
            return exs[:n]
        # couldn't get the full set → fall back to the bundled sample, honestly labelled
        if full:
            sys.stderr.write(f"[june-bench] using the {len(bundled)}-query bundled sample "
                             f"(run `june-bench fetch --datasets {dataset}` for the full set).\n")
            return bundled
        sys.stderr.write(f"[june-bench] custom {limit} needs the full dataset, which isn't available "
                         f"(offline?) — running the {len(bundled)}-query bundled sample instead.\n")
        return bundled

    return bundled[:limit]         # custom ≤ bundled → offline slice, capped


def _resolve_retrieval_inputs(args):  # noqa: ANN001
    """Resolve (access, dataset, limit) and — interactively, in plain language — the measurement mode
    (`args.turns`) and lanes (`args.dense`). Idempotent: the second call (the turns path re-invokes it)
    finds `_ret_resolved` and returns without re-prompting. Moat-safe: describes capabilities, never
    model names (asserted by tests/test_no_moat_leak.py)."""
    if not getattr(args, "_ret_resolved", False):
        print("\nJune — retrieval benchmark (how well June FINDS the right evidence)")
        print("June retrieves each question's evidence out of a conversation — no answers, no answer model.\n")
        access = (getattr(args, "key", "") or os.environ.get("JUNE_BENCH_JUNE_KEY", "")
                  or _ask("1) Access key from Junemind:\n   > "))
        dataset = getattr(args, "dataset", "") or os.environ.get("JUNE_BENCH_DATASET", "")
        if not dataset:
            if _is_tty():
                print("\n2) Which dataset?")
                print("   [1] locomo        (long conversations)")
                print("   [2] longmemeval   (long-term memory)")
                print("   [3] financebench  (financial docs)")
                dataset = _RETRIEVAL_MENU.get(_ask("   > ", default="1") or "1", "locomo")
            else:
                dataset = "locomo"
        if not access:
            print("\nNeed an access key. Re-run with --key or set JUNE_BENCH_JUNE_KEY.", file=sys.stderr)
            raise SystemExit(2)
        args.key, args.dataset = access, dataset

        # 3) Measurement mode (turn-grain is the faithful, locomo-only reproduction). Only ask if the
        #    user hasn't already forced it with a flag (--turns leaves it True, absent leaves it None).
        if getattr(args, "turns", None) is None:
            if _is_tty() and dataset == "locomo":
                print("\n3) Which measurement?")
                print("   [1] quick     — find the right session within a conversation (fast)")
                print("   [2] faithful  — turn-level, reproduces June's published LoCoMo numbers "
                      "(recommended)")
                args.turns = (_ask("   > ", default="2") or "2") == "2"
            else:
                args.turns = False

        # 4) Lanes: lexical-only (fast) vs + the semantic lane (slower, the fused number).
        if getattr(args, "dense", None) is None:
            if _is_tty():
                print("\n4) Which lanes?")
                print("   [1] lexical only     — fast; carries most of the recall")
                print("   [2] + semantic lane  — slower, fuller (the fused number)")
                args.dense = (_ask("   > ", default="1") or "1") == "2"
            else:
                args.dense = False

        # 5) How many questions? A preset menu (no number to guess); 'full' = the whole bundled slice.
        if not getattr(args, "questions", 0):
            if _is_tty():
                print("\n5) How many questions?")
                print("   [1] quick   — 20 (fast, bundled offline sample)")
                print("   [2] full    — the MAX available: the entire dataset where present "
                      "(1,986 / 500 / 150 — large; turn-grain uses its 149-Q slice)")
                print("   [3] custom  — enter a number (≤20 offline; >20 fetches the full set once)")
                pick = _ask("   > ", default="2") or "2"
                if pick == "1":
                    args.questions = 20
                elif pick == "3":
                    try:
                        args.questions = max(1, int(_ask("   how many? > ", default="20") or "20"))
                    except ValueError:
                        args.questions = 0
                else:  # full — sentinel; the runner caps it to the fixture's actual size
                    args.questions = _FULL_SLICE
        args._ret_resolved = True

    access = getattr(args, "key", "") or os.environ.get("JUNE_BENCH_JUNE_KEY", "")
    dataset = getattr(args, "dataset", "") or "locomo"
    # 'full' (or no choice) → run the whole bundled slice; both runners cap to the fixture's real size.
    limit = getattr(args, "questions", 0) or _FULL_SLICE
    return access, dataset, int(limit)


def run_reproduce_retrieval(args) -> int:  # noqa: ANN001
    from june_bench.retrieval import score_retrieval
    from june_bench.systems import june_retrieval

    access, dataset, limit = _resolve_retrieval_inputs(args)
    # Faithful turn-grain path is chosen interactively (menu) or by --turns → delegate.
    if getattr(args, "turns", False):
        return run_reproduce_retrieval_turns(args)
    # Retrieval needs NO answer LLM — just an access key + the endpoint's lanes.
    os.environ["JUNE_BENCH_JUNE_URL"] = os.environ.get("JUNE_BENCH_JUNE_URL") or PRESET["url"]
    os.environ["JUNE_BENCH_JUNE_KEY"] = access

    examples = _load_retrieval_examples(dataset, limit)
    diagnose = bool(getattr(args, "diagnose", False))
    pooled = bool(getattr(args, "pooled", False))
    deep = bool(getattr(args, "deep", False))
    dense = bool(getattr(args, "dense", False))
    # Backfill (embedding every ingested doc) engages the DENSE lane — but on conversational data the
    # SPARSE lane carries recall (dense is a small fused add, and inert on the lean bench). Embedding is
    # also the slow, lock-prone step. So DEFAULT = no backfill (sparse-driven, fast); `--dense` opts in.
    if dense or pooled:
        os.environ["JUNE_BENCH_JUNE_BACKFILL"] = "1"
    else:
        os.environ.pop("JUNE_BENCH_JUNE_BACKFILL", None)
    # DEFAULT = per-query HAYSTACK, no rerank — the SAME scope + config June's local champion was
    # measured at (retrieve each query's gold within its own conversation; fused, no rerank). This is
    # the apples-to-apples reproduction. `--pooled` is the harder cross-conversation variant (retrieve
    # out of every conversation's docs at once); `--deep` opts into reranking, which the local
    # diagnosis measured NET-NEGATIVE on this data (a reranker demotes an already-healthy top-5).
    if pooled:
        os.environ["JUNE_BENCH_RETRIEVAL_POOL"] = "1"
    else:
        os.environ.pop("JUNE_BENCH_RETRIEVAL_POOL", None)
    # k: with a per-query haystack (~a dozen docs) k=50 simply returns them all; pooled needs the
    # deeper candidate pool. Rerank fires only when --deep is set (and a reranker is wired server-side).
    k = 50
    if deep and not diagnose:
        os.environ["JUNE_BENCH_RETRIEVAL_DEEP"] = "1"
    else:
        os.environ.pop("JUNE_BENCH_RETRIEVAL_DEEP", None)
    _scope = ("cross-conversation POOL" if pooled else "per-query haystack (matches local)")
    print(f"• {dataset}: {_scope}; retrieving evidence for {len(examples)} queries"
          f"{' with rerank (deep)' if deep else ''}…\n")
    # Resumable: every completed query is checkpointed, so a dropped run (laptop sleep, network change,
    # endpoint blip) resumes from where it stopped instead of restarting. Keyed by the run's config so a
    # different lane/scope/size gets its own file. Disable with JUNE_BENCH_NO_CHECKPOINT=1.
    import pathlib
    _size = "full" if limit >= _FULL_SLICE else str(len(examples))
    _ckpt = None if os.environ.get("JUNE_BENCH_NO_CHECKPOINT") else str(
        pathlib.Path(os.environ.get("JUNE_BENCH_CHECKPOINT_DIR",
                     str(pathlib.Path.home() / ".cache" / "june-bench" / "checkpoints")))
        / f"ret-{dataset}-{'dense' if dense else 'lex'}-{'pool' if pooled else 'hay'}-{_size}.jsonl")
    if _ckpt and getattr(args, "fresh", False):
        try:
            os.remove(_ckpt)                                # --fresh: discard any prior progress
        except OSError:
            pass
    if _ckpt:
        print(f"• checkpoint: {_ckpt}\n"
              f"  (a dropped run resumes here on re-run; --fresh or delete the file to restart)\n")
    system = june_retrieval.from_env(k=k)
    rankings, golds = june_retrieval.run_retrieval(system, examples, on_progress=_retrieval_progress,
                                                   checkpoint_path=_ckpt)
    summary = score_retrieval(rankings, golds, ks=(1, 5, 10), turn_grain=True)
    _print_retrieval_result(summary, dataset, args)
    if diagnose:
        _diagnose_retrieval(rankings, golds)
    return 0


def _retrieval_progress(done: int, total: int) -> None:
    """In-place progress bar on stderr (so it doesn't pollute piped stdout results)."""
    if not total:
        return
    width = 24
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  retrieving [{bar}] {done}/{total}")
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")


def run_reproduce_retrieval_turns(args) -> int:  # noqa: ANN001
    """FAITHFUL turn-grain reproduction of June's local LoCoMo champion (fused R@5 0.856 / R@10 0.925).

    The default `reproduce-retrieval` scores at SESSION grain (retrieve the right whole session out of a
    conversation's ~19) — easy, and NOT the task the local number was measured at. The local champion was
    TURN grain: each session is split into turns, June retrieves over the conversation's ~380 turn-chunks,
    and a session counts as found if ANY of its turns is retrieved (session-MAX). This reproduces exactly
    that: per conversation it ingests every turn ONCE (so each query's haystack is its own conversation's
    turns — matching local — and the box does one ingest per conversation, not per query → far less lock
    pressure), searches each query, and scores session-MAX via the harness's `turn_grain=True` collapse.
    Sparse-only by default (BM25 carries; `--dense` adds the embedded lane)."""
    import asyncio
    import json
    import pathlib
    import sys as _sys
    import uuid

    from june_bench.datasets import loaders
    from june_bench.retrieval import score_retrieval
    from june_bench.systems.june_retrieval import JuneRetrievalSystem

    access, _dataset, limit = _resolve_retrieval_inputs(args)   # dataset ignored; this path is locomo-turns
    url = os.environ.get("JUNE_BENCH_JUNE_URL") or PRESET["url"]
    dense = bool(getattr(args, "dense", False))
    fx = pathlib.Path(loaders._FIXTURES) / "locomo_turns_reproduce.june.json"
    if not fx.exists():
        raise SystemExit(f"missing turn-grain fixture {fx.name}")
    convs = json.loads(fx.read_text(encoding="utf-8"))["conversations"]
    avg_turns = sum(len(c["docs"]) for c in convs) // max(1, len(convs))
    print(f"• locomo (TURN grain, session-MAX): each query retrieves over its own conversation's "
          f"~{avg_turns} turn-chunks — the SAME task as June's local champion (fused R@5 0.856 / "
          f"R@10 0.925). {'Fused + dense' if dense else 'Sparse-only'}, no rerank.\n")

    system = JuneRetrievalSystem(url, api_key=access, k=50, isolate=True, backfill=dense, cleanup=True)

    cats: dict[str, str] = {}
    total_q = min(limit, sum(len(c["queries"]) for c in convs))

    async def _go():
        rankings: dict[str, list[str]] = {}
        golds: dict[str, list[str]] = {}
        count = errors = 0
        # Dense-lane telemetry: backfill embeds each conversation's turns so the dense lane has vectors.
        # It is fail-soft (a lock/4xx is swallowed so sparse still ranks) — which can silently degrade
        # `--dense` to sparse. Count outcomes so the result line can say whether dense ACTUALLY engaged.
        bf_embedded = bf_ok = bf_fail = 0
        try:
            for c in convs:
                if count >= limit:
                    break
                docs = [(d["id"], d["text"]) for d in c["docs"]]
                canvas = await system._create_canvas(f"turns-{c['conv_id']}")
                headers = {"X-Canvas": canvas}
                ns = await system._ingest_docs(docs, headers)          # ONE ingest per conversation
                nid_to_doc = {str(uuid.uuid5(ns, i)): i for i, _ in docs}
                if dense:
                    # Backfill embeds every turn-chunk (hundreds) through the embedder on CPU — this is
                    # SLOW and must not inherit the default per-request timeout, or the client gives up
                    # while the server is still embedding (→ dense silently never engages). Give it a
                    # generous window (override via JUNE_BENCH_BACKFILL_TIMEOUT).
                    bf_timeout = env_float("JUNE_BENCH_BACKFILL_TIMEOUT", 900.0, lo=1.0)
                    _sys.stderr.write(f"\r  embedding {len(docs)} turn-chunks for {c['conv_id']} "
                                      f"(dense lane, first pass is slow)…\n")
                    # Retry on the single-writer SQLite lock: the backfill's `DELETE FROM node_embeddings`
                    # upsert hits 'database is locked' when the PREVIOUS conversation's write hasn't
                    # settled (busy_timeout doesn't cover write-write conflicts). A short backoff lets the
                    # WAL checkpoint, then the write succeeds. Attempts/backoff tunable via env.
                    bf_attempts = env_int("JUNE_BENCH_BACKFILL_RETRIES", 5, lo=1, hi=100)
                    last_exc: Exception | None = None
                    for _bfa in range(bf_attempts):
                        try:
                            rb = await system._client.post(system._backfill_path, json={}, headers=headers,
                                                           timeout=bf_timeout)
                            rb.raise_for_status()
                            bf_embedded += int(rb.json().get("embedded", 0))
                            bf_ok += 1
                            last_exc = None
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_exc = exc
                            if _bfa < bf_attempts - 1:
                                await asyncio.sleep(min(3.0 * (_bfa + 1), 15.0))  # let the WAL settle
                    if last_exc is not None:
                        bf_fail += 1
                        _sys.stderr.write(f"\n[june-bench] backfill failed for {c['conv_id']} after "
                                          f"{bf_attempts} tries: {type(last_exc).__name__}: {last_exc}\n")
                for q in c["queries"]:
                    if count >= limit:
                        break
                    golds[q["id"]] = list(q["gold_turns"])             # turn-ids; collapsed by turn_grain
                    cats[q["id"]] = q.get("question_type", "?")
                    try:
                        r = await system._client.post(system._search_path, headers=headers,
                                                      json={"query": q["query"], "limit": system._k})
                        r.raise_for_status()
                        ranked = [nid_to_doc.get(str(it.get("node_id")))
                                  for it in r.json().get("items", [])]
                        rankings[q["id"]] = [x for x in ranked if x]
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        rankings[q["id"]] = []
                        _sys.stderr.write(f"\n[june-bench] turns search errored qid={q['id']}: "
                                          f"{type(exc).__name__}: {exc}\n")
                    count += 1
                    _retrieval_progress(count, total_q)
                # NOTE: we do NOT delete the canvas mid-run. Each canvas is a workspace-isolated
                # scratch space, so leaving it is harmless — and the delete triggers `clear_workspace`
                # (a bulk DELETE on knowledge_edges/nodes) which, right after a dense backfill's WAL
                # writes, hits SQLite "database is locked", 500s, and closes the keep-alive connection —
                # which then breaks the NEXT conversation's create_canvas. Skipping cleanup is what
                # lets a --dense turn-grain run complete on a single-writer SQLite endpoint. Wipe the
                # bench DB volume between runs for a clean slate (see REPRODUCE.md).
                if os.environ.get("JUNE_BENCH_TURNS_CLEANUP", "").strip().lower() in ("1", "true", "yes"):
                    try:
                        await system._client.delete(f"{system._canvas_path}/{canvas}", headers=headers)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            await system.aclose()
        return rankings, golds, errors, (bf_embedded, bf_ok, bf_fail)

    rankings, golds, errors, (bf_embedded, bf_ok, bf_fail) = asyncio.run(_go())
    summary = score_retrieval(rankings, golds, ks=(1, 5, 10), turn_grain=True)
    r, nd = summary.get("recall", {}), summary.get("ndcg", {})
    line = "─" * 58
    print("\n" + line)
    print(f"  Dataset: locomo (TURN grain, session-MAX)   (n={summary.get('n_queries', 0)} queries)")
    print(f"  Recall:  @1 {r.get(1,0):.2f} · @5 {r.get(5,0):.2f} · @10 {r.get(10,0):.2f}")
    print(f"  nDCG:    @10 {nd.get(10,0):.2f}    MRR: {summary.get('mrr',0):.2f}")
    if dense:
        # Make the dense lane's real state VISIBLE: --dense only helps if backfill actually embedded
        # the turns. A silent backfill failure (lock / no embedder) degrades to sparse — say so.
        if bf_ok and not bf_fail:
            print(f"  Dense:   engaged — backfilled {bf_embedded} vectors across {bf_ok} conversations")
        elif bf_ok:
            print(f"  Dense:   ⚠ PARTIAL — backfill succeeded on {bf_ok}/{bf_ok + bf_fail} conversations "
                  f"({bf_embedded} vectors); the rest fell back to sparse (lock pressure)")
        else:
            print(f"  Dense:   ⚠ NOT engaged — backfill FAILED on all {bf_fail} conversations "
                  f"(embedder unwired or SQLite lock). This ran SPARSE-ONLY despite --dense; check "
                  f"/v1/embeddings/health and restart the bench app.")
    # Per-category (LoCoMo cats differ in difficulty: cat4 single-hop easy, cat1 multi-hop hard). The
    # local per-cat rates are the fair comparison — an aggregate depends on the category MIX sampled.
    order = sorted({v for v in cats.values()})
    if order:
        print("  By type:")
        for ct in order:
            ids = [q for q, v in cats.items() if v == ct]
            sub = score_retrieval({i: rankings.get(i, []) for i in ids},
                                  {i: golds[i] for i in ids}, ks=(1, 5, 10), turn_grain=True)
            sr = sub.get("recall", {})
            print(f"     {ct:5} (n={sub.get('n_queries',0):>2})  R@5 {sr.get(5,0):.2f}  R@10 {sr.get(10,0):.2f}")
    print("  Local:   sparse turn-grain R@10 0.888 · fused R@10 0.925 (with the dense lane) — the target")
    print(f"  How:     TURN grain, session-MAX — sessions split into turns; a session is 'found' if ANY "
          f"of its turns is retrieved. The exact task June's local numbers were measured at. "
          f"{'Fused+dense' if dense else 'Sparse-only'}, no rerank")
    print(line)
    if errors:
        print(f"\n  ⚠ {errors} queries errored (scored no-hit) — likely SQLite lock pressure; re-run or "
              f"restart the bench app.")
    return 0


def _diagnose_retrieval(rankings, golds) -> None:  # noqa: ANN001
    """Split the loss into RETRIEVAL vs RANKING (mirrors the local locomo diagnosis). For each query,
    find the rank of the first gold in the deep (k=50) list, collapsed to turn-grain parents:
      * gold within top-10  → counted by recall@10 already
      * gold in 11..50       → RANKING loss (a reranker promotes it — the fix)
      * gold absent from 50  → RETRIEVAL loss (a lane didn't surface it — the dense lane/candidates)"""
    from june_bench.retrieval import collapse_gold, collapse_turns
    in10 = in50 = absent = scored = 0
    for qid, gold in golds.items():
        if not gold:
            continue
        scored += 1
        ranked = collapse_turns(list(rankings.get(qid, [])))
        gset = collapse_gold(gold)
        rank = next((i + 1 for i, d in enumerate(ranked) if d in gset), None)
        if rank is None:
            absent += 1
        elif rank <= 10:
            in10 += 1
        else:
            in50 += 1
    line = "─" * 58
    print("\n" + line)
    print("  DIAGNOSIS (where the gold lands, deep k=50, turn-grain)")
    print(f"    in top-10   : {in10}/{scored}   (already found — recall@10)")
    print(f"    in 11..50   : {in50}/{scored}   (RANKING loss → a reranker recovers these)")
    print(f"    absent < 50 : {absent}/{scored}  (RETRIEVAL loss → a lane isn't surfacing them)")
    print(line)
    if scored:
        if absent > in50:
            print("  → Verdict: RETRIEVAL-bound. The gold isn't even in the top-50 for most misses — a "
                  "lane isn't engaging (check the endpoint's dense lane / embedder / backfill, or the "
                  "candidate cap). A reranker CANNOT fix this.")
        elif in50 >= max(1, absent):
            print("  → Verdict: RANKING-bound. The gold IS retrieved (in top-50) but sits below 10 — this "
                  "is exactly what a reranker fixes (your local diagnosis's finding).")


def _print_retrieval_result(summary: dict, dataset: str, args) -> None:  # noqa: ANN001
    r, nd = summary.get("recall", {}), summary.get("ndcg", {})
    n = summary.get("n_queries", 0)
    line = "─" * 58
    print("\n" + line)
    print(f"  Dataset: {dataset}   (n={n} queries)")
    print(f"  Recall:  @1 {r.get(1,0):.2f} · @5 {r.get(5,0):.2f} · @10 {r.get(10,0):.2f}")
    print(f"  nDCG:    @10 {nd.get(10,0):.2f}    MRR: {summary.get('mrr',0):.2f}")
    how = _RETRIEVAL_HOW_POOL if getattr(args, "pooled", False) else _RETRIEVAL_HOW_HAYSTACK
    print(f"  How:     {how}")
    print(line)
    if n and r.get(5, 0) == 0.0:
        print("\n  Recall 0 usually means the dense lane isn't engaged on the endpoint (embedder=none) "
              "or backfill didn't run — tell the host to check /v1/embeddings/health.")
    if getattr(args, "show_config", False):
        print(f"\n  [--show-config] pooled retrieval · fused lanes · k=10 · turn-grain · "
              f"n_queries={n}")


__all__ = ["PRESET", "run_reproduce", "run_reproduce_retrieval"]
