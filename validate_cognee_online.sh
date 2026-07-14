#!/usr/bin/env bash
# validate_cognee_online.sh — one-shot Cognee online-validation (closes FL-SB4).
#
# Bakes the COGNEE_ONLINE_VALIDATION.md runbook into a single command: sets the matched local stack
# (fastembed bge-large + OpenRouter-Opus, identical to the in-repo head-to-head), runs the PRE-FLIGHT
# guard that aborts on the exact `EMBEDDING_PROVIDER != fastembed` condition that caused the past 422 +
# OpenAI billing, then a cheap 2-question SMOKE. The paid full/H2H runs are OPT-IN and confirm-gated so
# nothing spends money by accident.
#
# Cognee runs LOCALLY in this process (it never hits the June AWS box) — needs ~1.3GB for fastembed
# bge-large + cognee's file stores. Run on your Mac, not the RAM-tight bench box.
#
# FAIRNESS + COST (from FAIRNESS_AND_METHODOLOGY): the matched row is gpt-4o-vs-gpt-4o with bge-large on
# BOTH sides — so Cognee defaults to gpt-4o here, NOT Opus. Cognee on Opus was offered first, cost $90+,
# and NEVER completed (aborted) — gpt-4o is the charitable, metered fallback (~$10.64/100Q). This script
# refuses an Opus Cognee run unless you force it. June-Opus is reported SEPARATELY as the efficiency
# ceiling ($2.57/100Q), not as a matched row.
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...                 # required
#   bash validate_cognee_online.sh                      # pre-flight + smoke (safe, cheap)
#   bash validate_cognee_online.sh full                 # + Cognee-only full 100Q gpt-4o (~$10.64, confirm-gated)
#   bash validate_cognee_online.sh h2h                  # + matched June-vs-Cognee 100Q, gpt-4o both (confirm-gated)
# Env overrides (all optional): LIMIT (full run, default 100), COGNEE_SEARCH_TYPE (GRAPH_COMPLETION|_COT
#   — use _COT to match June's multi-hop reasoner), LLM_MODEL (default openrouter/openai/gpt-4o),
#   YES=1 (skip the paid-run confirmation), JUNE_BENCH_JUNE_URL / JUNE_BENCH_JUNE_KEY (required for `h2h`).
set -euo pipefail

MODE="${1:-smoke}"
LIMIT="${LIMIT:-100}"

say()  { printf '\n\033[1m• %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓ %s\033[0m\n' "$*"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0 · prerequisites ────────────────────────────────────────────────────────
[[ -n "${OPENROUTER_API_KEY:-}" ]] || die "set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=sk-or-...)"
command -v june-bench >/dev/null 2>&1 || die "june-bench not on PATH — pip install \"june-bench[cognee,june-api]==0.0.20\""

# ── 1 · the matched local Cognee stack (identical to .env.cognee) ────────────
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-fastembed}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
export EMBEDDING_DIMENSIONS="${EMBEDDING_DIMENSIONS:-1024}"
export HUGGINGFACE_TOKENIZER="${HUGGINGFACE_TOKENIZER:-BAAI/bge-large-en-v1.5}"
export LLM_PROVIDER="${LLM_PROVIDER:-custom}"
export LLM_MODEL="${LLM_MODEL:-openrouter/openai/gpt-4o}"   # matched row = gpt-4o (Cognee-Opus was $90+, aborted)
export LLM_ENDPOINT="${LLM_ENDPOINT:-https://openrouter.ai/api/v1}"
export LLM_API_KEY="${LLM_API_KEY:-$OPENROUTER_API_KEY}"
export LLM_INSTRUCTOR_MODE="${LLM_INSTRUCTOR_MODE:-tool_call}"
export COGNEE_SKIP_CONNECTION_TEST="${COGNEE_SKIP_CONNECTION_TEST:-true}"
export COGNEE_SEARCH_TYPE="${COGNEE_SEARCH_TYPE:-GRAPH_COMPLETION}"
# harness judge (verbosity-agnostic ruler) — same key is fine
export JUNE_JUDGE_LLM_URL="${JUNE_JUDGE_LLM_URL:-https://openrouter.ai/api/v1/chat/completions}"
export JUNE_JUDGE_LLM_MODEL="${JUNE_JUDGE_LLM_MODEL:-openai/gpt-4o}"
export JUNE_JUDGE_LLM_KEY="${JUNE_JUDGE_LLM_KEY:-$OPENROUTER_API_KEY}"

# ── 2 · PRE-FLIGHT (the guards that would have caught the past failure) ───────
say "pre-flight"
# (a) the exact past bug: embedder must be fastembed (keyless/free), never openai (keyed/422)
[[ "$EMBEDDING_PROVIDER" == "fastembed" ]] \
  || die "EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER — set it to fastembed or you'll 422 + get billed (the past failure)"
ok "embedder = fastembed ($EMBEDDING_MODEL, keyless/free — same as June's dense lane)"

# (a2) block the $90 aborted path: Cognee on Opus. Force with COGNEE_ALLOW_OPUS=1 if you truly mean it.
if printf '%s' "$LLM_MODEL" | grep -qi 'opus'; then
  [[ "${COGNEE_ALLOW_OPUS:-}" == "1" ]] \
    || die "LLM_MODEL=$LLM_MODEL puts Cognee on Opus — that was offered first, cost \$90+, and NEVER finished (aborted). Use gpt-4o (the metered, charitable fallback), or set COGNEE_ALLOW_OPUS=1 to override."
  printf '  \033[33m! COGNEE_ALLOW_OPUS=1 — running Cognee on Opus anyway (last time: \$90+, aborted)\033[0m\n'
fi
ok "Cognee LLM = $LLM_MODEL"

# (b) fastembed importable + the bge-large model is served
python3 - <<PY || die "fastembed can't serve $EMBEDDING_MODEL — pip install fastembed"
from fastembed import TextEmbedding
m = "$EMBEDDING_MODEL"
assert any(e["model"] == m for e in TextEmbedding.list_supported_models()), m
print("  \033[32m✓ fastembed can serve", m, "(first run downloads ~1.3GB)\033[0m")
PY

# (c) OpenRouter LLM reachable with THIS key+model — fails cheap, before any ingest
_llm_probe_model="${LLM_MODEL#openrouter/}"
if curl -sS --max-time 30 https://openrouter.ai/api/v1/chat/completions \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" \
      -d "{\"model\":\"$_llm_probe_model\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
      2>/dev/null | grep -q '"choices"'; then
  ok "OpenRouter reachable for $_llm_probe_model"
else
  die "OpenRouter/key/model problem for $_llm_probe_model — fix before running (no money spent yet)"
fi

# (d) for h2h, the June endpoint must be configured
if [[ "$MODE" == "h2h" ]]; then
  [[ -n "${JUNE_BENCH_JUNE_URL:-}" && -n "${JUNE_BENCH_JUNE_KEY:-}" ]] \
    || die "h2h needs JUNE_BENCH_JUNE_URL + JUNE_BENCH_JUNE_KEY (the June side)"
  export JUNE_BENCH_JUNE_POOL="${JUNE_BENCH_JUNE_POOL:-1}"
  export JUNE_BENCH_JUNE_BACKFILL="${JUNE_BENCH_JUNE_BACKFILL:-1}"
  ok "June endpoint set ($JUNE_BENCH_JUNE_URL, pool+backfill on)"
fi

# ── 3 · SMOKE (2 questions, ~cents) — always run; proves the path before any full run ──
say "smoke — 2 questions (cheap proof the stack answers, no 422)"
june-bench suite --systems cognee --datasets hotpot --split smoke --limit 2 --judge \
  --model "cognee-smoke" --out RESULTS_cognee_online_smoke.md
echo; cat RESULTS_cognee_online_smoke.md; echo
ok "smoke complete → RESULTS_cognee_online_smoke.md (EM noisy at n=2; F1/judge should be > 0)"

[[ "$MODE" == "smoke" ]] && { say "done (smoke only). Re-run with 'full' or 'h2h' for the paid n=$LIMIT run."; exit 0; }

# ── 4 · confirm before spending on the full run ──────────────────────────────
if [[ "${YES:-}" != "1" ]]; then
  _est="~\$10.64 (Cognee-gpt4o metered, 100Q)"; printf '%s' "$LLM_MODEL" | grep -qi opus && _est="\$90+ (Opus — aborted last time!)"
  printf '\n\033[33m! The %s run answers %s questions with %s — this SPENDS money: %s.\033[0m\n' "$MODE" "$LIMIT" "$LLM_MODEL" "$_est"
  read -r -p "  proceed? [y/N] " a; [[ "$a" == "y" || "$a" == "Y" ]] || die "aborted (no full run)"
fi

# ── 5 · the paid run ─────────────────────────────────────────────────────────
if [[ "$MODE" == "full" ]]; then
  say "full — Cognee only, $LIMIT questions"
  june-bench suite --systems cognee --datasets hotpot --split full --limit "$LIMIT" --judge \
    --model "cognee-opus-n$LIMIT" --out "RESULTS_cognee_online_n$LIMIT.md"
  echo; cat "RESULTS_cognee_online_n$LIMIT.md"
  ok "→ RESULTS_cognee_online_n$LIMIT.md"
elif [[ "$MODE" == "h2h" ]]; then
  say "h2h — June endpoint vs local Cognee, $LIMIT questions, matched open-pool + same judge"
  export JUNE_BENCH_JUNE_POOL="${JUNE_BENCH_JUNE_POOL:-1}"   # 0.0.18: pools BOTH sides (matched task)
  june-bench suite --systems june-api,cognee --datasets hotpot --split full --limit "$LIMIT" --judge \
    --model "opus-h2h-n$LIMIT" --out "RESULTS_h2h_online_n$LIMIT.md"
  echo; cat "RESULTS_h2h_online_n$LIMIT.md"
  ok "→ RESULTS_h2h_online_n$LIMIT.md  (matched gpt-4o target: June-gpt4o ~0.63/0.80 vs Cognee-CoT ~0.530/0.658; June-Opus 0.760/0.894 is the SEPARATE efficiency ceiling)"
else
  die "unknown mode '$MODE' — use: smoke | full | h2h"
fi

say "done. Commit the RESULTS_*.md next to the local head-to-head to close FL-SB4 with a dated, matched number."
