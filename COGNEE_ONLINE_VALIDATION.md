# Cognee online-validation runbook (close FL-SB4)

Goal: run the online Cognee head-to-head through the **pip package** (`june-bench --system cognee`) and
prove it reproduces the local head-to-head — *without* re-triggering the failure that sank the first
online attempt (an OpenAI-embeddings **422** that a list-comprehension turned into a whole-run crash, and
that quietly billed OpenAI).

## The one thing that matters

**Cognee runs LOCALLY, in the client process — it does NOT hit the June AWS endpoint.** It builds its own
KG with a local embedder and calls an LLM directly. So the online-vs-local distinction for *Cognee* is only
"driven through the harness" vs "driven through the in-repo `cognee_h2h_driver.py`". The stack is
byte-identical **iff you set the same env**. The first online run failed for exactly one reason: the
**embedder env drifted** from local `fastembed / bge-large` (keyless, free) to `OpenAI text-embedding-3-large`
(keyed, billed) → 422.

The matched stack (copy of `chat-2026-06-16-beat-cognee/head_to_head/.env.cognee.example`):

| Knob | Value | Why |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fastembed` | local, keyless, free — **the load-bearing fix**. NOT `openai`. |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | same model June's dense lane uses → fair |
| `EMBEDDING_DIMENSIONS` | `1024` | bge-large is 1024-dim |
| `LLM_PROVIDER` | `custom` | cognee 1.1.2's native Anthropic adapter is broken → use OpenAI-compatible path |
| `LLM_MODEL` | `openrouter/openai/gpt-4o` | **the matched row is gpt-4o-vs-gpt-4o.** NOT Opus — see fairness note below |
| `LLM_ENDPOINT` | `https://openrouter.ai/api/v1` | OpenRouter speaks OpenAI's protocol |
| `LLM_API_KEY` | `$OPENROUTER_API_KEY` | your key |
| `LLM_INSTRUCTOR_MODE` | `tool_call` | OpenRouter accepts OpenAI `tool_choice` |
| `COGNEE_SKIP_CONNECTION_TEST` | `true` | skip cognee's flaky str-based probe |
| `COGNEE_SEARCH_TYPE` | `GRAPH_COMPLETION` (or `_COT` for the multi-hop-fair row) | cognee's retriever tier |

> Alternative LLM route: set `ANTHROPIC_API_KEY` and use cognee's native Anthropic path — the adapter's
> `_patch_cognee_anthropic_mode()` fixes the 1.1.2 structured-output (instructor) mode bug automatically. The OpenRouter-custom
> route above is the local-proven one; prefer it.

## Fairness — what's held identical, and why gpt-4o (not Opus)

The whole point of the head-to-head is to isolate the *architecture*, so everything matchable is matched
(per `FAIRNESS_AND_METHODOLOGY`):

* **Same embedder both sides** — `BAAI/bge-large-en-v1.5` (1024-dim). Cognee is **not** starved of resources;
  it gets the exact model June's dense lane uses. (The first online run's unfairness was the *opposite* —
  it accidentally gave Cognee OpenAI `text-embedding-3-large`, a different, keyed, billed model.)
* **Same LLM for the matched row** — `gpt-4o` on both. June-Opus is reported **separately** as the
  efficiency ceiling, never as a matched row.
* **Same 100 question ids, same scorer (EM/F1), same LLM judge, same terse answer prompt.**
* **Reasoning matched where possible** — set `COGNEE_SEARCH_TYPE=GRAPH_COMPLETION_COT` so Cognee's
  chain-of-thought graph completion answers June's grounded multi-hop reasoner (`hops=4`); report June with
  its R4 verify **off** for the apples-to-apples row (Cognee has no verify/revise rung).
* **Same retrieval task (0.0.18+)** — `JUNE_BENCH_JUNE_POOL=1` now pools **both** sides: June retrieves
  its gold from the pool AND Cognee builds its KG **once** over the same pool then retrieves (not handed
  each question's passages, no per-question KG teardown). Before 0.0.18 this was mismatched — June was on
  the hard retrieval task while Cognee got each question's own passages. The **local** head-to-head was
  always matched (`run_corpus_builder` builds once); only the online adapter had regressed.
* **What legitimately differs = what's being measured:** June's pure-core graph + grounded reasoner vs
  Cognee's graph-RAG. You don't bolt June's reranker onto Cognee — that component isn't Cognee's; each
  system uses its native retrieval over the *same* embeddings + model.

**Why not Opus for Cognee:** Cognee was offered Opus **first**. It cost **\$90+ and never completed
(aborted)** — so the scored, published Cognee number is the gpt-4o run, and gpt-4o is the *charitable*
fallback. The wrapper refuses an Opus Cognee run unless you set `COGNEE_ALLOW_OPUS=1`. Cost per 100Q
(metered): **June-Opus \$2.57 · June-gpt4o ~\$1.3 · Cognee-gpt4o \$10.64 · Cognee-Opus \$90+ (aborted)** —
June on its *premium* model beats Cognee on the *cheaper* model at ~¼ the cost (a Pareto win).

---

## The easy path (recommended) — one command, plain-language

For third parties, skip everything below and run this (june-bench 0.0.28+):

```bash
pip install "june-bench[cognee,june-api]"   # bundles cognee + fastembed — no separate steps
june-bench reproduce-h2h --key <YOUR_ACCESS_KEY> --questions 100
```

Cognee defaults to `bge-large-en-v1.5` — the model June's dense lane uses (a commodity open embedder, the
same one named in `FAIRNESS_AND_METHODOLOGY`) — so it's the **same-embedder** matched run out of the box,
nothing to export. It's a benchmark parameter, not June's moat (that's the cost-gated pipeline + graph +
reasoner). The command asks only for your OpenRouter key (pays for both systems' gpt-4o answers), then
bakes the matched stack behind the scenes — same embedder, same answer model, same open-pool task, same
judge — runs a pre-flight that catches the past failure class before any spend, batches the pool upload so
a big run doesn't lock the endpoint, and prints a June-vs-Cognee side-by-side with the metered cost.

> To swap the embedder, pass `--embedder <id>` (or `--admin-key` to auto-discover June's live embedder).
> `export JUNE_BENCH_EMBEDDER=<id>` also overrides the default globally.

The manual env + bash steps below remain for CI/scripting and for understanding what the command does.

## Step 0 · Where to run it

Run on a machine with **RAM for fastembed bge-large (~1.3 GB weights + working set) plus cognee's stores**
(sqlite/lancedb/kuzu, file-based under `./.cognee`). Your Mac is fine; the RAM-tight 8 GB AWS box is not the
place for the Cognee side (it's already hosting June's bge server). This does **not** rebuild or touch the
AWS deployment.

## Step 1 · Install

```bash
python3 -m venv .venv-cognee && source .venv-cognee/bin/activate
pip install -U pip
pip install "june-bench[cognee,june-api]==0.0.17"     # cognee[evals] + the httpx client for the June side
python -c "import fastembed" || pip install fastembed  # local bge embedder (keyless)
```

## Step 2 · Set the matched env (paste your key once)

```bash
export OPENROUTER_API_KEY=sk-or-...        # your key

# ── Cognee's LOCAL stack (identical to .env.cognee) ──
export EMBEDDING_PROVIDER=fastembed
export EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
export EMBEDDING_DIMENSIONS=1024
export HUGGINGFACE_TOKENIZER=BAAI/bge-large-en-v1.5
export LLM_PROVIDER=custom
export LLM_MODEL=openrouter/openai/gpt-4o          # matched row (Cognee-Opus was $90+, aborted)
export LLM_ENDPOINT=https://openrouter.ai/api/v1
export LLM_API_KEY=$OPENROUTER_API_KEY
export LLM_INSTRUCTOR_MODE=tool_call
export COGNEE_SKIP_CONNECTION_TEST=true
export COGNEE_SEARCH_TYPE=GRAPH_COMPLETION

# ── The harness LLM judge (verbosity-agnostic ruler; same key is fine) ──
export JUNE_JUDGE_LLM_URL=https://openrouter.ai/api/v1/chat/completions
export JUNE_JUDGE_LLM_MODEL=openai/gpt-4o
export JUNE_JUDGE_LLM_KEY=$OPENROUTER_API_KEY
```

## Step 3 · PRE-FLIGHT — the guard that would have caught the 422 (do NOT skip)

```bash
# (a) the embedder must be fastembed, never openai — this single check prevents the exact past failure
case "$EMBEDDING_PROVIDER" in
  fastembed) echo "✓ embedder = fastembed (keyless, free)";;
  *) echo "✗ ABORT: EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER — set it to fastembed or you'll 422 + get billed"; return 1 2>/dev/null || exit 1;;
esac
# (b) fastembed importable + weights resolvable
python - <<'PY'
from fastembed import TextEmbedding
m = "BAAI/bge-large-en-v1.5"
assert any(e["model"] == m for e in TextEmbedding.list_supported_models()), f"{m} not available in fastembed"
print("✓ fastembed can serve", m, "(first run downloads ~1.3GB)")
PY
# (c) OpenRouter LLM reachable with THIS key + model (fails cheap, before any ingest)
curl -sS https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-4o","max_tokens":5,"messages":[{"role":"user","content":"ping"}]}' \
  | grep -q '"choices"' && echo "✓ OpenRouter gpt-4o reachable" || echo "✗ OpenRouter/key/model problem — fix before running"
```

## Step 4 · SMOKE (2 questions) — cheap proof before the paid full run

```bash
june-bench suite --systems cognee --datasets hotpot --split smoke --limit 2 --judge \
  --model "cognee-opus+smoke" --out RESULTS_cognee_online_smoke.md
cat RESULTS_cognee_online_smoke.md
```

What a healthy smoke shows:

* stderr line `cognee reasoning search types available: [...]` (adapter reached cognee),
* **no** 422 / embeddings error; the run finishes both cells,
* EM may be low at n=2 (noise) but **F1/judge should be non-zero** — if EM≈0 *and* F1≈0 *and* judge≈0, the
  answer-extraction or prompt regressed (see Troubleshooting).

## Step 5 · The head-to-head (June endpoint vs local Cognee, same slice, same judge)

```bash
# June side points at your bench endpoint; Cognee side is local (env above)
export JUNE_BENCH_JUNE_URL=https://bench.june.januraine.ai
export JUNE_BENCH_JUNE_KEY=<your bench key>
export JUNE_BENCH_JUNE_POOL=1 JUNE_BENCH_JUNE_BACKFILL=1

# JUNE_BENCH_JUNE_POOL=1 now pools BOTH sides (0.0.18+): June retrieves from the pool AND Cognee builds
# its KG once over the same pool, then retrieves — a MATCHED open-pool task (not June-pool vs Cognee-context).
# Confirm the stderr line "cognee OPEN-POOL: building the KG once over the whole pool" appears.
june-bench suite --systems june-api,cognee --datasets hotpot --split full --limit 100 --judge \
  --model "gpt4o-h2h" --out RESULTS_h2h_online_n100.md
cat RESULTS_h2h_online_n100.md
```

Target board (same 100-slice, June scorer — from `FAIRNESS_AND_METHODOLOGY` / memory `june-vs-cognee-cost`
/ `june-openpool-h2h-result`). The bench endpoint answers with gpt-4o, so the command above **is** the
matched row:

| open-pool · 100Q | EM | F1 | corr | cost/100Q |
|---|---|---|---|---|
| **matched** — June-gpt4o | ~0.63 | ~0.80 | ~0.83 | ~\$1.3 |
| **matched** — Cognee-CoT (gpt-4o) | 0.530 | 0.658 | 0.700 | \$10.64 |
| *efficiency ceiling* — June-Opus (separate) | 0.760 | 0.894 | 0.911 | \$2.57 |

The matched (gpt-4o-both) row is the fair comparison. June-Opus is reported **separately** — it beats
Cognee-gpt4o on every ruler at ~¼ the cost, but it's not a same-model row. **Do not** run Cognee on Opus
(\$90+, aborted).

## Step 6 · Record

The `--out` table already stamps each result's effective config (moat-safe: capabilities only, no model
ids from the endpoint). Commit the two `RESULTS_*.md` files alongside the local head-to-head so the online
number is reproducible and dated. This closes FL-SB4 ("dev-only / not validated in-sandbox") with a real,
matched, judged run.

---

## Troubleshooting (each maps to a real past failure)

| Symptom | Cause | Fix |
|---|---|---|
| `422` / `embeddings` error, run aborts | `EMBEDDING_PROVIDER` drifted to `openai` | set it to `fastembed` (Step 2); Step 3(a) catches this |
| whole run dies on one bad question | old harness list-comprehension | fixed in 0.0.16+ (fail-soft per example + cascade-abort) — ensure `==0.0.17` |
| EM ≈ 0 but judge says correct | Cognee answered in full sentences | adapter uses the shipped terse `answer_simple_question_benchmark.txt`; override only via `COGNEE_SYSTEM_PROMPT_PATH` |
| EM ≈ 0 with UUIDs in the answer text | old `str(dict)` bug | fixed by `_extract_answer` (pulls `search_result`) — ensure `==0.0.17` |
| judge column all 0 | judge rate-limited/misconfigured | raise `JUNE_JUDGE_LLM_RETRIES`, or use a separate `JUNE_JUDGE_LLM_KEY` — NOT a sign Cognee is wrong |
| `COGNEE_SEARCH_TYPE` not found warning | older cognee without CoT | upgrade `cognee[evals]`, or accept the `GRAPH_COMPLETION` fallback |

Verified before publishing this runbook: adapter contract green (`test_sb4_cognee` 10/10, `test_sb6_wiring`
5/5 on 0.0.17); config knobs above cross-checked against `systems/cognee.py` defaults and the local
`.env.cognee.example`.
