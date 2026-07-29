# Reproduce June's open-pool HotpotQA number

Run the exact benchmark behind June's published open-pool headline **on your own machine**, against a
hosted June endpoint, in four commands. You bring your own LLM key (you pay only for answer synthesis;
June's retrieval + graph run on the endpoint). No June source, no models, no Docker on your side.

**Published target (open-pool, n=100, `claude-opus-4-8` served Anthropic-direct):** EM 0.72–0.75 · F1 0.85–0.88 at 97–98% coverage — answered-and-correct ≈ **0.73**
*Re-baselined 2026-07-27 (local, ×2) and independently re-verified on the hosted endpoint 2026-07-29: **EM 0.75 · F1 0.88 · judged-correct 90% at 97% coverage**. The pre-drift aggregator-era target (EM ≈ 0.63 via OpenRouter, no coverage column) is retired — see “Serving platform matters” below. `gpt-4o` OpenAI-direct: provisional 0.65 / 0.83 at 95%.*

---

## The easy way — one command

```bash
pip install "june-bench[june-api]"
june-bench reproduce
```

It asks you four things in plain English — the access key you were given, **which platform serves
the answer model** (OpenRouter, or OpenAI / Anthropic / Google direct — see "Serving platform
matters" below; the choice is stamped on the result), the matching API key (you pay ~$1 for a full
run), and how much to run — then runs it with a live progress bar and prints the score with a ✓/✗
verdict. The judge automatically follows your platform (same key, that platform's cheap tier). The
dataset ships **inside** the package, so there's **no download** — it works offline / behind a
firewall. You never touch a config variable.
Non-interactive/CI: `june-bench reproduce --key … --llm-key … --platform anthropic --model
claude-opus-4-8 --questions 100`.

Prefer to see and control every setting yourself? The manual path is below.

### Retrieval too (recall@k / nDCG / MRR)

To reproduce how well June *finds* evidence (not how it answers), no LLM key needed:

```bash
june-bench reproduce-retrieval
```

It walks you through it in plain English — the access key, the dataset (locomo / longmemeval /
financebench), **which measurement** (quick, or the *faithful* turn-level reproduction of June's
published LoCoMo numbers), and **which lanes** (lexical-only, fast — or add the semantic lane) — then
prints recall@k / nDCG@10 / MRR. Data is bundled, so it runs offline, and nothing in the prompts or
output names a model.
Non-interactive: `june-bench reproduce-retrieval --key … --dataset locomo --turns` (add `--dense` for
the semantic lane).

### Retrieval modes (choose the task, honestly)

By **default** the command runs **per-query haystack** — for each query June retrieves the gold within
that query's own conversation. This is the same *scope* June's local numbers were measured at, and it's
fast (sparse BM25 carries recall; the dense lane is off by default).

| flag | what it does | when |
|---|---|---|
| *(none)* | per-query haystack, sparse-only, no rerank | quick, matches local scope |
| `--turns` | **faithful turn-grain** LoCoMo reproduction: sessions split into turns, retrieve over the conversation's ~380–690 turn-chunks, score **session-MAX** (a session is found if any of its turns surfaces) — the exact task behind June's local `R@10 0.888` (sparse) / `0.925` (fused). Reports per-category. | apples-to-apples with the local headline |
| `--dense` | add the embedded semantic lane (backfills embeddings; slower) | to reproduce the *fused* number |
| `--pooled` | harder cross-conversation variant (retrieve out of every conversation at once) | stress / realism |
| `--deep` | ask the endpoint to rerank | note: measured **net-negative** on conversational data — off by default |

**Faithful turn-grain run:**
```bash
june-bench reproduce-retrieval --key … --dataset locomo --turns --questions 149
```
Expect **R@10 ≈ 0.82** sparse (cat4 single-hop ≈ 0.96 ≈ local 0.97; cat1 multi-hop ≈ 0.53). The
aggregate depends on the category mix sampled — the bundled slice tracks LoCoMo's natural (cat4-heavy)
distribution; per-category is the cleanest comparison.

### Sample size: bundled slice vs the full published set

The "How many questions?" menu (or `--questions N`) controls the sample:

* **quick / custom** run the **bundled offline slice** (20 questions per dataset) that ships *inside*
  the package. It works with no download and proves retrieval works — but it is **small and easy**, so
  a 20-query number is a smoke check, not the published figure (recall pins near the top; a flat
  `R@1 = R@5 = R@10` is the tell that the slice is too easy, not a real curve).
* **full** runs the **entire published dataset** — LoCoMo **1,986** / LongMemEval **500** /
  FinanceBench **150** — which is what the site's headline numbers are measured on.

Because the full sets are large (LongMemEval alone is ~250 MB), they are **not bundled**. `full` reads
them from `JUNE_BENCH_DATA` / the fetch cache; if they're absent it falls back to the bundled slice and
tells you to fetch. **Every user gets them the same way — from the original public sources**, so `full`
is exactly as reproducible for a partner as for us:

```bash
june-bench fetch --datasets longmemeval        # LoCoMo → GitHub · LongMemEval / FinanceBench → HuggingFace
june-bench reproduce-retrieval --key … --dataset longmemeval    # → count: [2] full  (500 queries)
```

There is **no private data and no shortcut** — the bundled slice is the offline convenience, `fetch` +
`full` is the exact published number, and both are public. (Each dataset also has a documented manual
download path in `june-bench fetch --help` if a source URL ever moves.) Use **lexical only** for
LongMemEval `full` — sparse is its champion (~0.95) and it avoids the dense-lane write load.

### Parity note (why online == local)

The endpoint's lexical lane uses the **same `bm25_scores`** as the local benchmark, so on the identical
fixture the hosted number equals in-process BM25 exactly. If you see a much lower number (e.g. ~0.30),
you're almost certainly running **`--pooled`** (retrieving out of *all* conversations at once) and
comparing it to a local **haystack** number — a harder task, not a regression.

---

## What "open-pool" means (why it's the honest setting)

June is handed **no gold**. The harness ingests the deduped union of *all 100 questions'* passages
(~991) into one shared corpus, then each question must **retrieve** its own evidence out of the whole
pool (BM25 + dense + fusion) before a grounded, multi-hop answer. Recall is < 1.0 by
construction — the retrieval lanes are actually exercised, not bypassed.

---

## 1 · Install

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install "june-bench[june-api]"
```

## 2 · Get the dataset

```bash
june-bench fetch --datasets hotpot        # → ~/.cache/june-bench/data (HotpotQA distractor dev)
```

## 3 · Point at the endpoint + your key

```bash
export JUNE_BENCH_JUNE_URL=https://bench.june.januraine.ai   # the hosted June bench endpoint
export JUNE_BENCH_JUNE_KEY=<bench-api-key>                    # request one at access@januraine.ai
export JUNE_BENCH_JUNE_POOL=1 JUNE_BENCH_JUNE_BACKFILL=1  # open-pool + embed the pool

# Bring your OWN LLM key — you pay for answer synthesis (and the judge), not the host.
export JUNE_BENCH_LLM_KEY=<your-llm-key>
# Optional: which platform serves the answers (default openrouter — see the caveat below).
# Direct platforms need an endpoint that advertises support (the harness guards this for you).
export JUNE_BENCH_LLM_PLATFORM=openrouter        # or: openai · anthropic · google
export JUNE_JUDGE_LLM_URL=https://openrouter.ai/api/v1/chat/completions   # judge (or let the
export JUNE_JUDGE_LLM_MODEL=openai/gpt-4o                                 # platform menu set it)
export JUNE_JUDGE_LLM_KEY=<your-llm-key>
```

These env vars drive `june-bench suite` runs too — the platform selector isn't menu-only.

## 4 · Run

**Smoke first (~2 min, a few cents)** — confirm the whole path works before committing to the full run:

```bash
june-bench suite --systems june-api --datasets hotpot --split full --limit 5 \
  --model "gpt-4o+smoke" --out RESULTS_smoke.md
cat RESULTS_smoke.md
```

You want a table with a real EM/F1 (non-error) and the stderr line
`pooled QA: ingested N deduped docs into one shared corpus`. If that looks right, run the headline:

```bash
june-bench suite --systems june-api --datasets hotpot --split full --limit 100 \
  --judge --model "gpt-4o+open-pool" --out RESULTS_openpool.md
cat RESULTS_openpool.md
```

Takes several minutes (100 questions × multi-hop gpt-4o, sequential; the endpoint answers one at a
time). `--limit 24` is a faster, still-comparable middle ground.

---

## Reading the result

The result footnote records exactly what ran — for a valid reproduction against the hosted
endpoint's published configuration it must read (BYO runs show the host's *startup* synthesizer
here; the `As-run:` stamp is the line that records what actually answered):

```
mode=pool(open-pool·retrieves) · backfill=on · answerer=llm:openai/gpt-4o
· max_sources=12 · max_tokens=64 · reason=grounded:...·hops=4 · dense=on
```

Since 0.1.0 the result block also prints **coverage** (how many of the asked questions June chose
to answer — its abstentions are honest refusals, never gambles), **answered-and-correct per question
asked**, and an **`As-run:` stamp** (model · platform · endpoint · date). The ✓/✗ **verdict gates on
right-per-asked (`EM × coverage`) against the target's own coverage** — a run that answers fewer
questions can no longer print `✓ reproduced` on selective EM alone. Every completed run also appends
a machine-readable row to `~/.cache/june-bench/results.jsonl`.

- **right-per-asked within ~0.05 of the target at comparable coverage** → reproduced. ✅
- **selective EM fine but coverage well below the target's** → the verdict says so explicitly:
  same score, different system — investigate the abstentions (start with the serving platform).
- **EM ≈ 0.32** → the dense lane didn't engage (embedder shows `none`; tell the host).
- **EM ≈ 0** → the answerer fell to the extractive floor (your `JUNE_BENCH_LLM_KEY` didn't reach the
  model — check the key/model).

`--limit N` runs the first N of the same fixed 100-question slice, so any N is internally comparable.
Targets are **per model AND per serving platform** (each `_MODEL_TARGETS` entry records the EM, F1,
coverage, and serving context it was measured at): `anthropic/claude-opus-4-8` via OpenRouter carries
the aggregator-era ~0.76; **`claude-opus-4-8` served Anthropic-direct carries the 2026-07-27
re-baseline (0.72 / 0.85 at 98% coverage, temperature 1.0) — re-verified against the hosted
endpoint on 2026-07-29 at 0.75 / 0.88 / judged 90% at 97% coverage (✓ reproduced)**, and `gpt-4o`
OpenAI-direct carries a provisional 0.65 / 0.83 at 95%. Everything is the same identical EM/F1 scorer June scores itself
with — see `june_bench/score.py` and the `test_sb1_parity.py` gate.

---

## Run it against your OWN June instead

Point `JUNE_BENCH_JUNE_URL` at any June endpoint that has the dense lane + reasoner configured
(see June's deployment docs). The harness ships **no June
source** — it speaks only the documented `/v1/answer`, `/v1/ingest/text`, `/v1/canvases` HTTP
contract, so the same commands work against localhost or any host.

## Serving platform matters (measured, July 2026)

The answer model's **serving platform is part of the experiment**, and June is the system honest
enough to show it. June answers only what its evidence supports and refuses the rest — it does not
gamble. An aggregator (OpenRouter) routes each request to an unpinned, changing provider mix, and
when that serving drifts conservative, June's honest refusals rise; guess-style systems have no
refusal channel, so the same drift hides inside silently-changed guesses instead.

Measured on an identical engine and identical questions (2026-07-27):

| serving | gpt-4o | claude-opus-4-8 |
|---|---|---|
| via OpenRouter (aggregator) | 45–49 / 100 right-per-asked | 55 / 100 |
| served DIRECT (vendor API)  | **62 / 100** | **71–72 / 100** |

The hosted endpoint was rebuilt to this direct-platform shape on 2026-07-29 and re-measured at
**73 / 100 right-per-asked** (EM 0.75 at 97% coverage) — the endpoint now serves, and scores as,
the same June measured locally.

Use OpenRouter for one-key convenience and real-time cost metering. For accuracy-representative
or publishable numbers, choose a **direct platform** in the menu — every result stamps the
platform it ran on (`As-run:`), so numbers from different serving paths are never conflated.
