# Reproduce June's open-pool HotpotQA number

Run the exact benchmark behind June's published open-pool headline **on your own machine**, against a
hosted June endpoint, in four commands. You bring your own LLM key (you pay only for answer synthesis;
June's retrieval + graph run on the endpoint). No June source, no models, no Docker on your side.

**Published target (open-pool, n=100):** EM ≈ 0.63 · F1 ≈ 0.80
Independently reproduced on AWS at **EM 0.6512 / F1 0.8164**.

---

## The easy way — one command

```bash
pip install "june-bench[june-api]"
june-bench reproduce
```

It asks you three things in plain English — the access key you were given, your own OpenRouter key
(you pay ~$1 for a full run; get one at https://openrouter.ai/keys), and how much to run — then runs
it with a live progress bar and prints the score with a ✓/✗ verdict. The dataset ships **inside** the
package, so there's **no download** — it works offline / behind a firewall. You never touch a config
variable. Non-interactive/CI: `june-bench reproduce --key … --llm-key … --questions 100`.

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
export JUNE_BENCH_LLM_KEY=<your-openrouter-key>
export JUNE_JUDGE_LLM_URL=https://openrouter.ai/api/v1/chat/completions
export JUNE_JUDGE_LLM_MODEL=openai/gpt-4o
export JUNE_JUDGE_LLM_KEY=<your-openrouter-key>
```

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

The result footnote records exactly what ran — for a valid reproduction it must read:

```
mode=pool(open-pool·retrieves) · backfill=on · answerer=llm:openai/gpt-4o
· max_sources=12 · max_tokens=64 · reason=grounded:...·hops=4 · dense=on
```

- **EM ≈ 0.63–0.65 / F1 ≈ 0.80** → reproduced. ✅
- **EM ≈ 0.32** → the dense lane didn't engage (embedder shows `none`; tell the host).
- **EM ≈ 0** → the answerer fell to the extractive floor (your `JUNE_BENCH_LLM_KEY` didn't reach the
  model — check the key/model).

`--limit N` runs the first N of the same fixed 100-question slice, so any N is internally comparable.
Swap `--model` and your keys to `anthropic/claude-opus-4-8` (temperature 1.0) to target the Opus
number (~0.76). Everything is the same identical EM/F1 scorer June scores itself with — see
`june_bench/score.py` and the `test_sb1_parity.py` gate.

---

## Run it against your OWN June instead

Point `JUNE_BENCH_JUNE_URL` at any June endpoint that has the dense lane + reasoner configured
(see June's deployment docs). The harness ships **no June
source** — it speaks only the documented `/v1/answer`, `/v1/ingest/text`, `/v1/canvases` HTTP
contract, so the same commands work against localhost or any host.
