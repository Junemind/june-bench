# june-bench

A pip-installable, **reproducible** benchmark suite for memory / QA systems — **June + pluggable
competitors** — over LoCoMo, LongMemEval, HotpotQA/2Wiki/MuSiQue, and FinanceBench, with the same
data and the same scorer.

```bash
pip install june-bench
june-bench list
june-bench run --system echo --dataset smoke --split smoke    # offline, no key, no download
```

## Reproduce the June vs Cognee head-to-head

One command runs **both** systems over the same HotpotQA open-pool, the same answer model, and the same
judge, and prints a side-by-side with the metered API cost:

```bash
pip install "june-bench[cognee,june-api]"     # bundles cognee + fastembed
june-bench reproduce-h2h --key <YOUR_ACCESS_KEY> --questions 100
```

* **Access key** — June's endpoint is hardware-limited (not yet funded), so runs are key-gated. Request
  one at **access@januraine.ai**; the reply includes your key and this exact command.
* **Same-embedder by default** — Cognee automatically embeds with `bge-large-en-v1.5`, the commodity open
  model June's dense lane uses, so it's a **same-embedder** matched run out of the box (nothing to export).
  This embedder is a disclosed benchmark parameter, not June's moat; pass `--embedder <id>` to swap it.
* **You bring an OpenRouter key** (prompted) — it pays for *both* systems' gpt-4o answers (~$21 for the
  chain-of-thought tier at n=100); the host never holds or pays for it.
* Cognee runs **locally** (needs RAM + a one-time ~1.3 GB fastembed download); June answers over its
  endpoint. The command batches the pool upload, blocks the $90 Opus-on-Cognee path, and meters real cost.

`june-bench reproduce` runs the June-only HotpotQA number the same way; `reproduce-retrieval` scores
June's recall@k/nDCG/MRR. All three are plain-language and need no `JUNE_BENCH_*` env vars.

A benchmark is `run(system, dataset) → records → score`. Two typed ports are the only extension
points:

* **`System`** — the thing benchmarked. `JuneApiSystem` (default; a thin HTTP client to June's
  `/v1/answer`, so **no June source is shipped**), `JuneLocalSystem` (`[june-local]` extra; a
  source-protected compiled wheel), `CogneeSystem` (`[cognee]` extra), or any future system as one
  adapter.
* **`Dataset`** — what it runs on. The four benchmarks behind a registry.

The scorer is the canonical SQuAD/HotpotQA EM/F1 + selective-accuracy/coverage/cost — Cognee-comparable.
Tiny **smoke fixtures ship in the wheel** (offline wiring proof); full splits are **fetched, sha-verified,
from a pinned release**. No score is ever baked into the package — every result row records
dataset + scorer + system + model + cost, so a published number is reproducible by a stranger, with the
exact command above.

## Honest-measurement notes

Both systems get the same documents, questions, gold answers, scorer, and embedder. Databases are
reset before runs so results are never contaminated by prior state. Costs are measured from provider
billing deltas, not estimated.

## Protocol notes (read before comparing numbers)

june-bench runs a **matched-pair protocol**: identical evidence pool, answer model, and judge for
every system, scored with EM/F1 plus a fixed LLM judge. Two deliberate differences from the official
benchmark settings: the default mode pools QA over the corpus (the official LoCoMo/LongMemEval
settings are per-conversation), and default runs use 100-question slices. This makes results
**directly comparable between systems run here** — and NOT comparable to published leaderboard
numbers, which use different protocols. Compare systems, not leaderboards.

Note on difficulty: pooling is the *harder* direction. The official settings give each question its
own haystack (e.g. ~40 sessions in LongMemEval_S); the pool is the union of every conversation in the
run, so each question faces strictly more distractors — including cross-conversation confusables the
official design never tests. Both systems face the same pool.

## Dataset licenses

Full splits are fetched from their official sources, sha-verified (see `june-bench fetch`). The small
bundled reproduce/smoke fixtures are subsets of HotpotQA (CC BY-SA 4.0), LongMemEval (MIT),
LoCoMo (CC BY-NC 4.0), and FinanceBench (CC BY-NC 4.0) — see `DATA_LICENSES.md` for attribution
and modification notes.

## Links

- Junê: https://june.januraine.ai
- Published results: https://june.januraine.ai (benchmarks section)
- Releases (desktop apps): https://github.com/Junemind/June_releases
