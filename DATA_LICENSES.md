# Dataset licenses & attribution

june-bench itself is MIT-licensed. The benchmark **data** it uses belongs to its original authors.
Full splits are never redistributed here — `june-bench fetch` downloads them from their official
sources and sha-verifies them. What IS bundled (in `june_bench/datasets/fixtures/`) are small,
modified subsets used for offline smoke tests and for pinning the exact question slices of the
published reproduce runs. Modifications: subsetting, reformatting into june-bench's record schema,
and (for conversational sets) flattening sessions into retrieval documents.

| Dataset | Source | License | Bundled fixture |
|---|---|---|---|
| HotpotQA | https://hotpotqa.github.io | CC BY-SA 4.0 | `hotpot.smoke.json`, `hotpot_reproduce.json` |
| LongMemEval | https://github.com/xiaowu0162/LongMemEval | MIT | `longmemeval_reproduce.june.json` |
| LoCoMo | https://github.com/snap-research/locomo | CC BY-NC 4.0 | `locomo_reproduce.june.json`, `locomo_turns_reproduce.june.json` |
| FinanceBench | https://huggingface.co/datasets/PatronusAI/financebench | CC BY-NC 4.0 | `financebench_reproduce.june.json` |
| 2WikiMultihopQA | https://github.com/Alab-NII/2wikimultihop | fetched only — not bundled | — |
| MuSiQue | https://github.com/StonyBrookNLP/musique | fetched only — not bundled | — |

Attribution notices:

- **HotpotQA** — Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
  Answering* (EMNLP 2018). Shared under CC BY-SA 4.0; the bundled subset is likewise shareable
  under CC BY-SA 4.0.
- **LongMemEval** — Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
  Memory* (ICLR 2025). MIT License, © 2024 Di Wu.
- **LoCoMo** — Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*
  (2024), Snap Research. CC BY-NC 4.0 — the bundled subset is provided for non-commercial
  benchmark-reproduction use only.
- **FinanceBench** — Islam et al., *FinanceBench: A New Benchmark for Financial Question Answering*
  (2023), Patronus AI. CC BY-NC 4.0 — the bundled subset is provided for non-commercial
  benchmark-reproduction use only.

If you are a rights holder and want a bundled subset removed in favor of fetch-at-runtime, open an
issue — the loaders already support deriving every fixture from the officially fetched files.
