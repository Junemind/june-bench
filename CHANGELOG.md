# Changelog — june-bench

All notable changes to the public `june-bench` harness. Versions are published to PyPI via a
`bench-v*` tag (Trusted Publishing). The harness core stays stdlib-only; heavy systems live behind
extras.

## 0.0.31 — 2026-07-06

### Added — pre-run readiness gate ("box already hot" protection)
- **Before the pooled ingest, the run now waits for the endpoint to have capacity** instead of diving a
  big write onto an already-saturated box and failing after retries. Signal is the round-trip latency of
  a lightweight DB-touching health probe (`GET /v1/canvases/health`, median of a few samples) — if a
  trivial call is slow, the box's CPU is contended and writes will be too. Over threshold → prints
  `endpoint busy (… > …) — waiting up to Ns for load to settle…` and polls until it clears, then
  continues; if still busy at `max_wait` it proceeds anyway (batching + write-retries absorb the rest).
  **Bounded** (never hangs) and **fail-open** (an unmeasurable probe or a persistently busy box both
  proceed — the gate can only *delay*, never block a run). Wired in the runner before the sweep/ingest;
  optional per-System hook (Cognee/oracles skip it). Env-tunable: `JUNE_BENCH_READY_PROBE_S` (1.5s),
  `JUNE_BENCH_READY_MAX_WAIT_S` (120s), `JUNE_BENCH_READY_INTERVAL_S` (5s), `JUNE_BENCH_READY_GATE=0` to
  disable. Complements the endpoint-side CPU cap: the cap *prevents* self-inflicted saturation, the gate
  *waits out* saturation from any other source (prod spike, overlapping run) before committing to a write.

## 0.0.30 — 2026-07-06

### Added — pre-run sweep of orphaned pool canvases (self-healing)
- **Before a pooled run ingests, it now deletes any leftover `bench-pool` canvases from earlier runs** so
  orphans (from a hard-killed run, or a run that leaked on an older build) can't accumulate in the
  endpoint's single-writer SQLite and contend later writes into `database is locked`. `GET /v1/canvases`
  is user-fenced (returns only this access key's canvases) and the sweep deletes ONLY canvases named
  exactly `bench-pool` — a user's real canvases are never touched. Best-effort + fail-soft: an older
  endpoint without the list route, a transport blip, or a per-delete error is ignored and the run
  proceeds. Wired in the runner before `ingest_pool` (optional hook; Cognee/oracles are skipped). This
  removes the need for a manual endpoint DB reset between runs.

### Fixed — failed runs no longer leak their pool canvas
- **A run whose pool ingest 500'd left its `bench-pool` canvas behind in the endpoint DB.** `cleanup_pool`
  (runner `finally`, fires on every exit) deletes `self._pool_canvas`, but `ingest_pool` created the
  canvas first and only *registered* it (`self._pool_canvas = canvas`) **after** a successful ingest — so
  when the ingest itself failed, the handle was empty and the created canvas was orphaned. Leaked pools are
  exactly what bloat the single-writer SQLite into *more* `database is locked` failures on later runs, so
  the cleanup gap compounded the very problem it exists to prevent. **Fix:** register the handle the moment
  the canvas is created, before the ingest/backfill that can fail. New regression test drives an ingest-500
  and asserts the canvas is still torn down. (Successful runs were always cleaned up; only the failure path
  leaked.) Orphans from earlier failed runs are cleared by a one-time endpoint DB reset.

## 0.0.29 — 2026-07-06

### Fixed — 50-question runs no longer lock the endpoint (under-batching)
- **A 50q h2h sent all ~491 pool passages in ONE write and hit `database is locked` → HTTP 500, so June
  "did not complete."** The batch heuristic capped a "comfortable" single write at ~50 questions
  (`_QUESTIONS_PER_BATCH = 50`), so `_recommended_batches(50) = ceil(50/50) = 1` — one batch — and the
  `rec <= 1` guard in `_ask_ingest_batches` *also suppressed the prompt*, so the user was never offered a
  split. But 491 passages in one transaction is right at the endpoint's single-writer SQLite threshold
  (n=100 only cleared it because the operator manually chose 4 batches ≈ 248/write; the *recommended* 2
  ≈ 495/write would have been just as marginal). **Fix:** lower the cap to `_QUESTIONS_PER_BATCH = 30`, so
  `ceil(50/30) = 2` (50q now splits and prompts) and `ceil(100/30) = 4` (matches the known-good manual
  n=100 run). Output-neutral — same pool, same answers, only the number of upload transactions changes.
  Workaround on older installs: `JUNE_BENCH_POOL_INGEST_BATCHES=4`.

### Changed
- **`reproduce-h2h` now defaults to June's embedder — the same-embedder run is one command, no env var.**
  Previously the matched embedder had to be supplied every run (`--embedder bge-large-en-v1.5` or an
  exported `JUNE_BENCH_EMBEDDER`) because the model id was kept out of source by the moat guard. But the
  benchmark embedder is a *published parameter*, not the moat (the cost-gated extraction/graph pipeline
  is), so `bge-large-en-v1.5` is now the built-in default: `reproduce-h2h --key … --questions 100` is a
  same-embedder reproduction out of the box. `--embedder <id>` / `JUNE_BENCH_EMBEDDER` still override;
  `--admin-key` still auto-discovers. Interactive runs get a one-key a/b confirm instead of typing an id.
- **Moat guard narrowed, not weakened.** `bge`/`bge-large` are exempted from the shipped-package
  model-name scan (they're disclosed benchmark parameters); the internal model families (tokens stored
  encoded in the guard test) stay banned and tested across every `.py` in the package.

### Docs
- **README now carries the reproduce quickstart** — so the copy-paste June-vs-Cognee command is on the
  **public PyPI page** (the first place a stranger lands after `pip install`), not only in the private
  repo's runbook. Shows the exact `reproduce-h2h --key … --questions 100` command, where to request an
  access key (access@januraine.ai), the BYO-OpenRouter/cost note, and the `reproduce` /
  `reproduce-retrieval` siblings. Also removed the stale "Status: SB0 skeleton" line (the suite is fully
  shipped). README is the PyPI long-description → this version publishes it.

## 0.0.27 — 2026-07-05

### Fixed — big open-pool runs no longer lock the endpoint (n=100 works)
- **`reproduce` / `reproduce-h2h` at n≥~50 hit `database is locked` (→ 500) and June aborted.** The pooled
  ingest sent **one write per doc** (~1,600 for n=100), a transaction storm the endpoint's single-writer
  SQLite couldn't absorb even with `busy_timeout`. Now `JuneApiSystem.ingest_pool` uploads the shared pool
  in a **few batched writes** via `/v1/ingest/docs` (the same id-preserving, idempotent endpoint the
  retrieval path already uses) — collapsing ~1,600 transactions to a handful. **Output-neutral:** all docs
  land in the same pool canvas, so every question still retrieves over the whole union — only the number of
  upload transactions changes. Fail-soft: a server without `/v1/ingest/docs` (404) falls back to the
  per-doc path.
- **New interactive batch prompt** (large runs only). When the pool is big, both commands offer to split
  the upload and show a **computed** recommendation that scales with the set: `ceil(N_questions / 50)`
  (≈50 questions' worth of passages per batch) — 30q→1 (no prompt), 100q→2, 200q→4, 500q→10. Override via
  `JUNE_BENCH_POOL_INGEST_BATCHES`; unset ⇒ a safe ~500-docs/batch default, so non-interactive runs are
  covered too. 6 tests: batched-not-per-doc, batch-count honoured, 404 fallback, recommendation scaling.

### Changed — `reproduce-h2h` embedder default (no more retyping the id)
- If the operator sets `JUNE_BENCH_EMBEDDER=<id>`, the recommended embedder path **auto-assumes it** with
  an a/b confirm ("I'll use the default '<id>' … [a] use default · [b] choose a different id") instead of
  prompting for the id every run. Precedence: `--embedder` flag → `--admin-key` discover → `JUNE_BENCH_EMBEDDER`
  → interactive menu. Moat-safe: the id stays in the operator's env, never the shipped source (a fresh
  third party with no default still gets the full choice — they can't be silently handed June's embedder).
  3 tests: auto-assume, flag-beats-env, no-default→None.

## 0.0.26 — 2026-07-05

### Fixed — cost estimate is now tier-aware (was ~2x too low on CoT)
- The pre-run confirmation and the fallback cost ratio used Cognee's **base** figure ($10.64/100Q), but
  the h2h defaults to the **chain-of-thought** tier (to match June's multi-hop), which fires ~4–5 LLM
  rounds per question and costs ~**$20/100Q** — so a 30-q run quoted "~$3.3 for both" actually billed
  ~$6.2 for Cognee alone. Now `_confirm_cost`, the "how much to run?" menu, and the `_print_cost` fallback
  ratio all use the tier the user picked (`_cognee_per_100q(cot)`), phrased as "up to ~\$X". A completed
  run still shows the **measured** ratio from the credits delta; only the *estimate/fallback* changed. On
  CoT, Cognee is ~**15× June's API cost**, not 8×. 3 tests cover the tier-aware estimate, the CoT ratio,
  and the base-vs-CoT gap.

## 0.0.25 — 2026-07-05

### Fixed — pooled runs now self-clean (no endpoint-DB bloat)
- **A pooled QA / h2h run left its shared canvas (~hundreds of docs) in the endpoint's DB forever.**
  `JuneApiSystem` had no `cleanup_pool`, and the QA runner's `finally` only closed the HTTP client — so
  every `reproduce` / `reproduce-h2h` run accumulated another abandoned pool in the endpoint's
  single-writer SQLite, widening write-lock windows until later runs hit "database is locked" retries.
  (Retrieval was already fine — `run_retrieval` cleaned up.) Now `JuneApiSystem.cleanup_pool` deletes the
  run's pool canvas, and `runner.run_async` calls it in `finally` — on **every** exit path (success,
  per-example error, cascade-abort) and **before** the client closes. Unconditional, best-effort, and
  idempotent; a failure is harmless (isolation means no run ever reads another's canvas, so a leftover was
  only ever dead weight, never a correctness issue). 4 tests: direct delete + idempotency, deletion via
  the runner `finally`, and deletion even when every answer errors. **Note:** this stops *new*
  accumulation — pools left by pre-0.0.25 runs still need a one-time clear on the box.

## 0.0.24 — 2026-07-05

### Fixed
- **`reproduce-h2h` mode 1 crashed Cognee** with `Model openai/text-embedding-3-large is not supported in
  TextEmbedding`. Mode 1 ("no embedder specified") forced `EMBEDDING_PROVIDER=fastembed` but left the model
  unset, so Cognee fell back to its OpenAI default under the local provider — the same keyed OpenAI
  embedder behind the *original* online failure. Mode 1 now sets a **keyless local fastembed model chosen
  at runtime** (no hardcoded id → moat-safe) so it can't crash. The menu is relabeled: matching June's
  embedder (specify / admin-discover) is the **recommended same-embedder reproduction**; mode 1 is a
  clearly-labeled *different-embedder* run (fair on task/model/judge, not on the embedder). Regression test
  added.

### Added — `reproduce-h2h` cost section (actual, provider-authoritative)
- The result now prints what the LLM API **actually cost**, per system. Both June and Cognee answer on
  the caller's *same* OpenRouter key (June via the `X-LLM-Key` header, Cognee via litellm locally), so the
  harness snapshots OpenRouter's cumulative spend (`GET /api/v1/credits` → `total_usage`) **before and
  after each system's phase** — the delta is the **real amount OpenRouter billed** that run, straight from
  the provider (no token math, no estimation, and no endpoint change: June's cost is visible on the
  caller's key because it's a BYO-key answer). The ratio line reports the **measured** Cognee-vs-June
  multiple. If June's delta is ~0 it's labeled *server-side (not billed to your key)*.
- **Fail-soft.** The credits probe never raises; if it's unreachable the section falls back to the
  canonical metered per-100Q basis (June-gpt4o $1.30 · Cognee-gpt4o $10.64 → ~8×), clearly labeled as the
  reference. A metering hiccup can't affect a paid run. 6 tests cover the fallback, the server-side case,
  the measured ratio, and fail-soft probing. The point it surfaces: Cognee's chain-of-thought + one-time
  graph build cost roughly **8× June** per answer.

## 0.0.23 — 2026-07-05

### Fixed (correctness)
- **`reproduce-h2h`: Cognee failed with "LLM API key is not set" even with a valid key.** Cognee reads its
  LLM/embedder config from the environment at **import** time and caches it; `_preflight` imports cognee
  (to check it's installed) *before* `_apply_env` set the key — so cognee cached an empty config and its
  graph build (`cognify`) raised. Fix: `_apply_env` (pure — only sets `os.environ`) now runs **before**
  pre-flight and the cost gate, so the env is in place before anything imports cognee. June was unaffected
  (its answer is generated server-side). A regression test locks the `_apply_env`-before-`_preflight`
  ordering. No workaround (exporting `LLM_*` by hand) needed anymore.

## 0.0.22 — 2026-07-05

**Correctness fix — fetched full-set LoCoMo & LongMemEval scored recall 0** (surfaced by 0.0.15's
custom-from-full path, which let a full-set retrieval run complete to a score for the first time); plus a
3-way embedder choice for `reproduce-h2h` that resolves the moat-redaction conflict.

### Added — `reproduce-h2h` embedder: three explicit options (was: read a now-redacted public field)
The public `/v1/embeddings/health` no longer discloses the model id (moat rule), which broke h2h's
auto-discovery. It now offers a choice (flag or interactive):
- **1 · default** — Cognee's own default embedder; no matching. The comparison holds on the same task,
  pool, answer model, and judge. Moat-safe, zero-config, **recommended**.
- **2 · specify** — `--embedder <id>`: the operator names June's embedder, matched to a local fastembed
  model for an exact-embedder run.
- **3 · discover** — `--admin-key <key>`: reads the embedder from the **authenticated** `/v1/embeddings/config`
  route (new, admin-only, server-side) — the public endpoint stays redacted, so only an operator sees the id.

### Fixed
- **`_norm_longmemeval` / `_norm_locomo` were structurally wrong vs. the LOCAL converter that produced the
  published numbers** (`scripts/convert_dataset.py`) — so a fetched full-set run scored recall 0 (gold in a
  different id-space than the docs) and, even once ids matched, would *not* have reproduced the headline
  (LoCoMo was per-sample turn-grain vs. the local per-query session-grain). Both normalizers now replicate
  the converter exactly: **one query per question** (`<sample>#<qa>` / `<question_id>`), **one document per
  session** (`<qid>::<session>`, turns joined), and gold in the SAME id-space — LoCoMo mapping evidence
  dialog-ids (`D<n>:<t>`) to their **session** (`session_<n>`), LongMemEval using `answer_session_ids`.
  FinanceBench was already correct. **The bundled offline samples were never affected.**
- **`fetch` now validates gold ∈ corpus** and **fails loudly** if <50% of retrieval queries have their
  gold in the corpus — so a mismatched-id normalizer can never silently cache recall-0 data again. The
  round-trip tests now assert gold-in-corpus too (the gap that let this through).

### Action for users
Re-fetch the full sets to regenerate the corrected data, and clear any checkpoint from a buggy run:
`june-bench fetch --datasets longmemeval locomo --force` then delete
`~/.cache/june-bench/checkpoints/ret-{longmemeval,locomo}-*.jsonl` (or run with `--fresh`).

## 0.0.21 — 2026-07-05

### Fixed
- **`reproduce-h2h` embedder auto-discovery never worked.** It read the shared `probe_config`, which
  returns *capabilities only* and strips model ids (moat rule) — so the endpoint's `embedder` field was
  always absent and every run fell through to "Couldn't read June's embedder." Discovery now reads
  `/v1/embeddings/health` **directly** (new `_raw_embedder`) and matches the id against the local
  fastembed catalogue, so Cognee is auto-configured to June's embedder with no prompt. (Workaround on
  0.0.20: enter the model id when asked.)

## 0.0.20 — 2026-07-05

### Changed
- **`fastembed` is now bundled in the `[cognee]` extra.** Cognee's fair H2H needs the keyless local
  embedder, so `pip install "june-bench[cognee]"` now pulls it automatically — no separate
  `pip install fastembed` step (and no runtime pip-install, which would be surprising/unsafe). The
  pre-flight's missing-fastembed hint now points at the extra.

## 0.0.19 — 2026-07-05

### Added — `june-bench reproduce-h2h` (plain-language June-vs-Cognee, no bash, no env vars)
- A one-command interactive head-to-head, same UX as `reproduce`: asks only for the access key, the
  OpenRouter key, the Cognee reasoning tier, and how many questions — then bakes the **matched** stack
  (same embedder, same answer model, same open-pool task, same judge) and prints a side-by-side with the
  Δ EM. Replaces the `validate_cognee_online.sh` bash wrapper as the third-party path (the wrapper stays
  for CI/scripting).
- **Moat-safe embedder matching.** Cognee must embed with the same model June's dense lane uses; rather
  than hardcode a model id (which `test_no_model_name_anywhere_in_shipped_package` forbids), the command
  **discovers** it from June's live `/v1/embeddings/health` probe and matches it against the local
  fastembed catalogue by suffix — so no commodity model id ships in the source.
- **Pre-flight before spend**: blocks Cognee-on-Opus ($90+, aborted), checks cognee + fastembed are
  installed, and probes OpenRouter reachability — all before any ingest. Cost-confirmation gate before the
  paid run (`JUNE_BENCH_YES=1` to skip). 8 tests cover env baking, the opus block, embedder discovery, and
  CLI registration.

## 0.0.18 — 2026-07-05

### Fixed (fairness — matched open-pool H2H)
- **Cognee had no open-pool path → an unmatched retrieval task.** `JuneApiSystem` with `pooled=True`
  ingests the deduped union of all evidence once and each question **retrieves** its gold from the whole
  pool; the Cognee adapter only ever ingested *each question's own passages* at answer time (and
  `reset=True` pruned+rebuilt the whole KG **per question**). So a `--systems june-api,cognee` pool run
  put June on the hard retrieval task while handing Cognee its passages — and tore down the graph that is
  Cognee's value between every question. (The **local** head-to-head was already fair: its
  `run_corpus_builder` builds the corpus once, then answers all — this only regressed in the online
  adapter.)
- **New `CogneeSystem` pooled mode** mirrors June-pool *and* the local driver: `ingest_pool(examples)`
  builds the KG **once** over the deduped union, `answer` retrieves over it with **no** per-question
  ingest/prune, and the runner's per-example `ingest` is a no-op while pooled. Enabled by `COGNEE_POOL`,
  falling back to `JUNE_BENCH_JUNE_POOL` — so a **single** pool flag on an H2H pools **both** sides
  (`COGNEE_POOL=0` opts Cognee out). 4 new tests assert build-once, dedup, idempotency, no re-ingest, and
  that the runner hook pools via `ingest_pool`.
- **Anti-drift guard.** `_pool_from_env()` is a named helper with a test locking that Cognee reads the
  *same* `JUNE_BENCH_JUNE_POOL` flag June-api reads (asserted against `june_api.from_env`'s source) — so
  the two sides can't silently diverge into an unmatched H2H if June's pool flag is ever renamed.

## 0.0.17 — 2026-07-04

Closes the deferred audit items that were safe to close in-package (see `PRODUCTION_AUDIT.md`).

### Fixed / Added
- **Concurrent-run protection (H3).** A platform-safe advisory lock (`fcntl`, Unix) on the checkpoint
  means a second run with the same config is warned + denied instead of corrupting the file (no-op on
  Windows).
- **One shared health probe (H4).** `cli._probe_server` and `reproduce._health` now call one
  `probe_config` with a **narrow** except — a real bug propagates instead of reading as "endpoint down."
- **QA custom-cap parity.** `reproduce` warns when a custom N exceeds the bundled published 100 (and runs
  100) instead of silently capping. (No auto-fetch: the 100 *is* the headline set.)
- **M12** financebench distractor count derives from the loader cap (was a magic 150); non-crypto `md5`
  flagged `usedforsecurity=False`. **M15** judge `max_tokens` 4 → 8 (never truncate a yes/no). **L16**
  markdown table cells are escaped so a `|` in an error string can't break the results table.

### Still deferred (documented with actions in `PRODUCTION_AUDIT.md`)
Server-side ingest dedupe (a `june_service` change), streaming download (resource, not correctness), and
the in-`os.environ` secret refactor (construction-contract change) — each too large/out-of-package to
rush against "don't break a working run."

## 0.0.16 — 2026-07-04

Production-hardening pass (from a full-package audit). No behaviour change to a healthy run; the harness
is materially more robust under a busy/flaky endpoint, bad input, and interruption.

### Fixed (correctness / reliability)
- **Client-connection leak on the QA path (critical).** `runner.run_async` never closed the system's
  `httpx.AsyncClient` — leaked a connection pool on *every* `run`/`suite`/`reproduce` QA invocation. The
  runner now owns the lifecycle: the client is closed in a `finally` on every path (success, abort, error).
- **Cascade-abort could be defeated by resume.** A checkpoint *skip* used to reset the consecutive-error
  counter, so a dead endpoint interspersed with resumed hits would never trip the "endpoint down" guard —
  exactly the misleading partial run it exists to prevent. Skips no longer reset the counter.
- **Non-idempotent ingest retries could duplicate.** Retrieval ingest is id-preserving (uuid5 → upsert,
  safe). The QA text ingest now sends a content-derived `Idempotency-Key` so an endpoint that honours it
  dedupes a retried write (harmless if not; the pool is text-deduped upstream).
- **Durable checkpoints.** Each checkpoint append is now `flush()`+`fsync()`, so a kill/crash mid-run never
  loses a recorded (paid) result on resume.
- **Silent gold truncation is now loud (M11).** If the corpus cap (`JUNE_BENCH_MAX_CORPUS`) drops a *gold*
  doc from a long conversation, the loader warns once (recall would otherwise be silently capped).
- **`fetch` no longer leaks temp files** on a failed mirror, and writes the final dataset **atomically**
  (temp + rename) so a crash mid-write can't leave a truncated file.

### Changed (architecture)
- **One retry policy, one env parser.** Five hand-rolled retry loops (different backoffs, different
  retryable sets, no jitter) collapse into a single shared `retry_request` (jittered, to avoid
  thundering-herd on a shared endpoint). Every `JUNE_BENCH_*` int/float env var now parses through a
  hardened `env_int`/`env_float` — a bad value warns and falls back (clamped) instead of crashing the run
  (the corpus cap was parsed at *import* time, so a typo made the package un-importable).
- New tunable `JUNE_BENCH_CANVAS_RETRIES` (default 4); retrieval and QA canvas creation are now both retried.

## 0.0.15 — 2026-07-04

Reliability + a real custom-size path — from a real external user's session on the shared endpoint.

### Added
- **Custom count now runs any size, not just ≤20.** Previously "custom N" silently drew from the
  20-query bundled sample, so `custom 100` gave 20. Now a custom count **larger than the bundled sample
  auto-fetches the full published dataset once** (announced) and slices to N — so `custom 100` on
  LongMemEval gives a real 100, up to the full 500. Offline / fetch-failure falls back to the bundled
  sample with a clear message instead of crashing. Disable auto-fetch with `JUNE_BENCH_NO_AUTOFETCH=1`.

### Changed
- **QA ingest is now retry-resilient like retrieval.** `june_api._ingest_docs` previously had **no**
  retry — a single transient 5xx / `database is locked` failed the question. It now retries with backoff
  (shared `JUNE_BENCH_INGEST_RETRIES`), closing a real gap where the QA path was *more* fragile to the
  endpoint's SQLite write-locks than the retrieval path.
- **`JUNE_BENCH_INGEST_RETRIES`** (default 8) makes per-query ingest retry patience tunable across both
  the retrieval and QA paths.
- **Custom-from-full streams only N.** A custom count now streams exactly the requested queries from the
  full set (via the loader's `limit`) instead of parsing the whole file — so `custom 100` on LoCoMo no
  longer pulls all 1,986 into memory.
- **Cascade-abort default raised 5 → 8** (`JUNE_BENCH_MAX_CONSEC_ERRORS`) so a short cluster of
  data-specific 500s (e.g. a few heavy conversations hitting a write-lock) doesn't abort an otherwise
  healthy run. A truly dead endpoint still aborts; and resume recovers whatever errored.

## 0.0.14 — 2026-07-04

Resumable retrieval runs + a stronger moat guard.

### Added
- **Checkpoint / resume for both `reproduce-retrieval` and `reproduce` (QA).** Every completed item is
  appended to a JSONL checkpoint under `~/.cache/june-bench/checkpoints/`
  (`ret-<dataset>-<lane>-<scope>-<size>.jsonl` for retrieval, `qa-hotpot-<model>-<size>.jsonl` for QA);
  on a re-run those qids are loaded and skipped, so a dropped run (laptop sleep, network change,
  endpoint blip) resumes from where it stopped instead of restarting. For QA this also means the
  answer-model calls for completed questions are **not re-paid**. Only successes are checkpointed —
  failed items are retried. Disable with `JUNE_BENCH_NO_CHECKPOINT=1`; relocate with
  `JUNE_BENCH_CHECKPOINT_DIR`.
- The cascade-abort message now says "re-run to resume" and no longer mis-attributes a client-side
  network drop to a server OOM.

### Changed
- **Moat guard now scans the whole package**, not just `reproduce.py` — every shipped `.py` is checked
  (word-boundary match) so a model id can't hide in `systems/` or an error string.

## 0.0.13 — 2026-07-04

Moat hardening: the public package and the endpoint it talks to now describe the retrieval stack in
**capability** terms only — never a commodity model id.

### Changed
- **No model ids on any user-facing surface** (moat rule `moat-no-model-names`): `--show-config` and
  the result footnote now report `dense=on/off` instead of the embedder's model name; the reproduce
  docs/comments describe lanes (`BM25 + dense + fusion`) rather than naming the dense model.
- **`/v1/embeddings/health` + backfill responses redacted** server-side to `{enabled, lane}` /
  `{embedded, dense}` — the raw embedder id no longer leaves the endpoint. (Requires a backend
  redeploy to take effect on a live host.)
- **`--deep` relabeled** from a specific reranker architecture to the generic "reranker (deep pass)."

### Packaging
- **`tests/` pruned from the sdist** (new `MANIFEST.in`) and the guard's banned-token list encoded, so
  the published artifact carries no readable model id anywhere.

## 0.0.12 — 2026-07-04

Makes the **full published datasets** actually reproducible for anyone — `full` runs the real set, and
`fetch` works again after upstream sources moved.

### Fixed
- **`fetch` sources had bit-rotted** (would break every partner's full-set run):
  - **LongMemEval** — the `xiaowu0162/longmemeval` HF repo is deprecated; repointed at
    `longmemeval-cleaned/longmemeval_s_cleaned.json` (the cleaned set the published numbers use).
  - **FinanceBench** — `financebench_open_source.jsonl` → `financebench_merged.jsonl`; **and** the
    normalizer rewritten to emit a real retrieval task (global evidence-page pool, gold = the
    question's own page, distractors = others', namespaced per question with gold emitted first so it
    survives the corpus cap) instead of the old empty-gold evidence-grounded shape.
  - LoCoMo verified still live.
- **Dense backfill retries the SQLite write-lock** (`DELETE FROM node_embeddings … database is locked`)
  with WAL-settle backoff — this is what makes a `--dense` turn-grain run *engage across conversations*
  instead of degrading to sparse. (Confirmed: LoCoMo turns `--dense` → **R@10 0.92**, matching local's
  0.925, dense engaged across 3 conversations.)

### Added
- **`full` = the entire published dataset** (LoCoMo 1,986 / LongMemEval 500 / FinanceBench 150), read
  from `JUNE_BENCH_DATA` / the fetch cache via the streaming loader (no OOM on the 250 MB sets), with a
  clear bundled-sample fallback + `fetch` hint when the data isn't present. `quick`/`custom` stay on the
  bundled offline slice.
- REPRODUCE.md: a "bundled slice vs full published set" section — no private data, same public sources
  for everyone; and the flat-`R@1=R@5=R@10` tell for "this sample is too small".

## 0.0.11 — 2026-07-04

Robustness + UX for `reproduce-retrieval` (and matching `reproduce`), so a dense turn-grain run
completes instead of crashing and the operator can see what's happening.

### Fixed
- **`--dense` turn-grain no longer crashes.** The per-conversation canvas cleanup issued a
  `clear_workspace` DELETE that, right after a dense backfill's WAL writes, hit SQLite
  `database is locked`, 500'd, and closed the keep-alive connection — breaking the next
  conversation's `create_canvas` (looked like a server crash; the app was fine). Canvases are
  workspace-isolated, so we simply **don't delete them mid-run** (opt back in with
  `JUNE_BENCH_TURNS_CLEANUP=1`; wipe the bench DB volume between runs for a clean slate).
- **Backfill no longer times out silently.** Embedding hundreds of turn-chunks on CPU exceeds the
  default 120 s request timeout; the client used to give up while the server was still embedding, so
  `--dense` silently degraded to sparse. Backfill now uses a generous timeout
  (`JUNE_BENCH_BACKFILL_TIMEOUT`, default 900 s).
- `create_canvas` retries transient transport blips (a server-closed keep-alive connection).

### Added
- **Backfill visibility**: the result prints `Dense: engaged — backfilled N vectors`, `⚠ PARTIAL`, or
  `⚠ NOT engaged … ran SPARSE-ONLY` — a silent dense failure can no longer masquerade as a fused score.
- **Live progress bar** for both the turn-grain and default retrieval paths.
- **Question-count preset menu** (`quick / full / custom`) for both `reproduce-retrieval` and
  `reproduce` — no number to guess; `full` runs the whole bundled slice (no upper cap needed).

## 0.0.10 — 2026-07-04

Makes `reproduce-retrieval` as friendly as `reproduce` — a plain-language, moat-safe interactive
walkthrough (no model names), so a partner reproduces June's retrieval the same way they reproduce the
HotpotQA number.

### Added
- **Interactive menu** for `reproduce-retrieval`: after the access key + dataset it asks, in plain
  English, **which measurement** (quick session-grain vs *faithful* turn-grain — the one that
  reproduces the published LoCoMo numbers) and **which lanes** (lexical-only vs + the semantic lane).
  Every prompt describes a *capability*, never a model name.
- `--turns` / `--dense` now default to "ask" in interactive use and to sparse/haystack in
  non-interactive use, so existing scripted commands are unchanged. Faithful runs default to the full
  bundled 149-query slice.

## 0.0.9 — 2026-07-04

Makes `reproduce-retrieval` measure the **same task** June's local numbers were measured at — and
documents why the hosted number equals the local one.

### Added
- **`--turns`** — faithful **turn-grain** LoCoMo reproduction: each session is split into turns, June
  retrieves over the conversation's ~380–690 turn-chunks, and scoring is **session-MAX** (a session is
  found if any of its turns is retrieved). This is the exact task behind June's local `R@10 0.888`
  (sparse) / `0.925` (fused). Ingests **once per conversation** (not per query) to limit SQLite lock
  pressure, and prints a **per-category** breakdown (cat1 multi-hop … cat4 single-hop). New bundled
  fixture `locomo_turns_reproduce.june.json` (queries sampled ~proportional to LoCoMo's natural mix).
- **`--dense`** — opt into the embedded semantic lane (backfill). Off by default.
- **`--pooled`** / **`--deep`** — explicit opt-ins for the harder cross-conversation pool and for
  reranking (documented net-negative on conversational data).

### Changed
- **`reproduce-retrieval` default is now per-query haystack (not open-pool).** The open-pool default
  compared a 300-doc cross-conversation retrieval against local *haystack* numbers — a much harder
  task that read as a regression (~0.30). Haystack matches the local scope; `--pooled` restores the
  old behaviour. Backfill is **off by default** (sparse BM25 carries haystack recall; embedding was the
  slow, lock-prone step) — `--dense` re-enables it.

### Verified
- On the identical fixture the hosted endpoint's lexical lane returns **exactly** what in-process
  `bm25_scores` returns (0.72 = 0.72), i.e. the shipped retrieval == the local benchmark's BM25. The
  turn-grain run lands **R@10 ≈ 0.82** (cat4 0.96 ≈ local 0.97), the residual to 0.888 being category
  mix, not engine.

## 0.0.8 — 2026-07-03

Brings the **retrieval** benchmark to the same one-command, offline-safe, hosted-endpoint experience
the QA benchmark already has.

### Added
- **`june-bench reproduce-retrieval`** — plain-language reproduction of June's *retrieval* quality
  (recall@k / nDCG@10 / MRR) against a hosted endpoint. No answer LLM involved, so it asks only for an
  access key and a dataset (locomo / longmemeval / financebench). Runs **open-pool** retrieval (ingest
  the whole corpus once, retrieve each query's gold out of it) via `/v1/ingest/docs` + `/v1/search`,
  and prints the metrics with a moat-safe capability description (no model names).
- **Bundled retrieval slices** (`datasets/fixtures/{locomo,longmemeval,financebench}_reproduce.june.json`)
  — small pools (~300 docs) where each query's **gold docs are guaranteed present** plus distractors,
  so it's a real retrieval task that works offline / behind a firewall. (Full splits still via `fetch`.)

_Attribution: LoCoMo (Snap Research), LongMemEval, and FinanceBench are redistributed here as small
evaluation subsets under their respective source licenses._

## 0.0.7 — 2026-07-03

Makes `reproduce` **work on any network** and lets the caller **choose the model**.

### Added
- **Fully-open BYO model.** `reproduce` now asks which answer model to use (or `--model <id>` /
  `$JUNE_BENCH_LLM_MODEL`) — any OpenRouter id. The endpoint routes it via `X-LLM-Model` to **both**
  the answer synthesizer **and** the multi-hop reasoner, so June's pipeline runs end-to-end with the
  caller's model — the faithful-reproduction property (each published number ran one model
  throughout). Known models show their published target (gpt-4o ~0.63, Opus ~0.76 EM); any other
  model still runs — reported honestly as "June's pipeline + your model." The judge stays a fixed
  model so judged scores are comparable across answer models.
  _(Server: `X-LLM-Model` header on `/v1/answer` + `byo_key_reason_step_factory`; additive and
  default-off — absent header ⇒ unchanged behaviour.)_

### Changed
- **`reproduce` ships its own data.** The exact 100-question HotpotQA slice (byte-identical to the
  parity run) is now bundled inside the package (`datasets/fixtures/hotpot_reproduce.json`), so
  `june-bench reproduce` needs **no `fetch`** and works offline / behind restrictive firewalls — the
  earlier failure mode where a locked-down box (or a partner) couldn't reach `hotpotqa.github.io`.
  `fetch` is still there for the full splits / other datasets.

_Attribution: HotpotQA (Yang et al., 2018) is licensed CC BY-SA 4.0; this redistributed 100-question
subset is under the same license._

## 0.0.6 — 2026-07-03

The friendly, no-env-vars reproduction path — so a non-expert can reproduce the number without
understanding a single `JUNE_BENCH_*` variable.

### Added
- **`june-bench reproduce`** — one command, plain-language prompts. Asks only for an access key, the
  user's own OpenRouter key, and a run size ("[1] quick 5 / [2] full 100"); everything else (endpoint,
  open-pool, backfill, dataset, judge) is a baked-in preset. Auto-downloads the dataset, shows a live
  progress bar, and prints a plain result with a "✓ reproduced" verdict — no markdown table, no env
  vars. Non-secret only: keys come from flags → env → prompt and are never persisted.
- **Live progress callback** in the runner (`run(..., on_progress=)`) — the suite still runs silent;
  `reproduce` uses it for the bar.
- **Moat-safe output.** The friendly summary describes *capabilities* ("open-pool retrieval · fused
  dense + lexical search · grounded multi-hop reasoning"), never model names. Commodity model ids are
  shown only behind `--show-config` (the auditor path, read live from the endpoint). Enforced by
  `tests/test_reproduce.py::TestNoMoatLeak`.

## 0.0.5 — 2026-07-03

Adds the **open-pool QA mode** (the setting behind June's published open-pool headline) and fixes the
one snag every first-time reproducer hit. Verified end-to-end against a hosted June endpoint on AWS:
online gpt-4o open-pool = **EM 0.6512 / F1 0.8164 (n=100)**, matching the local 0.630 / 0.797 within
sampling noise.

### Added
- **Open-pool QA (`JUNE_BENCH_JUNE_POOL=1`).** Ingests the deduped **union of all items' passages
  once** into a single shared canvas, then answers each question over the *whole pool* — so June's
  retrieval lanes must FIND the gold (recall < 1.0), the harder setting that mirrors the local
  `run.py --retrieval pool`. Previously the online QA path only had record (passages handed in) or
  per-question ingest; there was no way to reproduce the open-pool number over HTTP. `JuneApiSystem`
  gains `ingest_pool()`; the runner calls it once before the loop; `test_pooled_qa.py` pins the
  contract (pool ingested once, shared canvas, zero per-question ingest).

### Fixed
- **`fetch` → run "file not found".** The loader searched `$JUNE_BENCH_DATA` / `<pkg>/data` /
  `<pkg>/benchmarks/apex_qa` but **not** `~/.cache/june-bench/data`, where `june-bench fetch` actually
  writes — so a standalone `fetch` then `suite`/`retrieve` failed even though the file was downloaded.
  The loader now searches the fetch cache too. (The #1 first-run reproduction snag.)

## 0.0.4 — 2026-06-24

The release that makes the **online path** (`--system june-api`, June reached over HTTP) actually
work against a real June endpoint, plus self-serve dataset download. First end-to-end test of the
HTTP loop surfaced two client bugs — both fixed here.

### Fixed
- **Auth header.** `JuneApiSystem` sent the API key as `Authorization: Bearer <key>`, but June's
  service authenticates on the **`X-API-Key`** header — so every request 401'd. The client now sends
  `X-API-Key`, matching the service contract. (Without this, no third-party `--system june-api` run
  could authenticate.)
- **Canvas isolation lifecycle.** For per-question isolation the client put an invented name
  (`bench-<qid>`) straight into the `X-Canvas` header, but June resolves `X-Canvas` as a
  **server-issued canvas id (UUID)**, fail-closed against the registry — so ingest 404'd
  ("canvas not found"). The client now follows the real lifecycle: `POST /v1/canvases` →
  use the returned id as `X-Canvas` for ingest/answer → `DELETE` on cleanup. (The server's
  fail-closed check is correct security and was left unchanged.)

### Added
- **`june-bench fetch`** — download the full dataset splits (HotpotQA, 2Wiki, MuSiQue, LoCoMo,
  LongMemEval, FinanceBench), sha-verified from pinned sources, into `~/.cache/june-bench/data`
  (or `$JUNE_BENCH_DATA`). Lets a third party with no local `data/` reproduce the full suite.
  `--from <file>` normalizes a local raw dump when a mirror is down.
- **BYO-key answering.** `JuneApiSystem` forwards the caller's own LLM key as `X-LLM-Key`, so a
  hosted June endpoint synthesizes on the caller's dime (host never holds the key). Falls back to
  `OPENROUTER_API_KEY`. Env: `JUNE_BENCH_LLM_KEY`.
- **Dense-lane backfill** (`JUNE_BENCH_JUNE_BACKFILL=1`) — embeds freshly-ingested content via
  `/v1/embeddings/backfill` so an embedder-configured endpoint answers on its dense lane; fail-soft when
  no embedder is wired.
- **Per-question canvas cleanup** (`JUNE_BENCH_JUNE_CLEANUP=1`) — deletes each question's canvas
  after answering so workspaces don't accumulate over a run.
- Configurable client timeout via `JUNE_BENCH_JUNE_TIMEOUT` (default 120s for memory datasets).

### Tests
- SB3 mocks updated to the corrected create→use→delete canvas contract.
- Fixed `test_factory_errors_without_cognee`: its import `try/except Exception` swallowed its own
  `self.skipTest()` (SkipTest subclasses Exception), so on a machine **with** cognee installed it ran
  the error-path assertion and falsely failed. The skip now lives outside the try. Suite: 70 pass
  (cognee absent) / 69 pass + 1 skip (cognee installed).
- Built sdist + wheel pass `twine check`; the wheel installs into a clean env and runs
  `june-bench list` / `run --system echo`.

## 0.0.3 — 2026-06-23
- Relicensed MIT. Context-QA: feed both the in-prompt `context` passages and the `corpus` docs into
  a per-question canvas so context datasets answer with evidence and questions don't cross-contaminate.

## 0.0.2 — 2026-06-23
- Relicensed MIT; PyPI metadata + Trusted-Publishing workflow.

## 0.0.1
- Initial: contracts (`System`/`Dataset` ports), runner, scorer, `echo`/`null` smoke systems,
  bundled smoke fixtures.
