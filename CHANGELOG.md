# Changelog

All notable changes to KongBrain are documented here.

## [0.5.0] - 2026-04-25

Configurable embedding providers. Closes #1.

### Features
- **Configurable embedding provider**: New `embedding.provider` config field. Options: `local` (BGE-M3 via node-llama-cpp, default and unchanged) or `openai-compat` (any OpenAI-compatible `/v1/embeddings` endpoint — OpenAI, Azure OpenAI, Together, Anyscale, vLLM, LM Studio, Ollama, DeepInfra, Fireworks).
- **OpenAI-compatible provider**: `fetch`-based, no SDK dependency. Batches inputs at 96/request, retries 429 + 5xx with exponential backoff and `Retry-After` honoring, hard-fails on 401/403/404 and `insufficient_quota` with clear error messages, verifies returned dimensionality matches config.
- **Per-row provider tagging**: Every vector-bearing table (`turn`, `concept`, `memory`, `artifact`, `identity_chunk`, `skill`, `reflection`, `monologue`) gets an `embedding_provider` column. Searches filter by the active provider so vectors from different models (different vector spaces) never mix in HNSW results.
- **Re-embed migration tool**: `npx kongbrain-reembed --from <provider-id> [--dry-run] [--tables …] [--batch …]`. Resumable on interruption (the WHERE filter naturally excludes processed rows). Reports per-table progress and estimated cost.
- **Startup mismatch warning**: Logs a clear notice (with row counts and migration command) when the configured provider does not match what is in the database.
- **Provider env overrides**: `KONGBRAIN_EMBED_PROVIDER` flips provider without editing config; `OPENAI_BASE_URL` overrides endpoint (matches the official OpenAI SDK convention); `embedding.openaiCompat.apiKeyEnv` names the env var holding the secret so keys never appear in config files.
- **Plugin manifest**: `openclaw.plugin.json` extended with `provider` / `dimensions` / `openaiCompat` schema and uiHints with inline help text.

### Infrastructure
- **Idempotent schema migration + backfill**: All schema additions use `IF NOT EXISTS`. On first startup with this version, existing rows are tagged with `local-bge-m3`. Runs cleanly on every subsequent startup as a no-op.

### Tests
- 439 → 469 tests. New: 17 OpenAI provider unit tests (success, batching, dim mismatch, retry, hard-fail), 4 config tests (provider, env overrides, fallback), 8 migration tests (full migrate, table filter, dry-run, blank-text, multi-batch, refusal, no-op, format), 4 backfill upgrade-path integration tests, plus 6 gated live tests (real OpenAI / real DB / real reembed) that skip in CI.

### Documentation
- README and README.npm updated with embedding-provider comparison table, switching instructions for OpenAI / Ollama / vLLM, and migration command.

### Upgrade notes
- **No action required for existing local BGE-M3 deployments.** The schema migration adds the new column and tags all existing rows as `local-bge-m3`. Search continues to work identically.
- **To switch providers**: set `embedding.provider: "openai-compat"` and `OPENAI_API_KEY`. On restart you will see a warning about rows in the old vector space. Run `npx kongbrain-reembed --from local-bge-m3 --dry-run` to estimate cost (~$0.04 per ~3,400 turns on text-embedding-3-small), then drop `--dry-run` to migrate. Resumable if interrupted.

## [0.4.4] - 2026-04-04

### Performance
- **WMR rebalance**: Cosine-dominant scoring weights, dampen access count feedback loop that was reinforcing already-popular memories.
- **Tag-boosted concept retrieval**: Surface topically relevant concepts even when embedding similarity alone misses them.

### Bug Fixes
- **Empty LLM extraction responses**: `outputFormat` injected via pi-ai's `onPayload` hook caused Anthropic API to return 0 content blocks. Removed structured output from pi-ai code path; daemon's JSON parsing cascade handles free-text reliably.
- **`SELECT WHERE id IN $ids` binding**: Same silent no-op as `bumpAccessCounts` — SurrealDB string arrays don't resolve to record references. Fixed in `getSessionRetrievedMemories` and ACAN `fetchTrainingData`.
- **ACAN NaN/Infinity validation**: `loadWeights` now rejects corrupted weights (null, NaN, Infinity in bias, W_final, or spot-checked W_q/W_k rows).
- **Lazy daemon start**: If gateway restarts mid-session, `afterTurn` now starts the daemon on demand instead of silently skipping extraction.
- **`getOrCreateSession` in afterTurn**: Resumed sessions after gateway restart no longer return null.
- **Model object unwrapping**: `defaults.model` can be `{primary: "provider/model"}` — unwrap and split provider/model format.

### Infrastructure
- **CI pipeline**: GitHub Actions with SurrealDB service container, Node 22, 439 tests (unit + integration).
- **PR checks**: Type checking + unit tests on all pull requests.

### Tests
- 415 → 439 tests. New: ACAN NaN/Infinity validation (7), score stability/performance (3), `SELECT IN` integration test, additional integration coverage.

## [0.4.2] - 2026-04-03

### Performance
- **DB query batching**: `queryBatch()` sends N SQL statements in 1 round-trip. `graphExpand` (208 queries → 1-2/hop), `queryCausalContext` (120 → 1-2), `vectorSearch` (7 → 1).
- **Embedding reuse**: User embeddings from ingest stashed in session state and reused in retrieval, eliminating 1-4 redundant BGE-M3 calls per turn.
- **Token estimation**: Aligned with Claude Code — 4 bytes/token (was 3.4), JSON at 2 bytes/token, images at 2000 tokens, 33% safety margin.
- **Content stripping**: Old thinking blocks, images, tool results, and assistant filler text surgically replaced with compact stubs. Saves 20-80k tokens/session.
- **Prompt compression**: Rules suffix (~400 → ~80 tokens), planning gate (~250 → ~60), IKONG description (~120 → ~20), cognitive check (~300 → ~120).
- **Structured output**: All internal LLM calls use `json_schema` output format when supported. Eliminates markdown fencing and preamble.
- **Budget model**: 4-way split (conversation 23%, retrieval 38.5%, core 15.5%, tools 23%) with SPA cap at 8% of context window.
- **Parallel DB calls**: `scoreResults` parallelizes utility cache + reflection session lookups. Single `getSessionTurns` fetch in `afterTurn` reused by all consumers.
- **Tier 0 dedup**: Core memory fetched once per `assemble()`, passed to inner transform (was fetched twice).
- **Cognitive check frequency**: Every 5 turns (was 3), skipped when `skipRetrieval=true`.

### Security
- **Edge name validation**: `VALID_EDGES` whitelist + `assertValidEdge()` prevents SQL injection via edge interpolation in `graphExpand` and `queryCausalContext`.

### Bug Fixes
- Tool limit enforcement: `>` → `>=` (was allowing 1 extra call past limit).
- Daemon batch merge instead of overwrite (prevents turn data loss when batches arrive faster than extraction).
- Reflection dedup: `typeof` check on score (prevents undefined bypass creating duplicates).
- Extraction fallback: Warns on no-JSON and regex fallback failure (was silent).
- Shutdown errors logged instead of swallowed.
- Config comment: `midSessionCleanupThreshold` documented as 100k, actual default 25k — fixed.
- Cognitive bootstrap importance: Float scale (0.85) → integer scale (9) matching rest of codebase.

### Documentation
- JSDoc on all critical exported functions.
- Named constants replacing magic numbers (`DEDUP_COSINE_THRESHOLD`, `MAX_FRONTIER_SEEDS`, `EDGE_NEIGHBOR_LIMIT`, etc.).
- README: Added Performance section with batching, estimation, stripping, and structured output details.

### Tests
- 88 → 415 tests (21 test files). Full coverage: ACAN scorer, hooks, memory daemon extraction, skills, soul system, wakeup, concept extraction, session persistence, tools, subagent lifecycle, and SurrealDB integration tests against live database.

## [0.4.1] - 2026-04-02

### Performance
- Inline intent classification: Parse LOOKUP/EDIT/REFACTOR from assistant text to set tool limits without extra LLM call.
- Default tool budget reduced to 9.
- LRU embedding cache (512 entries).
- System prompt caching split (static vs dynamic sections).
- Token delta computation (prevent quadratic overcounting).
- Concept backfilling with embedding similarity.

### Bug Fixes
- 17 bugs resolved from deep codebase review.

## [0.4.0] - 2026-04-01

### Features
- Spawned subagent edge wiring.
- Soul graduation in mid-session cleanup.
- Fibonacci memory resurfacing.
- ACAN (Attentive Cross-Attention Network) for learned retrieval scoring.
- Cognitive checks with directive injection.
- Handoff file emergency persistence.

## [0.3.x] - 2026-03

Initial release series. Graph-backed persistent memory engine with SurrealDB, BGE-M3 embeddings, 9-type knowledge extraction, soul graduation system, and adaptive intent classification.
