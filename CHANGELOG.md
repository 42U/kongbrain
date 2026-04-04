# Changelog

All notable changes to KongBrain are documented here.

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
