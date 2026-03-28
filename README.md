# KongBrain

OpenClaw ships with a lobster brain. It works — lobsters have survived 350 million years — but they also solve problems by walking backwards and occasionally eating each other. When a conversation gets too long, the lobster brain does what lobsters do best: it panics, truncates everything before message 47, and carries on like nothing happened. Your carefully explained architecture? Gone. That bug you described in detail twenty minutes ago? Never heard of it.

KongBrain is a brain transplant. You're replacing that crustacean context window with a primate cortex backed by a 46-table graph database, vector embeddings, and the kind of persistent memory that lets your AI remember what you said last Tuesday — and judge you for it.

Apes remember. Apes use tools. Apes hold grudges about your code style and learn from them. Lobsters eat garbage off the ocean floor and forget about it immediately.

The surgery takes about 2 minutes. No anesthesia required.

---

## What Changes

| | Lobster Brain (default) | Ape Brain (KongBrain) |
|---|---|---|
| **Memory** | Sliding window. Old messages fall off a cliff. | Graph-persistent. Every turn, concept, skill, and causal chain stored with vector embeddings. |
| **Recall** | Whatever fits in the context window right now. | Cosine similarity + graph expansion + learned attention scoring across your entire history. |
| **Adaptation** | Same retrieval budget every turn, regardless of intent. | 10 intent categories. Simple question? Minimal retrieval. Complex debugging? Full graph search + elevated thinking. |
| **Learning** | None. Every session starts from zero. | Skills extracted from successful workflows, causal chains graduated into reusable procedures, corrections remembered permanently. |
| **Self-awareness** | Thermostat-level. | Periodic cognitive checks grade its own retrieval quality, detect contradictions, suppress noise, and extract your preferences. Eventually graduates a soul document. |
| **Compaction** | LLM-summarizes your conversation mid-flow (disruptive). | Graph retrieval IS the compaction — no interruptions, no lossy summaries. |

## Quick Start

From zero to ape brain in under 5 minutes.

### 1. Install OpenClaw (if you haven't already)

```bash
npm install -g openclaw
```

### 2. Start SurrealDB

Pick one:

```bash
# Native install
curl -sSf https://install.surrealdb.com | sh
surreal start --user root --pass root --bind 0.0.0.0:8042 file:~/.kongbrain/surreal.db
```

```bash
# Docker
docker run -d --name surrealdb -p 8042:8000 \
  -v ~/.kongbrain/surreal-data:/data \
  surrealdb/surrealdb:latest start \
  --user root --pass root file:/data/surreal.db
```

### 3. Install KongBrain

```bash
openclaw plugins install @openclaw/kongbrain
```

### 4. Activate

Add to your OpenClaw config (`~/.openclaw/config.json`):

```json
{
  "contextEngine": "kongbrain"
}
```

### 5. Talk to your ape

```bash
openclaw
```

That's it. KongBrain uses whatever LLM provider and model you already have configured in OpenClaw — Anthropic, OpenAI, Google, Ollama, whatever. No separate API keys needed for the brain itself.

The BGE-M3 embedding model (~420MB) downloads automatically on first startup. All 46 database tables and indexes are created automatically on first run. No manual setup required.

### Options

All options have sensible defaults. Override via plugin config or environment variables:

| Option | Env Var | Default |
|--------|---------|---------|
| `surreal.url` | `SURREAL_URL` | `ws://localhost:8042/rpc` |
| `surreal.user` | `SURREAL_USER` | `root` |
| `surreal.pass` | `SURREAL_PASS` | `root` |
| `surreal.ns` | `SURREAL_NS` | `kong` |
| `surreal.db` | `SURREAL_DB` | `memory` |
| `embedding.modelPath` | `KONGBRAIN_EMBEDDING_MODEL` | Auto-downloaded BGE-M3 Q4_K_M |
| `embedding.dimensions` | - | `1024` |

Full config example:

```json
{
  "contextEngine": "kongbrain",
  "plugins": {
    "entries": {
      "kongbrain": {
        "config": {
          "surreal": {
            "url": "ws://localhost:8042/rpc",
            "user": "root",
            "pass": "root",
            "ns": "kong",
            "db": "memory"
          }
        }
      }
    }
  }
}
```

## How It Works

### Every Turn

```
User query
  -> Intent classification (zero-shot BGE-M3, 10 categories)
  -> Orchestrator preflight (adapts retrieval budget, thinking level, tool limits)
  -> Graph retrieval (vector search + HNSW + graph expansion + edge traversal)
  -> Scoring (WMR weighted signals + ACAN learned attention)
  -> Token-constrained selection
  -> Context injection (system prompt addition — no fake messages)
```

### Between Turns

A background memory daemon (worker thread) incrementally extracts 9 knowledge types from your conversation: causal chains, monologue traces, concepts, corrections, preferences, artifacts, decisions, skills, and resolved-memory markers.

### Between Sessions

At session end, KongBrain runs a combined extraction pass: skill graduation, metacognitive reflection, causal chain consolidation, and (eventually) soul graduation. A handoff note is written so the next session wakes up knowing what happened.

At session start, a wake-up briefing is synthesized from the handoff, recent monologues, and identity state — injected as inner speech so the agent knows who it is and what it was doing.

### Tools

Three tools are registered for the LLM:

- **`recall`** — Search graph memory by query
- **`core_memory`** — Read/write persistent core directives (tiered: always-loaded vs session-pinned)
- **`introspect`** — Inspect database state, verify memory counts, run diagnostics

## Architecture

Persistence: SurrealDB graph with 46 tables, HNSW indexes on 1024-dim embeddings, edge relations (`part_of`, `mentions`, `responds_to`, `causes`, `supersedes`).

Intelligence: Intent-adaptive orchestration, 6-signal retrieval quality evaluation, predictive prefetch, periodic cognitive checks with LLM-judged relevance grades, fibonacci resurfacing for proactive memory.

Identity: Seeded identity chunks -> earned soul document. The soul is written BY the agent based on its own graph data once graduation thresholds are met (minimum sessions, memory count, reflection depth, skill count, causal chains).

## From Source

If you want to install from a local checkout instead of npm:

```bash
git clone https://github.com/42U/kongbrain.git
cd kongbrain
pnpm install
pnpm build
openclaw plugins link .
```

Then activate it the same way — set `"contextEngine": "kongbrain"` in your OpenClaw config.

## Development

```bash
git clone https://github.com/42U/kongbrain.git
cd kongbrain
pnpm install
pnpm build
```

Run tests:

```bash
pnpm test
```

Run OpenClaw against your local build:

```bash
openclaw plugins link .
openclaw
```

## Contributing

1. Clone the repo and install dependencies (`pnpm install`)
2. Make your changes
3. Build (`pnpm build`) and run tests (`pnpm test`)
4. Open a PR against `master`

The lobster doesn't accept contributions. The ape does.

## License

Same as OpenClaw.
