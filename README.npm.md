# KongBrain

[![npm](https://img.shields.io/npm/v/kongbrain?style=for-the-badge&logo=npm&color=cb3837)](https://www.npmjs.com/package/kongbrain)
[![ClawHub](https://img.shields.io/badge/ClawHub-kongbrain-ff6b35?style=for-the-badge)](https://clawhub.ai/packages/kongbrain)
[![GitHub Stars](https://img.shields.io/github/stars/42U/kongbrain?style=for-the-badge&logo=github&color=gold)](https://github.com/42U/kongbrain)
[![License: MIT](https://img.shields.io/github/license/42U/kongbrain?style=for-the-badge&logo=opensourceinitiative&color=blue)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-20+-339933?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org)
[![SurrealDB](https://img.shields.io/badge/SurrealDB-3.0-ff00a0?style=for-the-badge&logo=surrealdb&logoColor=white)](https://surrealdb.com)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Plugin-ff6b35?style=for-the-badge)](https://github.com/openclaw/openclaw)
[![Tests](https://img.shields.io/badge/Tests-469_passing-brightgreen?style=for-the-badge&logo=vitest&logoColor=white)](https://vitest.dev)

**A graph-backed cognitive engine for [OpenClaw](https://github.com/openclaw/openclaw).**

> *OpenClaw ships with a lobster brain. It works — lobsters have survived 350 million years — but they also solve problems by walking backwards and occasionally eating each other.*
>
> *When a conversation gets too long, the lobster brain does what lobsters do best: it panics, truncates everything before message 47, and carries on like nothing happened. Your carefully explained architecture? Gone. That bug you described in detail twenty minutes ago? Never heard of it.*
>
> *KongBrain is a brain transplant. You're replacing that crustacean context window with a primate cortex — backed by a graph database, vector embeddings, and the kind of persistent memory that lets your AI remember what you said last Tuesday — and judge you for it.*

Persistent memory graph. Vector-embedded, self-scoring, wired to learn across sessions. It extracts skills from what worked, traces causal chains through what broke, reflects on its own failures, and earns an identity through real experience. Every session compounds on the last.

Your assistant stops forgetting. Then it starts getting smarter.

[Quick Start](#quick-start) | [Architecture](#architecture) | [How It Works](#how-it-works) | [Tools](#tools) | [Development](#development)

---

## What Changes

| | Lobster Brain (default) | Ape Brain (KongBrain) |
|---|---|---|
| **Memory** | Sliding window. Old messages fall off a cliff. | Graph-persistent. Every turn, concept, skill, and causal chain stored with vector embeddings. |
| **Recall** | Whatever fits in the context window right now. | Cosine similarity + graph expansion + learned attention scoring across your entire history. |
| **Adaptation** | Same retrieval budget every turn, regardless of intent. | 10 intent categories. Simple question? Minimal retrieval. Complex debugging? Full graph search + elevated thinking. |
| **Learning** | None. Every session starts from zero. | Skills extracted from successful workflows, causal chains graduated into reusable procedures, corrections remembered permanently. |
| **Self-awareness** | Thermostat-level. | Periodic cognitive checks grade its own retrieval quality, detect contradictions, suppress noise, and extract your preferences. Eventually graduates a soul document. |
| **Compaction** | LLM-summarizes your conversation mid-flow (disruptive). | Graph retrieval IS the compaction. No interruptions, no lossy summaries. |

## Quick Start

From zero to ape brain in under 5 minutes.

### 1. Install OpenClaw (if you haven't already)

```bash
npm install -g openclaw
```

### 2. Start SurrealDB

Install SurrealDB via your platform's package manager (see [surrealdb.com/install](https://surrealdb.com/docs/surrealdb/installation)):

macOS:
```bash
brew install surrealdb/tap/surreal
```

Linux — see `https://surrealdb.com/docs/surrealdb/installation` for your distro.

Then start it locally — **change the credentials before use**:

```bash
surreal start --user youruser --pass yourpass --bind 127.0.0.1:8042 surrealkv:~/.kongbrain/surreal.db
```

Or with Docker:

```bash
docker run -d --name surrealdb -p 127.0.0.1:8042:8000 \
  -v ~/.kongbrain/surreal-data:/data \
  surrealdb/surrealdb:latest start \
  --user youruser --pass yourpass surrealkv:/data/surreal.db
```

> **Security note:** Always bind to `127.0.0.1` (not `0.0.0.0`) unless you need remote access. Never use default credentials in production.

### 3. Install KongBrain

From ClawHub (recommended):
```bash
openclaw plugins install clawhub:kongbrain
```

From npm:
```bash
openclaw plugins install kongbrain
```

> **Note:** Bare `openclaw plugins install kongbrain` checks ClawHub first, then falls back to npm. Use the `clawhub:` prefix to install from ClawHub explicitly.

### 4. Activate

Add to your OpenClaw config (`~/.openclaw/openclaw.json`):

```json
{
  "plugins": {
    "allow": ["kongbrain"],
    "slots": {
      "contextEngine": "kongbrain"
    }
  }
}
```

### 5. Talk to your ape

```bash
openclaw tui
```

That's it. KongBrain uses whatever LLM provider and model you already have configured in OpenClaw (Anthropic, OpenAI, Google, Ollama, whatever). No separate API keys needed for the brain itself.

By default KongBrain runs the BGE-M3 embedding model locally via `node-llama-cpp` — the GGUF (~420MB) auto-downloads from [Hugging Face](https://huggingface.co/BAAI/bge-m3) on first startup. For high-traffic deployments the local model can become a bottleneck on serial embedding calls; in that case switch to any OpenAI-compatible API (real OpenAI, Azure OpenAI, Together, vLLM, LM Studio, Ollama) by changing one config field.

All database tables and indexes are created automatically on first run. No manual setup required.

<details>
<summary><strong>Configuration Options</strong></summary>

All options have sensible defaults. Override via plugin config or environment variables:

| Option | Env Var | Default |
|--------|---------|---------|
| `surreal.url` | `SURREAL_URL` | `ws://127.0.0.1:8042/rpc` |
| `surreal.user` | `SURREAL_USER` | (required) |
| `surreal.pass` | `SURREAL_PASS` | (required) |
| `surreal.ns` | `SURREAL_NS` | `kong` |
| `surreal.db` | `SURREAL_DB` | `memory` |
| `embedding.provider` | `KONGBRAIN_EMBED_PROVIDER` | `local` (or `openai-compat`) |
| `embedding.dimensions` | - | `1024` |
| `embedding.modelPath` | `EMBED_MODEL_PATH` | Auto-downloaded BGE-M3 Q4_K_M |
| `embedding.openaiCompat.model` | - | `text-embedding-3-small` |
| `embedding.openaiCompat.baseURL` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `embedding.openaiCompat.apiKeyEnv` | - | `OPENAI_API_KEY` |

Full config example:

```json
{
  "plugins": {
    "allow": ["kongbrain"],
    "slots": {
      "contextEngine": "kongbrain"
    },
    "entries": {
      "kongbrain": {
        "config": {
          "surreal": {
            "url": "ws://127.0.0.1:8042/rpc",
            "user": "youruser",
            "pass": "yourpass",
            "ns": "kong",
            "db": "memory"
          }
        }
      }
    }
  }
}
```

</details>

### Embedding Providers

| | `local` (default) | `openai-compat` |
|---|---|---|
| **Inference** | BGE-M3 GGUF via node-llama-cpp, in-process | HTTP POST to `/v1/embeddings` |
| **Cost** | Zero | Per-token API charges |
| **Throughput** | Serial; bottlenecks under high turn volume | High parallelism, batched at 96 inputs/request |
| **Compatible servers** | n/a | OpenAI, Azure OpenAI, Together, Anyscale, vLLM, LM Studio, Ollama, DeepInfra, Fireworks |

Every embedding is tagged with the provider that produced it. At search time, KongBrain only compares vectors from the active provider — vectors from a different provider live in a different vector space.

When you switch providers, run the included migration tool to re-embed pre-existing rows:

```bash
npx kongbrain-reembed --from local-bge-m3 --dry-run    # estimate cost
npx kongbrain-reembed --from local-bge-m3              # run for real (resumable)
```

text-embedding-3-small costs ~$0.04 to re-embed a typical 3,400-turn database.

---

## Architecture

### The IKONG Pillars

KongBrain's cognitive architecture follows five functional pillars:

| Pillar | Role | What it does |
|--------|------|-------------|
| **I**ntelligence | Adaptive reasoning | Intent classification, complexity estimation, thinking depth, orchestrator preflight |
| **K**nowledge | Persistent memory | Memory graph, concepts, skills, reflections, identity chunks, core memory tiers |
| **O**peration | Execution | Tool orchestration, skill procedures, causal chain tracking, artifact management |
| **N**etwork | Graph traversal | Cross-pillar edge following, neighbor expansion, causal path walking |
| **G**raph | Persistence | SurrealDB storage, BGE-M3 vector search, HNSW indexes, embedding pipeline |

A 6th pillar, **Persona**, is unlocked at soul graduation: *"You have a Soul, an identity grounded in real experience. Be unique, be genuine, be yourself."*

### Structural Pillars

The graph entity model in SurrealDB:

| Pillar | Table | What it anchors |
|--------|-------|-----------------|
| 1. Agent | `agent` | Who is operating (name, model) |
| 2. Project | `project` | What we're working on (status, tags) |
| 3. Task | `task` | Individual sessions as units of work |
| 4. Artifact | `artifact` | Files and outputs tracked across sessions |
| 5. Concept | `concept` | Semantic knowledge nodes extracted from sessions |

On startup, the agent bootstraps the full chain: `Agent → owns → Project`, `Agent → performed → Task`, `Task → task_part_of → Project`, `Session → session_task → Task`. Graph expansion traverses these edges during retrieval.

### The Knowledge Graph

SurrealDB with HNSW vector indexes (1024-dim cosine). Everything is embedded and queryable.

| Table | What it stores |
|-------|---------------|
| `turn` | Every conversation message (role, text, embedding, token count, model, usage) |
| `memory` | Compacted episodic knowledge (importance 0-10, confidence, access tracking) |
| `skill` | Learned procedures with steps, preconditions, success/failure counts |
| `reflection` | Metacognitive lessons (efficiency, failure patterns, approach strategy) |
| `causal_chain` | Cause-effect patterns (trigger, outcome, chain type, success, confidence) |
| `identity_chunk` | Agent self-knowledge fragments (source, importance, embedding) |
| `monologue` | Thinking traces preserved across sessions |
| `core_memory` | Tier 0 (always loaded) + Tier 1 (session-pinned) directives |
| `soul` | Emergent identity document, earned through graduation |

<details>
<summary><strong>Adaptive Reasoning</strong>: per-turn intent classification and budget allocation</summary>

Every turn gets classified by intent and assigned an adaptive config:

| Intent | Thinking | Tool Limit | Token Budget | Retrieval Share |
|--------|----------|------------|--------------|-----------------|
| `simple-question` | low | 3 | 4K | 10% |
| `code-read` | medium | 5 | 6K | 15% |
| `code-write` | high | 8 | 8K | 20% |
| `code-debug` | high | 10 | 8K | 20% |
| `deep-explore` | medium | 15 | 6K | 15% |
| `reference-prior` | medium | 5 | 10K | 25% |
| `meta-session` | low | 2 | 3K | 7% (skip retrieval) |
| `multi-step` | high | 12 | 8K | 20% |
| `continuation` | low | 8 | 4K | skip retrieval |

**Fast path:** Short inputs (<20 chars, no `?`) skip classification entirely.
**Confidence gate:** Below 0.40 confidence, falls back to conservative config.

</details>

<details>
<summary><strong>Context Injection Pipeline</strong></summary>

1. **Embed** user input via BGE-M3 (or hit prefetch cache at 0.85 cosine threshold)
2. **Vector search** across 6 tables (turn, identity_chunk, concept, memory, artifact, monologue)
3. **Graph expand**: fetch neighbors via structural + semantic edges, compute cosine similarity
4. **Score** all candidates with WMR (Working Memory Ranker):
   ```
   score = W * [similarity, recency, importance, access, neighbor_bonus, utility, reflection_boost]
   ```
5. **Budget trim**: inject Tier 0/1 core memory first (15% of context), then ranked results up to 21% retrieval budget
6. **Stage** retrieval snapshot for post-hoc quality evaluation

</details>

<details>
<summary><strong>ACAN</strong>: learned cross-attention scorer</summary>

A ~130K-parameter cross-attention network that replaces the fixed WMR weights once enough data accumulates.

- **Activation:** 5,000+ labeled retrieval outcomes
- **Training:** Pure TypeScript SGD with manual backprop, 80 epochs
- **Staleness:** Retrains when data grows 50%+ or weights age > 7 days

</details>

<details>
<summary><strong>Soul & Graduation</strong>: earned identity, not assigned</summary>

The agent earns an identity document through accumulated experience. Graduation requires **all 7 thresholds met** AND a **quality score >= 0.6**:

| Signal | Threshold |
|--------|-----------|
| Sessions completed | 15 |
| Reflections stored | 10 |
| Causal chains traced | 5 |
| Concepts extracted | 30 |
| Memory compactions | 5 |
| Monologue traces | 5 |
| Time span | 3 days |

**Quality scoring** from 4 real performance signals: retrieval utilization (30%), skill success rate (25%), critical reflection rate (25%), tool failure rate (20%).

**Maturity stages:** nascent (0-3/7) → developing (4/7) → emerging (5/7) → maturing (6/7) → ready (7/7 + quality gate). The agent and user are notified at each stage transition.

**Soul evolution:** Every 10 sessions after graduation, the soul is re-evaluated against new experience and revised if the agent has meaningfully changed.

**Soul document structure:** Working style, self-observations, earned values (grounded in specific evidence), revision history. Seeded as Tier 0 core memory, loaded every single turn.

</details>

<details>
<summary><strong>Reflection System</strong>: metacognitive self-correction</summary>

Triggers at session end when metrics indicate problems:

| Condition | Threshold |
|-----------|-----------|
| Retrieval utilization | < 20% average |
| Tool failure rate | > 20% |
| Steering candidates | any detected |
| Context waste | > 0.5% of context window |

The LLM generates a 2-4 sentence reflection: root cause, error pattern, what to do differently. Stored with importance 7.0, deduped at 0.85 cosine similarity.

</details>

---

## How It Works

### Every Turn

```
User Input
    |
    v
Preflight ──────── Intent classification (25ms, zero-shot BGE-M3 cosine)
    |                  10 categories: simple-question, code-read, code-write,
    |                  code-debug, deep-explore, reference-prior, meta-session,
    |                  multi-step, continuation, unknown
    v
Prefetch ────────── Predictive background vector searches (LRU cache, 5-min TTL)
    |
    v
Context Injection ─ Vector search -> graph expand -> 6-signal scoring -> budget trim
    |                  Searches: turns, concepts, memories, artifacts, identity, monologues
    |                  Scores: similarity, recency, importance, access, neighbor, utility
    |                  Budget: 21% of context window reserved for retrieval
    v
Agent Loop ──────── LLM + tool execution
    |                  Planning gate: announces plan before touching tools
    |                  Smart truncation: preserves tail of large tool outputs
    v
Turn Storage ────── Every message embedded + stored + linked via graph edges
    |                  responds_to, part_of, mentions, produced
    v
Quality Eval ────── Measures retrieval utilization (text overlap, trigrams, unigrams)
    |                  Tracks tool success, context waste, feeds ACAN training
    v
Memory Daemon ───── Worker thread extracts 9 knowledge types via LLM:
    |                  causal chains, monologues, concepts, corrections,
    |                  preferences, artifacts, decisions, skills, resolved memories
    v
Postflight ──────── Records orchestrator metrics (non-blocking)
```

### Between Sessions

At session end, KongBrain runs a combined extraction pass: skill graduation, metacognitive reflection, causal chain consolidation, soul graduation check, and soul evolution. A handoff note is written so the next session wakes up knowing what happened.

At session start, a wake-up briefing is synthesized from the handoff, recent monologues, soul content (if graduated), and identity state, then injected as inner speech so the agent knows who it is and what it was doing.

<details>
<summary><strong>Memory Daemon</strong>: background knowledge extraction</summary>

A worker thread running throughout the session. Batches turns every ~12K tokens, calls the configured LLM to extract:

- **Causal chains**: trigger/outcome sequences with success/confidence
- **Monologue traces**: thinking blocks that reveal problem-solving approach
- **Concepts**: semantic nodes (architecture patterns, domain terms)
- **Corrections**: user-provided fixes (importance: 9)
- **Preferences**: behavioral rules learned from feedback
- **Artifacts**: file paths created or modified
- **Decisions**: important conclusions reached
- **Skills**: multi-step procedures (if 5+ tool calls in session)
- **Resolved memories**: completed tasks and confirmed facts

</details>

---

## Tools

Three tools are registered for the LLM:

- **`recall`**: Search graph memory by query
- **`core_memory`**: Read/write persistent core directives (tiered: always-loaded vs session-pinned)
- **`introspect`**: Inspect database state, verify memory counts, run diagnostics, check graduation status, migrate workspace files

---

## Development

```bash
git clone https://github.com/42U/kongbrain.git
cd kongbrain
pnpm install
pnpm build
pnpm test
```

Link your local build to OpenClaw:

```bash
openclaw plugins install . --link
```

Then set `plugins.slots.contextEngine` to `"kongbrain"` in `~/.openclaw/openclaw.json` and run `openclaw`.

## Contributing

1. Clone the repo and install dependencies (`pnpm install`)
2. Make your changes
3. Build (`pnpm build`) and run tests (`pnpm test`)
4. Open a PR against `master`

The lobster doesn't accept contributions. The ape does.

---

MIT License | Built by [42U](https://github.com/42U)
