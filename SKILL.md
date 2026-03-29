---
name: kongbrain
description: Graph-backed persistent memory engine for OpenClaw. Replaces the default context window with SurrealDB + vector embeddings that learn across sessions.
version: 0.3.3
emoji: 🧠
homepage: https://github.com/42U/kongbrain
metadata:
  openclaw:
    requires:
      bins:
        - surreal
      env:
        - SURREAL_URL
        - SURREAL_USER
        - SURREAL_PASS
        - SURREAL_NS
        - SURREAL_DB
    primaryEnv: SURREAL_URL
    install:
      - kind: node
        package: kongbrain
        bins: []
---

# KongBrain

Graph-backed persistent memory engine for OpenClaw. Replaces the default context window with SurrealDB + vector embeddings that learn across sessions.

## What it does

KongBrain gives your OpenClaw agent persistent, structured memory:

- **Session tracking** — records conversations and extracts knowledge automatically
- **9 memory categories** — knowledge, goals, reflections, handoffs, corrections, preferences, decisions, skills, and causal chains
- **Vector search** — BGE-M3 embeddings for semantic recall
- **Graph relationships** — memories linked via SurrealDB graph edges for traversal
- **Tiered memory** — core memories always loaded, session memories pinned, rest searched on demand
- **Mid-session extraction** — extracts knowledge during conversation, not just at exit
- **Crash resilience** — deferred cleanup processes orphaned sessions on next startup

## Requirements

- **SurrealDB** — running instance (local or remote)
- **Ollama** or local BGE-M3 model for embeddings
- **Node.js** >= 18

## Configuration

Set environment variables or provide a `.env` file:

```
SURREAL_URL=ws://localhost:8000/rpc
SURREAL_USER=root
SURREAL_PASS=root
SURREAL_NS=kongbrain
SURREAL_DB=kongbrain
```

## Usage

Install as an OpenClaw plugin:

```bash
clawhub install kongbrain
```

Or via npm:

```bash
npm install kongbrain
```

KongBrain hooks into OpenClaw's plugin lifecycle automatically. Memory extraction runs in the background via a daemon worker thread using a tiered model hierarchy (Opus for chat, Sonnet for medium tasks, Haiku for simple extraction).
