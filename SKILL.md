---
name: laqrumbrain
description: Graph-backed persistent memory engine for OpenClaw. Replaces the default context window with SurrealDB + vector embeddings that learn across sessions.
version: 0.5.2
homepage: https://github.com/42U/laqrumbrain
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
      optionalEnv:
        - LAQRUMBRAIN_EMBED_PROVIDER
        - EMBED_MODEL_PATH
        - OPENAI_BASE_URL
        - OPENAI_API_KEY
    primaryEnv: SURREAL_URL
    install:
      - kind: node
        package: laqrumbrain
        bins: []
---

# LaqrumBrain

Graph-backed persistent memory engine for OpenClaw. Replaces the default context window with SurrealDB + vector embeddings that learn across sessions.

## What it does

LaqrumBrain gives your OpenClaw agent persistent, structured memory:

- **Session tracking** - records conversations and extracts knowledge automatically
- **9 memory categories** - knowledge, goals, reflections, handoffs, corrections, preferences, decisions, skills, and causal chains
- **Vector search** - BGE-M3 embeddings for semantic recall
- **Graph relationships** - memories linked via SurrealDB graph edges for traversal
- **Tiered memory** - core memories always loaded, session memories pinned, rest searched on demand
- **Mid-session extraction** - extracts knowledge during conversation, not just at exit
- **Crash resilience** - deferred cleanup processes orphaned sessions on next startup

## Requirements

- **SurrealDB** - running instance (local or remote)
- **Node.js** >= 18

## Setup

### Install SurrealDB

See the official install guide: https://surrealdb.com/docs/surrealdb/installation

macOS:
```bash
brew install surrealdb/tap/surreal
```

Linux — see `https://surrealdb.com/docs/surrealdb/installation` for your distro.

Docker:
```bash
docker pull surrealdb/surrealdb:latest
```

### Start SurrealDB

Local only (recommended) — use strong credentials in production:
```bash
surreal start --user youruser --pass yourpass --bind 127.0.0.1:8000 surrealkv:~/.laqrumbrain/surreal.db
```

> **Security note:** Always bind to `127.0.0.1` (not `0.0.0.0`) unless you specifically need remote access. Change the default credentials before use.

For Docker:

```bash
docker run -d --name surrealdb -p 127.0.0.1:8000:8000 \
  -v ~/.laqrumbrain/surreal-data:/data \
  surrealdb/surrealdb:latest start \
  --user youruser --pass yourpass surrealkv:/data/surreal.db
```

## Configuration

Set environment variables or provide a `.env` file:

```
SURREAL_URL=ws://127.0.0.1:8000/rpc
SURREAL_USER=youruser
SURREAL_PASS=yourpass
SURREAL_NS=laqrumbrain
SURREAL_DB=laqrumbrain
```

## Usage

Install as an OpenClaw plugin:

```bash
openclaw plugins install clawhub:laqrumbrain
```

Or via npm:

```bash
npm install laqrumbrain
```

The BGE-M3 embedding model (~420MB) downloads automatically on first startup from Hugging Face (https://huggingface.co/BAAI/bge-m3). All database tables and indexes are created automatically on first run.

LaqrumBrain hooks into OpenClaw's plugin lifecycle automatically. Memory extraction runs in the background via a daemon worker thread.
