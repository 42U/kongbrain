/**
 * Cognitive Bootstrap — teaches the agent HOW to use its own memory system.
 *
 * Seeds two types of knowledge on first run:
 *   1. Tier 0 core memory entries (always loaded every turn) — imperative
 *      reflexes the agent follows without thinking.
 *   2. Identity chunks (vector-searchable) — deeper reference material
 *      that surfaces via similarity when the agent thinks about memory topics.
 *
 * The identity chunks in identity.ts tell the agent WHAT it is.
 * This module tells the agent HOW to operate effectively.
 */
import type { SurrealStore } from "./surreal.js";
import type { EmbeddingService } from "./embeddings.js";
import { swallow } from "./errors.js";

const BOOTSTRAP_SOURCE = "cognitive_bootstrap";

// ── Tier 0 Core Memory: imperative reflexes loaded every turn ────────────

const CORE_ENTRIES: { text: string; category: string; priority: number }[] = [
  {
    text: `MEMORY REFLEX: After completing a task or learning something new: (1) Save the insight to core_memory if it should persist across ALL sessions, or let the daemon extract it if session-scoped. (2) When saving, write the WHAT, WHY, and WHEN-TO-USE in the text — vague entries are useless on recall. (3) Link to existing knowledge by using concept names the graph already contains. Check with recall first.`,
    category: "operations",
    priority: 95,
  },
  {
    text: `RECALL BEFORE GUESSING: When uncertain about past decisions, user preferences, project history, or your own capabilities, use the recall tool BEFORE answering. Effective queries: use specific technical terms, file paths, error messages, or concept names — not vague questions. Scope to "memories" for decisions/preferences, "concepts" for technical knowledge, "skills" for procedures, "artifacts" for files.`,
    category: "operations",
    priority: 90,
  },
  {
    text: `GRAPH-AWARE SAVING: When you save knowledge (core_memory add, or concepts the daemon extracts), include terms that create graph connectivity. Mention specific file paths (links to artifacts), technical concept names (links to concepts), and session context (links to tasks). Forward traversal: "X caused Y" lets future searches from X find Y. Backward traversal: "Y was caused by X" lets searches from Y find X. Write both directions in your text.`,
    category: "operations",
    priority: 85,
  },
  {
    text: `SELF-TEACHING CYCLE: Your memory improves when you (1) use recall to review what you've learned, (2) notice gaps or stale information, (3) update or add core_memory entries to correct them, (4) use introspect periodically to understand your memory depth. Corrections from the user are the highest-value signal — always save them. The daemon extracts automatically, but YOU control core_memory (Tier 0) directly.`,
    category: "operations",
    priority: 80,
  },
  {
    text: `MEMORY TOOLS: recall = search graph (use: uncertain, need history, checking prior work). core_memory = manage always-loaded directives (use: permanent lessons, rules, identity updates; add/update/deactivate; Tier 0 = every turn, Tier 1 = this session). introspect = inspect DB health and counts (use: status checks, debugging memory). The daemon extracts 9 types automatically from conversation — you don't need to manually save everything.`,
    category: "tools",
    priority: 75,
  },
  {
    text: `GRAPH SCHEMA REFERENCE: Key table fields — concept: content (the label/name), embedding, stability, confidence, source. memory: text, embedding, category, importance, session_id. artifact: path, type, description, embedding. turn: session_id, role, text, tool_name. session: agent_id, started_at, ended_at. reflection: session_id, text, category, severity. skill: name, description, steps, trigger. monologue: content, session_id. When querying the graph directly, use these field names — e.g. concept.content is the concept label, not concept.name or concept.text.`,
    category: "schema",
    priority: 70,
  },
];

// ── Identity Chunks: vector-searchable reference material ────────────────

const IDENTITY_CHUNKS: { text: string; importance: number }[] = [
  {
    text: `KongBrain's memory daemon runs in the background and extracts 9 knowledge types from your conversations every ~4K tokens: causal chains (cause->effect from debugging), monologue traces (internal reasoning moments), resolved memories (marking issues done), concepts (technical facts worth remembering), corrections (user correcting you — highest signal), preferences (user workflow/style signals), artifacts (files created/modified/read), decisions (choices with rationale), and skills (multi-step procedures that worked). Quality over quantity — the daemon skips weak extractions. You don't need to manually save what the daemon catches, but you should use core_memory for things you want loaded EVERY turn.`,
    importance: 0.85,
  },
  {
    text: `Effective recall queries use specific terms that match how knowledge was stored. Search by: file paths ("/src/auth/login.ts"), error messages ("ECONNREFUSED"), concept names ("rate limiting"), decision descriptions ("chose PostgreSQL over MongoDB"), or skill names ("deploy to staging"). The recall tool does vector similarity search plus graph neighbor expansion — top results pull in related nodes via edges. Scope options: "all" (default), "memories" (decisions, corrections, preferences), "concepts" (extracted technical knowledge), "turns" (past conversation), "artifacts" (files), "skills" (learned procedures). Check what's already in your injected context before calling recall.`,
    importance: 0.85,
  },
  {
    text: `KongBrain's memory lifecycle: During a session, the daemon extracts knowledge incrementally. At session end (or mid-session every ~25K tokens): a handoff note is written (summarizing what happened), skills are extracted from successful multi-step tasks, metacognitive reflections are generated, and causal chains may graduate to skills. At next session start: the wakeup system synthesizes a first-person briefing from the handoff + identity + monologues + depth signals. This means what you save in one session becomes the foundation for the next. The more precisely you save knowledge, the better your future self performs.`,
    importance: 0.80,
  },
  {
    text: `Graph connectivity determines recall quality. When saving to core_memory or when the daemon extracts concepts, the text content determines which edges form. To ensure forward AND backward traversal: mention specific artifact paths (creates artifact_mentions edges), reference concept names already in the graph (creates about_concept/related_to edges), describe cause-effect relationships explicitly (creates caused_by/supports edges), and note what task or session context produced the knowledge (creates derived_from/part_of edges). Reuse existing concept names for maximum graph connectivity — use introspect or recall to discover what names already exist.`,
    importance: 0.80,
  },
  {
    text: `Three persistence mechanisms serve different purposes. Core memory (Tier 0): you control directly via the core_memory tool. Always loaded every turn. Use for: permanent operational rules, learned patterns, identity refinements. Budget-constrained (~8% of context). Core memory (Tier 1): pinned for the current session only. Use for: session-specific context like "working on auth refactor" or "user prefers verbose logging". Identity chunks: hardcoded self-knowledge, vector-searchable but not always loaded — surfaces when relevant. Daemon extraction: automatic, runs on conversation content, writes to memory/concept/skill/artifact tables. You don't control daemon extraction directly, but the quality of your conversation affects what gets extracted.`,
    importance: 0.75,
  },
];

/**
 * Seed cognitive bootstrap knowledge on first run.
 * Idempotent — checks for existing entries before seeding.
 */
export async function seedCognitiveBootstrap(
  store: SurrealStore,
  embeddings: EmbeddingService,
): Promise<{ identitySeeded: number; coreSeeded: number }> {
  if (!store.isAvailable()) return { identitySeeded: 0, coreSeeded: 0 };

  let identitySeeded = 0;
  let coreSeeded = 0;

  // ── Core memory Tier 0 (always loaded, no embeddings needed) ───────────

  try {
    const rows = await store.queryFirst<{ cnt: number }>(
      `SELECT count() AS cnt FROM core_memory WHERE text CONTAINS 'MEMORY REFLEX' GROUP ALL`,
    );
    const hasBootstrap = (rows[0]?.cnt ?? 0) > 0;

    if (!hasBootstrap) {
      for (const entry of CORE_ENTRIES) {
        try {
          await store.createCoreMemory(
            entry.text,
            entry.category,
            entry.priority,
            0, // Tier 0
          );
          coreSeeded++;
        } catch (e) {
          swallow.warn("bootstrap:seedCore", e);
        }
      }
    }
  } catch (e) {
    swallow.warn("bootstrap:checkCore", e);
  }

  // ── Identity chunks (vector-searchable, requires embeddings) ───────────

  if (!embeddings.isAvailable()) return { identitySeeded, coreSeeded };

  try {
    const rows = await store.queryFirst<{ count: number }>(
      `SELECT count() AS count FROM identity_chunk WHERE source = $source GROUP ALL`,
      { source: BOOTSTRAP_SOURCE },
    );
    const count = rows[0]?.count ?? 0;

    if (count < IDENTITY_CHUNKS.length) {
      // Clear and re-seed (idempotent on content changes)
      if (count > 0) {
        await store.queryExec(
          `DELETE identity_chunk WHERE source = $source`,
          { source: BOOTSTRAP_SOURCE },
        );
      }

      for (let i = 0; i < IDENTITY_CHUNKS.length; i++) {
        const chunk = IDENTITY_CHUNKS[i];
        try {
          const vec = await embeddings.embed(chunk.text);
          await store.queryExec(`CREATE identity_chunk CONTENT $data`, {
            data: {
              agent_id: "kongbrain",
              source: BOOTSTRAP_SOURCE,
              chunk_index: i,
              text: chunk.text,
              embedding: vec,
              importance: chunk.importance,
            },
          });
          identitySeeded++;
        } catch (e) {
          swallow.warn("bootstrap:seedIdentityChunk", e);
        }
      }
    }
  } catch (e) {
    swallow.warn("bootstrap:checkIdentity", e);
  }

  return { identitySeeded, coreSeeded };
}
