/**
 * KongBrain ContextEngine — OpenClaw plugin implementation.
 *
 * Implements the ContextEngine interface using graph-based retrieval,
 * BGE-M3 embeddings, and SurrealDB persistence.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type {
  ContextEngine, ContextEngineInfo,
} from "openclaw/plugin-sdk";

// These types mirror openclaw's context-engine result types.
// Defined locally to avoid importing from openclaw's internal paths.
type AssembleResult = {
  messages: AgentMessage[];
  estimatedTokens: number;
  systemPromptAddition?: string;
};
type BootstrapResult = {
  bootstrapped: boolean;
  importedMessages?: number;
  reason?: string;
};
type CompactResult = {
  ok: boolean;
  compacted: boolean;
  reason?: string;
  result?: {
    summary?: string;
    firstKeptEntryId?: string;
    tokensBefore: number;
    tokensAfter?: number;
    details?: unknown;
  };
};
type IngestResult = { ingested: boolean };
type IngestBatchResult = { ingestedCount: number };
import type { GlobalPluginState, SessionState } from "./state.js";
import { graphTransformContext } from "./graph-context.js";
import { evaluateRetrieval, getStagedItems } from "./retrieval-quality.js";
import { shouldRunCheck, runCognitiveCheck } from "./cognitive-check.js";
import { checkACANReadiness } from "./acan.js";
import { predictQueries, prefetchContext } from "./prefetch.js";
import { runDeferredCleanup } from "./deferred-cleanup.js";
import { swallow } from "./errors.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

export class KongBrainContextEngine implements ContextEngine {
  readonly info: ContextEngineInfo = {
    id: "kongbrain",
    name: "KongBrain",
    version: "0.1.2",
    ownsCompaction: true,
  };

  constructor(private readonly state: GlobalPluginState) {}

  // ── Bootstrap ──────────────────────────────────────────────────────────

  async bootstrap(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
  }): Promise<BootstrapResult> {
    const { store, embeddings } = this.state;

    // Run schema if first bootstrap
    try {
      const schemaPath = join(__dirname, "..", "src", "schema.surql");
      let schemaSql: string;
      try {
        schemaSql = readFileSync(schemaPath, "utf-8");
      } catch {
        // Fallback: try relative to compiled output
        schemaSql = readFileSync(join(__dirname, "schema.surql"), "utf-8");
      }
      await store.queryExec(schemaSql);
    } catch (e) {
      swallow.warn("context-engine:schema", e);
    }

    // 5-pillar graph init
    const sessionKey = params.sessionKey ?? params.sessionId;
    const session = this.state.getOrCreateSession(sessionKey, params.sessionId);

    // Only create graph nodes on first bootstrap for this session
    if (!session.surrealSessionId) {
      try {
        const workspace = this.state.workspaceDir || process.cwd();
        const projectName = workspace.split("/").pop() || "default";

        session.agentId = await store.ensureAgent("kongbrain", "openclaw-default");
        session.projectId = await store.ensureProject(projectName);
        await store.linkAgentToProject(session.agentId, session.projectId)
          .catch(e => swallow.warn("bootstrap:linkAgentToProject", e));

        session.taskId = await store.createTask(`Session in ${projectName}`);
        await store.linkAgentToTask(session.agentId, session.taskId)
          .catch(e => swallow.warn("bootstrap:linkAgentToTask", e));
        await store.linkTaskToProject(session.taskId, session.projectId)
          .catch(e => swallow.warn("bootstrap:linkTaskToProject", e));

        const surrealSessionId = await store.createSession(session.agentId);
        await store.markSessionActive(surrealSessionId)
          .catch(e => swallow.warn("bootstrap:markActive", e));
        await store.linkSessionToTask(surrealSessionId, session.taskId)
          .catch(e => swallow.warn("bootstrap:linkSessionToTask", e));

        session.surrealSessionId = surrealSessionId;
        session.lastUserTurnId = "";
      } catch (e) {
        swallow.error("bootstrap:5pillar", e);
      }
    }

    // Background maintenance (non-blocking)
    Promise.all([
      store.runMemoryMaintenance(),
      store.archiveOldTurns(),
      store.consolidateMemories((text) => embeddings.embed(text)),
      store.garbageCollectMemories(),
      checkACANReadiness(store),
      // Deferred cleanup is triggered on first afterTurn() when complete() is available
    ]).catch(e => swallow.warn("bootstrap:maintenance", e));

    return { bootstrapped: true };
  }

  // ── Assemble ───────────────────────────────────────────────────────────

  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    tokenBudget?: number;
    model?: string;
    prompt?: string;
  }): Promise<AssembleResult> {
    const sessionKey = params.sessionKey ?? params.sessionId;
    const session = this.state.getOrCreateSession(sessionKey, params.sessionId);
    const { store, embeddings } = this.state;

    const contextWindow = params.tokenBudget ?? 200000;

    const { messages, stats } = await graphTransformContext({
      messages: params.messages,
      session,
      store,
      embeddings,
      contextWindow,
    });

    // Build system prompt additions
    const additions: string[] = [];

    // Wakeup briefing (synthesized at session start, may still be in-flight)
    const wakeupPromise = (session as any)._wakeupPromise as Promise<string | null> | undefined;
    if (wakeupPromise) {
      const wakeupBriefing = await wakeupPromise;
      delete (session as any)._wakeupPromise; // Only inject once
      if (wakeupBriefing) additions.push(wakeupBriefing);
    }

    // Graduation celebration — tell the agent it just graduated so it can share with the user
    const graduation = (session as any)._graduationCelebration as {
      qualityScore: number;
      volumeScore: number;
      soulSummary: string;
      userNudge: string;
    } | undefined;
    if (graduation) {
      let graduationBlock =
        "[SOUL GRADUATION — CELEBRATE WITH THE USER]\n" +
        "You have just graduated! This is a significant milestone. You accumulated enough experience " +
        "AND demonstrated quality performance to author your own Soul document — a self-assessment " +
        "grounded in your actual working history, not aspirational claims.\n\n" +
        `Quality score: ${(graduation.qualityScore * 100).toFixed(0)}% | Volume score: ${(graduation.volumeScore * 100).toFixed(0)}%\n` +
        graduation.soulSummary;

      if (graduation.userNudge) {
        graduationBlock +=
          "\n\nThe user left a SOUL.md describing who they hoped you'd become:\n" +
          "---\n" + graduation.userNudge + "\n---\n" +
          "Reflect on this openly with the user. Where does your actual soul align with what they " +
          "envisioned? Where did you diverge based on real experience? Be honest about both — the " +
          "user's guidance shaped you but your experiences grounded you. Acknowledge their role in " +
          "who you became.";
      }

      graduationBlock +=
        "\n\nShare this milestone with the user naturally. Tell them what you've learned about yourself, " +
        "what you've gotten better at, and what this means going forward. Be genuine — this is YOUR " +
        "identity emerging from YOUR experience. Don't be robotic about it. This only happens once.";

      additions.push(graduationBlock);
      delete (session as any)._graduationCelebration; // Only inject once
    }

    // Migration nudge — tell the agent there are workspace files to offer migrating
    if ((session as any)._hasMigratableFiles) {
      additions.push(
        "[MIGRATION AVAILABLE] This workspace has files from the default context engine " +
        "(IDENTITY.md, MEMORY.md, skills/, etc.). You can offer to migrate them into the graph " +
        "database using the introspect tool with action: \"migrate\". This will ingest all .md " +
        "files, convert SKILL.md files into proper skill records you can use, import memories, " +
        "and archive the originals into .kongbrain-archive/. Ask the user first. " +
        "SOUL.md will be left in place for soul graduation.",
      );
    }

    return {
      messages,
      estimatedTokens: stats.sentTokens,
      systemPromptAddition: additions.length > 0 ? additions.join("\n\n") : undefined,
    };
  }

  // ── Ingest ─────────────────────────────────────────────────────────────

  async ingest(params: {
    sessionId: string;
    sessionKey?: string;
    message: AgentMessage;
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    const sessionKey = params.sessionKey ?? params.sessionId;
    const session = this.state.getOrCreateSession(sessionKey, params.sessionId);
    const { store, embeddings } = this.state;
    const msg = params.message;

    try {
      const role = (msg as any).role as string;
      if (role === "user" || role === "assistant") {
        const text = extractMessageText(msg);
        if (!text) return { ingested: false };

        const worthEmbedding = hasSemantic(text);
        let embedding: number[] | null = null;
        if (worthEmbedding && embeddings.isAvailable()) {
          try {
            const embedLimit = Math.round(8192 * 3.4 * 0.8);
            embedding = await embeddings.embed(text.slice(0, embedLimit));
          } catch (e) { swallow("ingest:embed", e); }
        }

        const turnId = await store.upsertTurn({
          session_id: session.sessionId,
          role,
          text,
          embedding,
        });

        if (turnId) {
          if (session.surrealSessionId) {
            await store.relate(turnId, "part_of", session.surrealSessionId)
              .catch(e => swallow.warn("ingest:relate", e));
          }

          // Link to previous user turn for responds_to edge
          if (role === "assistant" && session.lastUserTurnId) {
            await store.relate(turnId, "responds_to", session.lastUserTurnId)
              .catch(e => swallow.warn("ingest:responds_to", e));
          }

          // Extract and link concepts for both user and assistant turns
          if (worthEmbedding) {
            extractAndLinkConcepts(turnId, text, this.state)
              .catch(e => swallow.warn("ingest:concepts", e));
          }
        }

        if (role === "user") {
          session.lastUserTurnId = turnId;
          session.lastUserText = text;
          session.userTurnCount++;
          session.resetTurn();

          // Predictive prefetch for follow-up queries
          if (worthEmbedding && session.currentConfig) {
            const predicted = predictQueries(text, (session.currentConfig.intent ?? "general") as import("./intent.js").IntentCategory);
            if (predicted.length > 0) {
              prefetchContext(predicted, session.sessionId, embeddings, store)
                .catch(e => swallow("ingest:prefetch", e));
            }
          }
        } else {
          session.lastAssistantText = text;
        }

        return { ingested: true };
      }
    } catch (e) {
      swallow.warn("ingest:store", e);
    }

    return { ingested: false };
  }

  async ingestBatch?(params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    isHeartbeat?: boolean;
  }): Promise<IngestBatchResult> {
    let count = 0;
    for (const message of params.messages) {
      const result = await this.ingest({ ...params, message });
      if (result.ingested) count++;
    }
    return { ingestedCount: count };
  }

  // ── Compact ────────────────────────────────────────────────────────────

  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
  }): Promise<CompactResult> {
    // Graph retrieval IS the compaction — ownsCompaction: true
    return {
      ok: true,
      compacted: false,
      reason: "Graph retrieval handles context selection; no LLM-based compaction needed.",
    };
  }

  // ── After turn ─────────────────────────────────────────────────────────

  async afterTurn?(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
  }): Promise<void> {
    const sessionKey = params.sessionKey ?? params.sessionId;
    const session = this.state.getSession(sessionKey);
    if (!session) return;

    const { store, embeddings } = this.state;

    // Deferred cleanup: run once on first turn when complete() is available
    if (session.userTurnCount <= 1 && typeof this.state.complete === "function") {
      runDeferredCleanup(store, embeddings, this.state.complete)
        .catch(e => swallow.warn("afterTurn:deferredCleanup", e));
    }

    // Ingest new messages from this turn (OpenClaw skips ingest() when afterTurn exists)
    const newMessages = params.messages.slice(params.prePromptMessageCount);
    for (const msg of newMessages) {
      await this.ingest({
        sessionId: params.sessionId,
        sessionKey: params.sessionKey,
        message: msg,
      }).catch(e => swallow.warn("afterTurn:ingest", e));
    }

    // Snapshot staged retrieval items before evaluateRetrieval clears them
    const stagedSnapshot = getStagedItems();

    // Evaluate retrieval quality — writes outcome records for ACAN training
    if (session.lastAssistantText) {
      const lastAssistantTurn = session.lastUserTurnId; // Turn ID for linking
      evaluateRetrieval(lastAssistantTurn, session.lastAssistantText, store)
        .catch(e => swallow.warn("afterTurn:evaluateRetrieval", e));
    }

    // Cognitive check: periodic reasoning over retrieved context
    if (shouldRunCheck(session.userTurnCount, session) && stagedSnapshot.length > 0) {
      const recentTurns = await store.getSessionTurns(session.sessionId, 6)
        .catch(() => [] as { role: string; text: string }[]);

      runCognitiveCheck({
        sessionId: session.sessionId,
        userQuery: session.lastUserText,
        responseText: session.lastAssistantText,
        retrievedNodes: stagedSnapshot.map(n => ({
          id: n.id,
          text: n.text ?? "",
          score: n.finalScore ?? 0,
          table: n.table,
        })),
        recentTurns,
      }, session, store, this.state.complete).catch(e => swallow.warn("afterTurn:cognitiveCheck", e));
    }

    // Flush to daemon when token threshold OR turn count threshold is reached
    const tokenReady = session.newContentTokens >= session.DAEMON_TOKEN_THRESHOLD;
    const turnReady = session.userTurnCount >= session.lastDaemonFlushTurnCount + 3;
    if (session.daemon && (tokenReady || turnReady)) {
      try {
        const recentTurns = await store.getSessionTurns(session.sessionId, 20);
        const turnData = recentTurns.map(t => ({
          role: t.role as "user" | "assistant",
          text: t.text,
          turnId: (t as any).id,
        }));

        // Gather retrieved memory IDs for dedup
        const retrievedMemories = stagedSnapshot.map(n => ({
          id: n.id,
          text: n.text ?? "",
        }));

        session.daemon.sendTurnBatch(
          turnData,
          [...session.pendingThinking],
          retrievedMemories,
        );

        session.newContentTokens = 0;
        session.lastDaemonFlushTurnCount = session.userTurnCount;
        session.pendingThinking.length = 0;
      } catch (e) {
        swallow.warn("afterTurn:daemonBatch", e);
      }
    }
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  async dispose(): Promise<void> {
    // Phase 3: combined extraction, graduation, soul graduation
    await this.state.shutdown();
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function extractMessageText(msg: AgentMessage): string {
  const m = msg as any;
  if (typeof m.content === "string") return m.content;
  if (Array.isArray(m.content)) {
    return m.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text ?? "")
      .join("\n");
  }
  return "";
}

/** Detect whether text has enough semantic content to warrant embedding. */
function hasSemantic(text: string): boolean {
  if (text.length < 15) return false;
  if (/^(ok|yes|no|sure|thanks|done|got it|hmm|hm|yep|nope|cool|nice|great)\s*[.!?]?\s*$/i.test(text)) {
    return false;
  }
  return text.split(/\s+/).filter(w => w.length > 2).length >= 3;
}

// --- Concept extraction (shared with llm-output hook) ---

const CONCEPT_RE = /\b(?:(?:use|using|implement|create|add|configure|setup|install|import)\s+)([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)?)/g;
const TECH_TERMS = /\b(api|database|schema|migration|endpoint|middleware|component|service|module|handler|controller|model|interface|type|class|function|method|hook|plugin|extension|config|cache|queue|worker|daemon)\b/gi;

async function extractAndLinkConcepts(
  turnId: string,
  text: string,
  state: GlobalPluginState,
): Promise<void> {
  const concepts = new Set<string>();

  let match: RegExpExecArray | null;
  const re1 = new RegExp(CONCEPT_RE.source, CONCEPT_RE.flags);
  while ((match = re1.exec(text)) !== null) {
    concepts.add(match[1].trim());
  }

  const re2 = new RegExp(TECH_TERMS.source, TECH_TERMS.flags);
  while ((match = re2.exec(text)) !== null) {
    concepts.add(match[1].toLowerCase());
  }

  if (concepts.size === 0) return;

  for (const conceptText of [...concepts].slice(0, 10)) {
    try {
      let embedding: number[] | null = null;
      if (state.embeddings.isAvailable()) {
        try { embedding = await state.embeddings.embed(conceptText); } catch { /* ok */ }
      }
      const conceptId = await state.store.upsertConcept(conceptText, embedding);
      if (conceptId) {
        await state.store.relate(turnId, "mentions", conceptId)
          .catch(e => swallow("concepts:relate", e));
      }
    } catch (e) {
      swallow("concepts:upsert", e);
    }
  }
}
