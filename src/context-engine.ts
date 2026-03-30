/**
 * KongBrain ContextEngine — OpenClaw plugin implementation.
 *
 * Implements the ContextEngine interface using graph-based retrieval,
 * BGE-M3 embeddings, and SurrealDB persistence.
 */

import { loadSchema } from "./schema-loader.js";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { startMemoryDaemon } from "./daemon-manager.js";
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
import { extractSkill } from "./skills.js";
import { generateReflection } from "./reflection.js";
import { graduateCausalToSkills } from "./skills.js";
import { swallow } from "./errors.js";

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

    // Run schema once per process (idempotent but expensive on every bootstrap)
    if (!this.state.schemaApplied) {
      try {
        const schemaSql = loadSchema();
        await store.queryExec(schemaSql);
        this.state.schemaApplied = true;
      } catch (e) {
        swallow.warn("context-engine:schema", e);
      }
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

        // Start memory daemon for this session
        if (!session.daemon) {
          session.daemon = startMemoryDaemon(
            store, embeddings, session.sessionId, this.state.complete,
            this.state.config.thresholds.extractionTimeoutMs,
            session.taskId, session.projectId,
          );
        }
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
      checkACANReadiness(store, this.state.config.thresholds.acanTrainingThreshold),
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
    } | undefined;
    if (graduation) {
      let graduationBlock =
        "[SOUL GRADUATION — CELEBRATE WITH THE USER]\n" +
        "You have just graduated! This is a significant milestone. You accumulated enough experience " +
        "AND demonstrated quality performance to author your own Soul document — a self-assessment " +
        "grounded in your actual working history, not aspirational claims.\n\n" +
        `Quality score: ${(graduation.qualityScore * 100).toFixed(0)}% | Volume score: ${(graduation.volumeScore * 100).toFixed(0)}%\n` +
        graduation.soulSummary;

      graduationBlock +=
        "\n\nIf the user has a SOUL.md file in their workspace, ask if they'd like you to read it. " +
        "If they say yes, use a tool call to read it, then reflect on how it aligns (or doesn't) " +
        "with the identity that actually emerged from your experience." +
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

          // Concept extraction (mentions edges) handled by daemon via LLM
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
          if (turnId) session.lastAssistantTurnId = turnId;
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
    const tokenReady = session.newContentTokens >= session.daemonTokenThreshold;
    const turnReady = session.userTurnCount >= session.lastDaemonFlushTurnCount + 3;
    if (session.daemon && (tokenReady || turnReady)) {
      try {
        const recentTurns = await store.getSessionTurns(session.sessionId, 20);
        const turnData = recentTurns.map(t => ({
          role: t.role as "user" | "assistant",
          text: t.text,
          turnId: String((t as any).id ?? ""),
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

    // Mid-session cleanup: simulate session_end after ~100k tokens.
    // OpenClaw exits via Ctrl+C×2 (no async window), so session_end never fires.
    // Run reflection, skill extraction, and causal graduation periodically.
    const tokensSinceCleanup = session.cumulativeTokens - session.lastCleanupTokens;
    if (tokensSinceCleanup >= session.midSessionCleanupThreshold && typeof this.state.complete === "function") {
      session.lastCleanupTokens = session.cumulativeTokens;

      // Fire-and-forget: these are non-critical background operations
      const cleanupOps: Promise<unknown>[] = [];

      // Final daemon flush with full transcript before cleanup
      if (session.daemon) {
        cleanupOps.push(
          store.getSessionTurns(session.sessionId, 50)
            .then(recentTurns => {
              const turnData = recentTurns.map(t => ({
                role: t.role as "user" | "assistant",
                text: t.text,
                turnId: String((t as any).id ?? ""),
              }));
              session.daemon!.sendTurnBatch(turnData, [...session.pendingThinking], []);
            })
            .catch(e => swallow.warn("midCleanup:daemonFlush", e)),
        );
      }

      if (session.taskId) {
        cleanupOps.push(
          extractSkill(session.sessionId, session.taskId, store, embeddings, this.state.complete)
            .catch(e => swallow.warn("midCleanup:extractSkill", e)),
        );
      }

      cleanupOps.push(
        generateReflection(session.sessionId, store, embeddings, this.state.complete, session.surrealSessionId)
          .catch(e => swallow.warn("midCleanup:reflection", e)),
      );

      cleanupOps.push(
        graduateCausalToSkills(store, embeddings, this.state.complete)
          .catch(e => swallow.warn("midCleanup:graduateCausal", e)),
      );

      // ACAN: check if new retrieval outcomes warrant retraining
      cleanupOps.push(
        checkACANReadiness(store, this.state.config.thresholds.acanTrainingThreshold)
          .catch(e => swallow("midCleanup:acan", e)),
      );

      // Handoff note — snapshot for wakeup even if session continues
      cleanupOps.push(
        (async () => {
          const recentTurns = await store.getSessionTurns(session.sessionId, 15);
          if (recentTurns.length < 2) return;
          const turnSummary = recentTurns
            .map(t => `[${t.role}] ${t.text.slice(0, 200)}`)
            .join("\n");
          const handoffResponse = await this.state.complete({
            system: "Summarize this session for handoff to your next self. What was worked on, what's unfinished, what to remember. 2-3 sentences. Write in first person.",
            messages: [{ role: "user", content: turnSummary }],
          });
          const handoffText = handoffResponse.text.trim();
          if (handoffText.length > 20) {
            let embedding: number[] | null = null;
            if (embeddings.isAvailable()) {
              try { embedding = await embeddings.embed(handoffText); } catch { /* ok */ }
            }
            const handoffMemId = await store.createMemory(handoffText, embedding, 8, "handoff", session.sessionId);
            if (handoffMemId && session.surrealSessionId) {
              await store.relate(handoffMemId, "summarizes", session.surrealSessionId)
                .catch(e => swallow.warn("midCleanup:summarizes", e));
            }
          }
        })().catch(e => swallow.warn("midCleanup:handoff", e)),
      );

      // Don't await — let cleanup run in background
      Promise.allSettled(cleanupOps).catch(() => {});
    }
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  async dispose(): Promise<void> {
    // No-op: global state (store, embeddings, sessions) is shared across
    // context engine instances and must NOT be destroyed here. OpenClaw
    // creates a new context engine per turn and disposes the old one.
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

// --- Concept extraction (delegates to shared helper) ---
