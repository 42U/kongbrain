import type { KongBrainConfig } from "./config.js";
import type { SurrealStore } from "./surreal.js";
import type { EmbeddingService } from "./embeddings.js";
import type { AdaptiveConfig } from "./orchestrator.js";
import type { MemoryDaemon } from "./daemon-manager.js";

/** Parameters for an LLM completion call. */
export type CompleteParams = {
  system?: string;
  messages: { role: "user" | "assistant"; content: string }[];
  provider?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  reasoning?: "none" | "low" | "medium" | "high";
};

/** Result of an LLM completion call. */
export type CompleteResult = {
  text: string;
  thinking?: string;
  usage?: { input: number; output: number };
  provider?: string;
  model?: string;
  stopReason?: string;
};

/** Provider-agnostic LLM completion function. */
export type CompleteFn = (params: CompleteParams) => Promise<CompleteResult>;

// --- Per-session mutable state ---

const DEFAULT_TOOL_LIMIT = 10;

export class SessionState {
  readonly sessionId: string;
  readonly sessionKey: string;

  // Turn tracking
  lastUserTurnId = "";
  lastAssistantTurnId = "";
  lastUserText = "";
  lastAssistantText = "";
  toolCallCount = 0;
  toolLimit = DEFAULT_TOOL_LIMIT;
  turnTextLength = 0;
  toolCallsSinceLastText = 0;
  softInterrupted = false;
  turnStartMs = Date.now();
  userTurnCount = 0;

  // Thinking capture
  readonly pendingThinking: string[] = [];

  // Memory daemon
  daemon: MemoryDaemon | null = null;
  newContentTokens = 0;
  daemonTokenThreshold = 4000;
  lastDaemonFlushTurnCount = 0;

  // Cumulative session token tracking (for mid-session cleanup trigger)
  cumulativeTokens = 0;
  lastCleanupTokens = 0;
  midSessionCleanupThreshold = 25_000;
  /** Last cumulative usage total seen from OpenClaw — used to compute per-call deltas
   *  since getUsageTotals() returns running totals, not per-response values. */
  lastSeenUsageTotal = 0;

  // Cleanup tracking
  cleanedUp = false;

  // Current adaptive config (set by orchestrator preflight each turn)
  currentConfig: AdaptiveConfig | null = null;

  // Pending tool args for artifact tracking
  readonly pendingToolArgs = new Map<string, unknown>();

  // Tool call optimization state (claw-code patterns)
  /** Query vector from this turn's context retrieval — used to detect redundant recall calls. */
  lastQueryVec: number[] | null = null;
  /** Summary of what graphTransformContext injected — shown in planning gate. */
  lastRetrievalSummary = "";
  /** API request cycle counter — hard cap prevents runaway token spend. */
  apiCycleCount = 0;
  /** Tracks which static context sections the model has already seen in the conversation window.
   *  Persists across turns (NOT cleared in resetTurn) — cleared only when messages drop from window. */
  readonly injectedSections = new Set<string>();

  // 5-pillar IDs (populated at bootstrap)
  agentId = "";
  projectId = "";
  taskId = "";
  surrealSessionId = "";

  constructor(sessionId: string, sessionKey: string) {
    this.sessionId = sessionId;
    this.sessionKey = sessionKey;
  }

  /** Reset per-turn counters at the start of each prompt. */
  resetTurn(): void {
    this.toolCallCount = 0;
    this.toolLimit = DEFAULT_TOOL_LIMIT;
    this.turnTextLength = 0;
    this.toolCallsSinceLastText = 0;
    this.softInterrupted = false;
    this.turnStartMs = Date.now();
    this.pendingThinking.length = 0;
    this.lastRetrievalSummary = "";
    this.apiCycleCount = 0;
    // NOTE: lastQueryVec and injectedSections are NOT cleared here —
    // they persist across turns within the session.
  }
}

// --- Global plugin state (shared across all sessions) ---

/** Function to enqueue a system event visible to the user. */
export type EnqueueSystemEventFn = (text: string, options: { sessionKey: string }) => boolean;

export class GlobalPluginState {
  readonly config: KongBrainConfig;
  readonly store: SurrealStore;
  readonly embeddings: EmbeddingService;
  complete: CompleteFn;
  workspaceDir?: string;
  enqueueSystemEvent?: EnqueueSystemEventFn;
  schemaApplied = false;
  private sessions = new Map<string, SessionState>();

  constructor(
    config: KongBrainConfig,
    store: SurrealStore,
    embeddings: EmbeddingService,
    complete: CompleteFn,
  ) {
    this.config = config;
    this.store = store;
    this.embeddings = embeddings;
    this.complete = complete;
  }

  /** Get or create a SessionState for the given session key. */
  getOrCreateSession(sessionKey: string, sessionId: string): SessionState {
    let session = this.sessions.get(sessionKey);
    if (!session) {
      session = new SessionState(sessionId, sessionKey);
      session.daemonTokenThreshold = this.config.thresholds.daemonTokenThreshold;
      session.midSessionCleanupThreshold = this.config.thresholds.midSessionCleanupThreshold;
      this.sessions.set(sessionKey, session);
    }
    return session;
  }

  /** Get an existing session by key. */
  getSession(sessionKey: string): SessionState | undefined {
    return this.sessions.get(sessionKey);
  }

  /** Remove a session from the map (after dispose/cleanup). */
  removeSession(sessionKey: string): void {
    this.sessions.delete(sessionKey);
  }

  /** Shut down all shared resources. */
  async shutdown(): Promise<void> {
    this.sessions.clear();
    await this.embeddings.dispose();
    await this.store.dispose();
  }
}
