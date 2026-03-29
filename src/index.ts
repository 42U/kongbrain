/**
 * KongBrain — OpenClaw context-engine plugin entry point.
 *
 * Replaces the default context engine with graph-based retrieval using
 * SurrealDB persistence and BGE-M3 embeddings.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { parsePluginConfig } from "./config.js";
import { SurrealStore } from "./surreal.js";
import { EmbeddingService } from "./embeddings.js";
import { GlobalPluginState, type CompleteFn } from "./state.js";
import { KongBrainContextEngine } from "./context-engine.js";
import { createRecallToolDef } from "./tools/recall.js";
import { createCoreMemoryToolDef } from "./tools/core-memory.js";
import { createIntrospectToolDef } from "./tools/introspect.js";
import { createBeforePromptBuildHandler } from "./hooks/before-prompt-build.js";
import { createBeforeToolCallHandler } from "./hooks/before-tool-call.js";
import { createAfterToolCallHandler } from "./hooks/after-tool-call.js";
import { createLlmOutputHandler } from "./hooks/llm-output.js";
import { startMemoryDaemon } from "./daemon-manager.js";
import { seedIdentity } from "./identity.js";
import { synthesizeWakeup, synthesizeStartupCognition } from "./wakeup.js";
import { extractSkill } from "./skills.js";
import { generateReflection, setReflectionContextWindow } from "./reflection.js";
import { graduateCausalToSkills } from "./skills.js";
import { attemptGraduation, evolveSoul, checkStageTransition } from "./soul.js";
import { hasMigratableFiles, migrateWorkspace } from "./workspace-migrate.js";
import { writeHandoffFileSync } from "./handoff-file.js";
import { runDeferredCleanup } from "./deferred-cleanup.js";
import { swallow } from "./errors.js";

let globalState: GlobalPluginState | null = null;
let shutdownPromise: Promise<void> | null = null;
let registeredExitHandler: (() => void) | null = null;
let registeredSyncExitHandler: (() => void) | null = null;
let registered = false;

/**
 * Run the critical session-end extraction for all active sessions.
 * Called from both session_end hook and process exit handler.
 */
async function runSessionCleanup(
  session: import("./state.js").SessionState,
  state: GlobalPluginState,
): Promise<void> {
  const { store: s, embeddings: emb } = state;
  const endOps: Promise<unknown>[] = [];

  // Final daemon flush — send full session for extraction
  if (session.daemon) {
    endOps.push(
      (async () => {
        try {
          const recentTurns = await s.getSessionTurns(session.sessionId, 50);
          const turnData = recentTurns.map(t => ({
            role: t.role as "user" | "assistant",
            text: t.text,
            turnId: (t as any).id,
          }));
          session.daemon!.sendTurnBatch(turnData, [...session.pendingThinking], []);
        } catch (e) { swallow.warn("cleanup:finalDaemonFlush", e); }
        await session.daemon!.shutdown(45_000).catch(e => swallow.warn("cleanup:daemonShutdown", e));
        session.daemon = null;
      })(),
    );
  }

  const { complete } = state;

  // Skill extraction
  if (session.taskId) {
    endOps.push(
      extractSkill(session.sessionId, session.taskId, s, emb, complete)
        .catch(e => swallow.warn("cleanup:extractSkill", e)),
    );
  }

  // Metacognitive reflection
  endOps.push(
    generateReflection(session.sessionId, s, emb, complete)
      .catch(e => swallow.warn("cleanup:reflection", e)),
  );

  // Graduate causal chains -> skills
  endOps.push(
    graduateCausalToSkills(s, emb, complete)
      .catch(e => swallow.warn("cleanup:graduateCausal", e)),
  );

  // Soul graduation attempt — capture result for user notification
  const graduationPromise = attemptGraduation(s, complete, state.workspaceDir)
    .catch(e => { swallow.warn("cleanup:soulGraduation", e); return null; });
  endOps.push(graduationPromise);

  // The session-end LLM call is critical and needs the full 45s.
  await Promise.race([
    Promise.allSettled(endOps),
    new Promise(resolve => setTimeout(resolve, 45_000)),
  ]);

  // If soul graduation just happened, persist a graduation event so the next
  // session can celebrate with the user. We also fire a system event for
  // immediate visibility if the session is still active.
  try {
    const gradResult = await graduationPromise;
    if (gradResult?.graduated && gradResult.soul) {
      // Check if this is a NEW graduation (not a pre-existing soul)
      const isNewGraduation = gradResult.report.stage === "ready";
      if (isNewGraduation) {
        // Persist graduation event for next session pickup
        await s.queryExec(
          `CREATE graduation_event CONTENT $data`,
          {
            data: {
              session_id: session.sessionId,
              acknowledged: false,
              quality_score: gradResult.report.qualityScore,
              volume_score: gradResult.report.volumeScore,
              stage: gradResult.report.stage,
              created_at: new Date().toISOString(),
            },
          },
        ).catch(e => swallow.warn("cleanup:graduationEvent", e));

        // Fire system event for immediate user notification
        if (state.enqueueSystemEvent) {
          state.enqueueSystemEvent(
            "[GRADUATION] KongBrain has achieved soul graduation! " +
            "The agent has accumulated enough experience and demonstrated sufficient quality " +
            "to author its own identity document. It will share this milestone at the start of the next session.",
            { sessionKey: session.sessionKey },
          );
        }
      }
    }
  } catch (e) {
    swallow.warn("cleanup:graduationNotify", e);
  }

  // Soul evolution — if soul already exists, check if it should be revised
  // based on new experience (runs every 10 sessions after last revision)
  try {
    const gradResult = await graduationPromise;
    if (gradResult?.graduated && gradResult.report.stage !== "ready") {
      // Pre-existing soul — check for evolution
      await evolveSoul(s, complete);
    }
  } catch (e) {
    swallow.warn("cleanup:soulEvolution", e);
  }

  // Stage transition tracking — record progress and notify on level-ups
  try {
    const transition = await checkStageTransition(s);
    if (transition.transitioned && state.enqueueSystemEvent) {
      const stageLabels: Record<string, string> = {
        nascent: "Nascent (0-3/7)",
        developing: "Developing (4/7)",
        emerging: "Emerging (5/7)",
        maturing: "Maturing (6/7)",
        ready: "Ready (7/7 + quality gate)",
      };
      const prev = stageLabels[transition.previousStage ?? "nascent"] ?? transition.previousStage;
      const curr = stageLabels[transition.currentStage] ?? transition.currentStage;
      state.enqueueSystemEvent(
        `[MATURITY] Stage transition: ${prev} → ${curr}. ` +
        `Volume: ${transition.report.met.length}/7 | Quality: ${transition.report.qualityScore.toFixed(2)}`,
        { sessionKey: session.sessionKey },
      );
    }
  } catch (e) {
    swallow.warn("cleanup:stageTransition", e);
  }

  // Generate handoff note for next session wakeup
  try {
    const recentTurns = await s.getSessionTurns(session.sessionId, 15)
      .catch(() => [] as { role: string; text: string }[]);
    if (recentTurns.length >= 2) {
      const turnSummary = recentTurns
        .map(t => `[${t.role}] ${t.text.slice(0, 200)}`)
        .join("\n");

      const handoffResponse = await complete({
        system: "Summarize this session for handoff to your next self. What was worked on, what's unfinished, what to remember. 2-3 sentences. Write in first person.",
        messages: [{ role: "user", content: turnSummary }],
      });

      const handoffText = handoffResponse.text.trim();
      if (handoffText.length > 20) {
        let embedding: number[] | null = null;
        if (emb.isAvailable()) {
          try { embedding = await emb.embed(handoffText); } catch { /* ok */ }
        }
        await s.createMemory(handoffText, embedding, 8, "handoff", session.sessionId);
      }
    }
  } catch (e) {
    swallow.warn("cleanup:handoff", e);
  }
}

/**
 * Check if the agent just graduated in a recent session and hasn't told the user yet.
 * Sets a flag on the session so the context engine can inject graduation context.
 */
async function detectGraduationEvent(
  store: SurrealStore,
  session: import("./state.js").SessionState,
  state: GlobalPluginState,
): Promise<void> {
  if (!store.isAvailable()) return;

  // Check for unacknowledged graduation events
  const events = await store.queryFirst<{
    id: string;
    quality_score: number;
    volume_score: number;
  }>(
    `SELECT id, quality_score, volume_score FROM graduation_event
     WHERE acknowledged = false
     ORDER BY created_at DESC LIMIT 1`,
  ).catch(() => []);

  if (events.length === 0) return;

  const event = events[0];

  // Mark as acknowledged so we don't repeat
  await store.queryExec(
    `UPDATE $id SET acknowledged = true, acknowledged_at = time::now(), acknowledged_session = $sid`,
    { id: event.id, sid: session.sessionId },
  ).catch(e => swallow.warn("graduationDetect:ack", e));

  // Get the soul document for the agent to reference
  const soulRows = await store.queryFirst<{
    working_style: string[];
    self_observations: string[];
    earned_values: { value: string; grounded_in: string }[];
  }>(`SELECT working_style, self_observations, earned_values FROM soul:kongbrain`).catch(() => []);
  const soul = soulRows[0];

  // Build a summary the agent can use to talk about itself
  let soulSummary = "";
  if (soul) {
    const styles = (soul.working_style ?? []).slice(0, 3).map(s => `- ${s}`).join("\n");
    const obs = (soul.self_observations ?? []).slice(0, 3).map(s => `- ${s}`).join("\n");
    const vals = (soul.earned_values ?? []).slice(0, 3).map(v => `- ${v.value}: ${v.grounded_in}`).join("\n");
    soulSummary = `\n\nYour soul document contains:\nWorking style:\n${styles}\n\nSelf-observations:\n${obs}\n\nEarned values:\n${vals}`;
  }

  // Read SOUL.md — the user's original nudge about who they wanted the agent to be.
  // The agent should reflect on how this aligns (or doesn't) with what actually emerged.
  let userNudge = "";
  if (state.workspaceDir) {
    try {
      const soulMd = await readFile(join(state.workspaceDir, "SOUL.md"), "utf-8");
      if (soulMd.trim().length > 50) {
        userNudge = soulMd.trim().slice(0, 2000);
      }
    } catch {
      // No SOUL.md — that's fine, graduation happened without user guidance
    }
  }

  // Flag the session for context engine injection
  (session as any)._graduationCelebration = {
    qualityScore: event.quality_score,
    volumeScore: event.volume_score,
    soulSummary,
    userNudge,
  };

  // Also fire a system event so the user sees it immediately
  if (state.enqueueSystemEvent) {
    state.enqueueSystemEvent(
      "[MILESTONE] This is a special session — KongBrain has graduated and authored its own Soul document. " +
      "The agent will share what this means.",
      { sessionKey: session.sessionKey },
    );
  }
}

export default definePluginEntry({
  id: "kongbrain",
  name: "KongBrain",
  description: "Graph-backed cognitive context engine with SurrealDB persistence and BGE-M3 embeddings.",
  kind: "context-engine",

  register(api) {
    const config = parsePluginConfig(api.pluginConfig as Record<string, unknown> | undefined);
    const logger = api.logger;

    // Initialize shared resources — reuse existing globalState if register() is called
    // multiple times (OpenClaw may invoke the factory more than once). Hooks from the
    // first register() hold a closure over globalState, so replacing it would orphan them.
    if (!globalState) {
      const store = new SurrealStore(config.surreal);
      const embeddings = new EmbeddingService(config.embedding);
      // Build a CompleteFn using pi-ai directly since api.runtime.complete
      // is not available in OpenClaw 2026.3.24 (unreleased feature).
      const apiRef = api;
      const complete: CompleteFn = async (params) => {
        // Try runtime.complete first (future-proof for when it ships)
        if (typeof apiRef.runtime?.complete === "function") {
          return apiRef.runtime.complete(params);
        }
        // Fall back to calling pi-ai directly (runtime.complete not in OpenClaw 2026.3.24)
        const piAi = await import("@mariozechner/pi-ai");
        const provider = params.provider ?? apiRef.runtime.agent.defaults.provider;
        const modelId = params.model ?? apiRef.runtime.agent.defaults.model;
        const model = piAi.getModel(provider as any, modelId as any);
        if (!model) {
          throw new Error(`Model "${modelId}" not found for provider "${provider}"`);
        }
        // Resolve auth via OpenClaw's runtime (handles profiles, env vars, etc.)
        const cfg = apiRef.runtime.config.loadConfig();
        const auth = await apiRef.runtime.modelAuth.getApiKeyForModel({ model, cfg });
        // Build context
        const now = Date.now();
        const messages: any[] = params.messages.map(m =>
          m.role === "user"
            ? { role: "user", content: m.content, timestamp: now }
            : { role: "assistant", content: [{ type: "text", text: m.content }],
                api: model.api, provider: model.provider, model: model.id,
                usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                stopReason: "stop", timestamp: now }
        );
        const context = { systemPrompt: params.system, messages };
        // Pass apiKey directly in options so the provider can use it
        const response = await piAi.completeSimple(model, context, {
          apiKey: auth.apiKey,
        } as any);
        let text = "";
        let thinking: string | undefined;
        for (const block of response.content) {
          if (block.type === "text") text += block.text;
          else if ((block as any).type === "thinking") thinking = (thinking ?? "") + (block as any).thinking;
        }
        return { text, thinking, usage: { input: response.usage.input, output: response.usage.output } };
      };
      globalState = new GlobalPluginState(config, store, embeddings, complete);
    }
    globalState.workspaceDir = api.resolvePath(".");
    globalState.enqueueSystemEvent = (text, opts) =>
      api.runtime.system.enqueueSystemEvent(text, opts);

    const state = globalState;

    // Register the context engine factory
    api.registerContextEngine("kongbrain", async () => {
      const { store, embeddings } = state;

      // Connect to SurrealDB
      try {
        await store.initialize();
        logger.info(`SurrealDB connected: ${config.surreal.url}`);
      } catch (e) {
        logger.error(`SurrealDB connection failed: ${e}`);
        throw e;
      }

      // Initialize BGE-M3 embeddings
      try {
        await embeddings.initialize();
        logger.info(`BGE-M3 embeddings initialized: ${config.embedding.modelPath}`);
      } catch (e) {
        logger.warn(`Embeddings init failed — running in degraded mode: ${e}`);
      }

      // Seed identity chunks (idempotent, requires embeddings ready)
      seedIdentity(store, embeddings)
        .then(n => { if (n > 0) logger.info(`Seeded ${n} identity chunks`); })
        .catch(e => swallow.warn("factory:seedIdentity", e));

      return new KongBrainContextEngine(state);
    });

    // ── Hook handlers (register once — register() may be called multiple times) ──

    if (!registered) {
      api.on("before_prompt_build", createBeforePromptBuildHandler(globalState));
      api.on("before_tool_call", createBeforeToolCallHandler(globalState));
      api.on("after_tool_call", createAfterToolCallHandler(globalState));
      api.on("llm_output", createLlmOutputHandler(globalState));
    }

    // ── Session lifecycle (also register once) ─────────────────────────

    if (!registered) api.on("session_start", async (event) => {
      if (!globalState) return;
      const sessionKey = event.sessionKey ?? event.sessionId;
      const session = globalState.getOrCreateSession(sessionKey, event.sessionId);

      // Register tools
      try {
        api.registerTool(
          createRecallToolDef(globalState, session),
          { name: "recall" },
        );
        api.registerTool(
          createCoreMemoryToolDef(globalState, session),
          { name: "core_memory" },
        );
        api.registerTool(
          createIntrospectToolDef(globalState, session),
          { name: "introspect" },
        );
      } catch (e) {
        swallow.warn("index:registerTools", e);
      }

      // Start memory daemon worker thread
      try {
        session.daemon = startMemoryDaemon(
          config.surreal,
          config.embedding,
          session.sessionId,
          { provider: api.runtime.agent.defaults.provider, model: api.runtime.agent.defaults.model },
        );
      } catch (e) {
        swallow.warn("index:startDaemon", e);
      }

      // Check for workspace .md files from the default context engine
      if (globalState!.workspaceDir) {
        hasMigratableFiles(globalState!.workspaceDir)
          .then(hasMigratable => {
            if (hasMigratable) {
              (session as any)._hasMigratableFiles = true;
            }
          })
          .catch(e => swallow("index:migrationCheck", e));
      }

      // Set reflection context window from config
      setReflectionContextWindow(200000);

      // Check for recent graduation event (from a previous session)
      detectGraduationEvent(store, session, globalState!)
        .catch(e => swallow("index:graduationDetect", e));

      // Synthesize wakeup briefing (background, non-blocking)
      // The briefing is stored and later injected via assemble()'s systemPromptAddition
      synthesizeWakeup(store, globalState!.complete, session.sessionId, globalState!.workspaceDir)
        .then(briefing => {
          if (briefing) (session as any)._wakeupBriefing = briefing;
        })
        .catch(e => swallow.warn("index:wakeup", e));

      // Startup cognition (background)
      synthesizeStartupCognition(store, globalState!.complete)
        .then(cognition => {
          if (cognition) (session as any)._startupCognition = cognition;
        })
        .catch(e => swallow.warn("index:startupCognition", e));

      // Deferred cleanup: extract knowledge from orphaned sessions (background)
      runDeferredCleanup(store, embeddings, globalState!.complete)
        .then(n => { if (n > 0) logger.info(`Deferred cleanup: processed ${n} orphaned session(s)`); })
        .catch(e => swallow.warn("index:deferredCleanup", e));
    });

    if (!registered) api.on("session_end", async (event) => {
      if (!globalState) return;
      const sessionKey = event.sessionKey ?? event.sessionId;
      const session = globalState.getSession(sessionKey);
      if (!session) return;

      shutdownPromise = runSessionCleanup(session, globalState);
      await shutdownPromise;
      shutdownPromise = null;

      session.cleanedUp = true;
      if (session.surrealSessionId) {
        await store.markSessionEnded(session.surrealSessionId)
          .catch(e => swallow.warn("session_end:markEnded", e));
      }

      globalState.removeSession(sessionKey);
    });

    // -- Exit handlers --
    // OpenClaw TUI calls process.exit(0) on Ctrl+C×2 with no async window.
    // We use two layers:
    //   1. process.on("exit") — SYNC: writes handoff file to disk
    //   2. SIGTERM — async cleanup for non-TUI modes (gateway, daemon)
    // We do NOT register SIGINT — TUI owns that signal and always wins the race.

    // Clean up previous listeners (register() can be called multiple times)
    if (registeredExitHandler) {
      process.removeListener("SIGTERM", registeredExitHandler);
    }
    if (registeredSyncExitHandler) {
      process.removeListener("exit", registeredSyncExitHandler);
    }

    // Sync exit handler: writes handoff file for all uncleaned sessions
    const syncExitHandler = () => {
      if (!globalState?.workspaceDir) return;
      const sessions = [...(globalState as any).sessions.values()] as import("./state.js").SessionState[];
      for (const session of sessions) {
        if (session.cleanedUp) continue;
        writeHandoffFileSync({
          sessionId: session.sessionId,
          timestamp: new Date().toISOString(),
          userTurnCount: session.userTurnCount,
          lastUserText: session.lastUserText.slice(0, 500),
          lastAssistantText: session.lastAssistantText.slice(0, 500),
          unextractedTokens: session.newContentTokens,
        }, globalState!.workspaceDir!);
      }
    };

    // Async exit handler: full cleanup for SIGTERM (gateway/daemon mode)
    const asyncExitHandler = () => {
      if (!globalState) return;
      const sessions = [...(globalState as any).sessions.values()] as import("./state.js").SessionState[];
      if (sessions.length === 0 && !shutdownPromise) return;

      const cleanups = sessions.map(s => runSessionCleanup(s, globalState!));
      if (shutdownPromise) cleanups.push(shutdownPromise);

      const done = Promise.allSettled(cleanups).then(() => {
        globalState?.shutdown().catch(() => {});
      });

      done.then(() => process.exit(0)).catch(() => process.exit(1));
    };

    registeredSyncExitHandler = syncExitHandler;
    registeredExitHandler = asyncExitHandler;
    process.on("exit", syncExitHandler);
    process.once("SIGTERM", asyncExitHandler);

    if (!registered) {
      registered = true;
    }
  },
});
