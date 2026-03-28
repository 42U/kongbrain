/**
 * KongBrain — OpenClaw context-engine plugin entry point.
 *
 * Replaces the default context engine with graph-based retrieval using
 * SurrealDB persistence and BGE-M3 embeddings.
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { parsePluginConfig } from "./config.js";
import { SurrealStore } from "./surreal.js";
import { EmbeddingService } from "./embeddings.js";
import { GlobalPluginState } from "./state.js";
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
import { attemptGraduation } from "./soul.js";
import { hasMigratableFiles, migrateWorkspace } from "./workspace-migrate.js";
import { swallow } from "./errors.js";

let globalState: GlobalPluginState | null = null;
let shutdownPromise: Promise<void> | null = null;

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

  // Soul graduation attempt
  endOps.push(
    attemptGraduation(s, complete, state.workspaceDir)
      .catch(e => swallow.warn("cleanup:soulGraduation", e)),
  );

  // The session-end Opus call is critical and needs the full 45s.
  await Promise.race([
    Promise.allSettled(endOps),
    new Promise(resolve => setTimeout(resolve, 45_000)),
  ]);
}

export default definePluginEntry({
  id: "kongbrain",
  name: "KongBrain",
  description: "Graph-backed cognitive context engine with SurrealDB persistence and BGE-M3 embeddings.",
  kind: "context-engine",

  register(api) {
    const config = parsePluginConfig(api.pluginConfig as Record<string, unknown> | undefined);
    const logger = api.logger;

    // Initialize shared resources
    const store = new SurrealStore(config.surreal);
    const embeddings = new EmbeddingService(config.embedding);
    globalState = new GlobalPluginState(config, store, embeddings, api.runtime.complete);
    globalState.workspaceDir = api.resolvePath(".");

    // Register the context engine factory
    api.registerContextEngine("kongbrain", async () => {
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

      return new KongBrainContextEngine(globalState!);
    });

    // ── Hook handlers ──────────────────────────────────────────────────

    api.on("before_prompt_build", createBeforePromptBuildHandler(globalState));
    api.on("before_tool_call", createBeforeToolCallHandler(globalState));
    api.on("after_tool_call", createAfterToolCallHandler(globalState));
    api.on("llm_output", createLlmOutputHandler(globalState));

    // ── Session lifecycle ──────────────────────────────────────────────

    api.on("session_start", async (event) => {
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

      // Seed identity chunks (idempotent — skips if already seeded)
      seedIdentity(store, embeddings)
        .catch(e => swallow.warn("index:seedIdentity", e));

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

      // Synthesize wakeup briefing (background, non-blocking)
      // The briefing is stored and later injected via assemble()'s systemPromptAddition
      synthesizeWakeup(store, globalState!.complete, session.sessionId)
        .then(briefing => {
          if (briefing) {
            // Store for later injection via context engine
            (session as any)._wakeupBriefing = briefing;
          }
        })
        .catch(e => swallow.warn("index:wakeup", e));

      // Startup cognition (background)
      synthesizeStartupCognition(store, globalState!.complete)
        .then(cognition => {
          if (cognition) {
            (session as any)._startupCognition = cognition;
          }
        })
        .catch(e => swallow.warn("index:startupCognition", e));
    });

    api.on("session_end", async (event) => {
      if (!globalState) return;
      const sessionKey = event.sessionKey ?? event.sessionId;
      const session = globalState.getSession(sessionKey);
      if (!session) return;

      shutdownPromise = runSessionCleanup(session, globalState);
      await shutdownPromise;
      shutdownPromise = null;

      globalState.removeSession(sessionKey);
    });

    // OpenClaw's session_end is fire-and-forget and doesn't fire on CLI exit.
    // Register a process exit handler to ensure the critical Opus extraction
    // completes even when the user exits with Ctrl+D or /exit.
    const onProcessExit = () => {
      if (!globalState) return;
      // If session_end already ran, shutdownPromise is null — nothing to do.
      // Otherwise, run cleanup for all active sessions.
      const sessions = [...(globalState as any).sessions.values()] as import("./state.js").SessionState[];
      if (sessions.length === 0 && !shutdownPromise) return;

      // Keep the process alive until cleanup finishes.
      // beforeExit fires when the event loop drains — returning a promise
      // re-queues work and prevents immediate exit.
      const cleanups = sessions.map(s => runSessionCleanup(s, globalState!));
      if (shutdownPromise) cleanups.push(shutdownPromise);

      const done = Promise.allSettled(cleanups).then(() => {
        globalState?.shutdown().catch(() => {});
      });

      // Block exit until done
      done.then(() => process.exit(0)).catch(() => process.exit(1));
    };

    process.once("beforeExit", onProcessExit);
    process.once("SIGINT", onProcessExit);
    process.once("SIGTERM", onProcessExit);

    logger.info("KongBrain plugin registered");
  },
});
