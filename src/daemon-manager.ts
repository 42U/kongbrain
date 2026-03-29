/**
 * Daemon Manager — runs memory extraction in-process.
 *
 * Originally used a Worker thread, but OpenClaw loads plugins via jiti
 * (TypeScript only, no compiled JS), and Node's Worker constructor requires
 * .js files. Refactored to run extraction async in the main thread.
 * The extraction is I/O-bound (LLM calls + DB writes), not CPU-bound,
 * so in-process execution is fine.
 */
import type { SurrealConfig, EmbeddingConfig } from "./config.js";
import type { TurnData, PriorExtractions } from "./daemon-types.js";
import { SurrealStore } from "./surreal.js";
import { EmbeddingService } from "./embeddings.js";
import { swallow } from "./errors.js";

export type { TurnData } from "./daemon-types.js";

export interface MemoryDaemon {
  /** Fire-and-forget: send a batch of turns for incremental extraction. */
  sendTurnBatch(
    turns: TurnData[],
    thinking: string[],
    retrievedMemories: { id: string; text: string }[],
    priorExtractions?: PriorExtractions,
  ): void;
  /** Request current daemon status. */
  getStatus(): Promise<{ type: "status"; extractedTurns: number; pendingBatches: number; errors: number }>;
  /** Graceful shutdown: waits for current extraction, then cleans up. */
  shutdown(timeoutMs?: number): Promise<void>;
  /** How many turns has the daemon already extracted? */
  getExtractedTurnCount(): number;
}

export function startMemoryDaemon(
  surrealConfig: SurrealConfig,
  embeddingConfig: EmbeddingConfig,
  sessionId: string,
  llmConfig?: { provider?: string; model?: string },
): MemoryDaemon {
  // Daemon-local DB and embedding instances (separate connections)
  let store: SurrealStore | null = null;
  let embeddings: EmbeddingService | null = null;
  let initialized = false;
  let initFailed = false;
  let processing = false;
  let shuttingDown = false;
  let extractedTurnCount = 0;
  let errorCount = 0;

  const priorState: PriorExtractions = {
    conceptNames: [], artifactPaths: [], skillNames: [],
  };

  // Lazy init — connect on first batch, not at startup
  async function ensureInit(): Promise<boolean> {
    if (initialized) return true;
    if (initFailed) return false;
    try {
      store = new SurrealStore(surrealConfig);
      await store.initialize();
      embeddings = new EmbeddingService(embeddingConfig);
      await embeddings.initialize();
      initialized = true;
      return true;
    } catch (e) {
      swallow.warn("daemon:init", e);
      initFailed = true;
      return false;
    }
  }

  // Import extraction logic lazily to avoid circular deps
  async function runExtraction(
    turns: TurnData[],
    thinking: string[],
    retrievedMemories: { id: string; text: string }[],
    incomingPrior?: PriorExtractions,
  ): Promise<void> {
    if (!store || !embeddings) return;
    if (turns.length < 2) return;

    const provider = llmConfig?.provider;
    const modelId = llmConfig?.model;
    if (!provider || !modelId) {
      swallow.warn("daemon:extraction", new Error("Missing llmProvider/llmModel"));
      return;
    }

    // Merge incoming prior state
    if (incomingPrior) {
      for (const name of incomingPrior.conceptNames) {
        if (!priorState.conceptNames.includes(name)) priorState.conceptNames.push(name);
      }
      for (const path of incomingPrior.artifactPaths) {
        if (!priorState.artifactPaths.includes(path)) priorState.artifactPaths.push(path);
      }
      for (const name of incomingPrior.skillNames) {
        if (!priorState.skillNames.includes(name)) priorState.skillNames.push(name);
      }
    }

    // Dynamically import the extraction helpers from memory-daemon
    const { buildSystemPrompt, buildTranscript, writeExtractionResults } = await import("./memory-daemon.js");

    const transcript = buildTranscript(turns);
    const sections: string[] = [`[TRANSCRIPT]\n${transcript.slice(0, 60000)}`];

    if (thinking.length > 0) {
      sections.push(`[THINKING]\n${thinking.slice(-8).join("\n---\n").slice(0, 4000)}`);
    }

    if (retrievedMemories.length > 0) {
      const memList = retrievedMemories.map(m => `${m.id}: ${String(m.text).slice(0, 200)}`).join("\n");
      sections.push(`[RETRIEVED MEMORIES]\nMark any that have been fully addressed/fixed/completed.\n${memList}`);
    }

    const systemPrompt = buildSystemPrompt(thinking.length > 0, retrievedMemories.length > 0, priorState);

    const { completeSimple, getModel } = await import("@mariozechner/pi-ai");
    const model = (getModel as any)(provider, modelId);

    const response = await completeSimple(model, {
      systemPrompt,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: sections.join("\n\n"),
      }],
    });

    const responseText = response.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("");

    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return;

    let result: Record<string, any>;
    try {
      result = JSON.parse(jsonMatch[0]);
    } catch {
      try {
        result = JSON.parse(jsonMatch[0].replace(/,\s*([}\]])/g, "$1"));
      } catch {
        result = {};
        const fields = ["causal", "monologue", "resolved", "concepts", "corrections", "preferences", "artifacts", "decisions", "skills"];
        for (const field of fields) {
          const fieldMatch = jsonMatch[0].match(new RegExp(`"${field}"\\s*:\\s*(\\[[\\s\\S]*?\\])(?=\\s*[,}]\\s*"[a-z]|\\s*\\}$)`, "m"));
          if (fieldMatch) {
            try { result[field] = JSON.parse(fieldMatch[1]); } catch { /* skip */ }
          }
        }
        if (Object.keys(result).length === 0) return;
      }
    }

    const counts = await writeExtractionResults(result, sessionId, store, embeddings, priorState);
    extractedTurnCount = turns.length;
  }

  // Pending batch (only keep latest — newer batch supersedes older)
  let pendingBatch: {
    turns: TurnData[];
    thinking: string[];
    retrievedMemories: { id: string; text: string }[];
    priorExtractions?: PriorExtractions;
  } | null = null;

  async function processPending(): Promise<void> {
    if (processing || shuttingDown) return;
    while (pendingBatch) {
      processing = true;
      const batch = pendingBatch;
      pendingBatch = null;
      try {
        await runExtraction(batch.turns, batch.thinking, batch.retrievedMemories, batch.priorExtractions);
      } catch (e) {
        errorCount++;
        swallow.warn("daemon:extraction", e);
      } finally {
        processing = false;
      }
    }
  }

  return {
    sendTurnBatch(turns, thinking, retrievedMemories, priorExtractions) {
      if (shuttingDown) return;
      pendingBatch = { turns, thinking, retrievedMemories, priorExtractions };
      // Fire-and-forget: init if needed, then process
      ensureInit()
        .then(ok => { if (ok) return processPending(); })
        .catch(e => swallow.warn("daemon:sendBatch", e));
    },

    async getStatus() {
      return {
        type: "status" as const,
        extractedTurns: extractedTurnCount,
        pendingBatches: pendingBatch ? 1 : 0,
        errors: errorCount,
      };
    },

    async shutdown(timeoutMs = 45_000) {
      shuttingDown = true;
      // Wait for current extraction to finish
      if (processing) {
        await Promise.race([
          new Promise<void>(resolve => {
            const check = setInterval(() => {
              if (!processing) { clearInterval(check); resolve(); }
            }, 100);
          }),
          new Promise<void>(resolve => setTimeout(resolve, timeoutMs)),
        ]);
      }
      // Clean up daemon-local connections
      await Promise.allSettled([
        store?.dispose(),
        embeddings?.dispose(),
      ]).catch(() => {});
      store = null;
      embeddings = null;
    },

    getExtractedTurnCount() {
      return extractedTurnCount;
    },
  };
}
