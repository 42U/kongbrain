/**
 * Daemon Manager — spawns and manages the memory daemon worker thread.
 *
 * Provides a clean interface for sending turn batches, querying status,
 * and graceful shutdown. Used by the session lifecycle hooks.
 *
 * Ported from kongbrain — takes config as params instead of env globals.
 */
import { Worker } from "node:worker_threads";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { SurrealConfig, EmbeddingConfig } from "./config.js";
import type { DaemonMessage, DaemonResponse, DaemonWorkerData, TurnData, PriorExtractions } from "./daemon-types.js";
import { swallow } from "./errors.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

export type { TurnData } from "./daemon-types.js";

export interface MemoryDaemon {
  /** Fire-and-forget: send a batch of turns for incremental extraction. */
  sendTurnBatch(
    turns: TurnData[],
    thinking: string[],
    retrievedMemories: { id: string; text: string }[],
    priorExtractions?: PriorExtractions,
  ): void;
  /** Request current daemon status (async, waits for response). */
  getStatus(): Promise<DaemonResponse & { type: "status" }>;
  /** Graceful shutdown: waits for current extraction, then terminates. */
  shutdown(timeoutMs?: number): Promise<void>;
  /** Synchronous: how many turns has the daemon already extracted? */
  getExtractedTurnCount(): number;
}

export function startMemoryDaemon(
  surrealConfig: SurrealConfig,
  embeddingConfig: EmbeddingConfig,
  sessionId: string,
  llmConfig?: { provider?: string; model?: string },
): MemoryDaemon {
  const workerData: DaemonWorkerData = {
    surrealConfig,
    embeddingConfig,
    sessionId,
    llmProvider: llmConfig?.provider,
    llmModel: llmConfig?.model,
  };

  const worker = new Worker(join(__dirname, "memory-daemon.js"), { workerData });

  let extractedTurnCount = 0;
  let terminated = false;

  let pendingStatusResolve: ((resp: DaemonResponse & { type: "status" }) => void) | null = null;

  worker.on("message", (msg: DaemonResponse) => {
    switch (msg.type) {
      case "extraction_complete":
        extractedTurnCount = msg.extractedTurnCount;
        break;
      case "status":
        if (pendingStatusResolve) {
          pendingStatusResolve(msg as DaemonResponse & { type: "status" });
          pendingStatusResolve = null;
        }
        break;
      case "error":
        swallow.warn("daemon-manager:worker-error", new Error(msg.message));
        break;
    }
  });

  worker.on("error", (err) => {
    swallow.warn("daemon-manager:worker-thread-error", err);
  });

  worker.on("exit", (code) => {
    terminated = true;
    if (code !== 0) {
      swallow.warn("daemon-manager:worker-exit", new Error(`Daemon exited with code ${code}`));
    }
  });

  return {
    sendTurnBatch(turns, thinking, retrievedMemories, priorExtractions) {
      if (terminated) return;
      try {
        worker.postMessage({
          type: "turn_batch",
          turns,
          thinking,
          retrievedMemories,
          sessionId,
          priorExtractions,
        } satisfies DaemonMessage);
      } catch (e) { swallow.warn("daemon-manager:sendBatch", e); }
    },

    async getStatus() {
      if (terminated) return { type: "status" as const, extractedTurns: extractedTurnCount, pendingBatches: 0, errors: 0 };
      return new Promise<DaemonResponse & { type: "status" }>((resolve) => {
        const timer = setTimeout(() => {
          pendingStatusResolve = null;
          resolve({ type: "status", extractedTurns: extractedTurnCount, pendingBatches: -1, errors: -1 });
        }, 5000);
        pendingStatusResolve = (resp) => {
          clearTimeout(timer);
          resolve(resp);
        };
        worker.postMessage({ type: "status_request" } satisfies DaemonMessage);
      });
    },

    async shutdown(timeoutMs = 45_000) {
      if (terminated) return;
      return new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          worker.terminate().catch(() => {});
          terminated = true;
          resolve();
        }, timeoutMs);

        const onMessage = (msg: DaemonResponse) => {
          if (msg.type === "shutdown_complete") {
            clearTimeout(timer);
            worker.removeListener("message", onMessage);
            terminated = true;
            resolve();
          }
        };
        worker.on("message", onMessage);

        try {
          worker.postMessage({ type: "shutdown" } satisfies DaemonMessage);
        } catch {
          clearTimeout(timer);
          terminated = true;
          resolve();
        }
      });
    },

    getExtractedTurnCount() {
      return extractedTurnCount;
    },
  };
}
