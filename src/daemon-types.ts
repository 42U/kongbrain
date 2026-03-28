/**
 * Shared types for the memory daemon system.
 * Imported by both the worker thread (memory-daemon.ts) and the
 * main thread manager (daemon-manager.ts).
 */
import type { SurrealConfig, EmbeddingConfig } from "./config.js";

export interface TurnData {
  role: string;
  text: string;
  tool_name?: string;
  tool_result?: string;
  file_paths?: string[];
}

/** Data passed to the worker thread via workerData. */
export interface DaemonWorkerData {
  surrealConfig: SurrealConfig;
  embeddingConfig: EmbeddingConfig;
  sessionId: string;
  /** LLM provider name (resolved from OpenClaw config at daemon start). */
  llmProvider?: string;
  /** LLM model ID (resolved from OpenClaw config at daemon start). */
  llmModel?: string;
}

/** Previously extracted item names — for dedup across daemon runs. */
export interface PriorExtractions {
  conceptNames: string[];
  artifactPaths: string[];
  skillNames: string[];
}

/** Messages from main thread -> daemon worker. */
export type DaemonMessage =
  | {
      type: "turn_batch";
      turns: TurnData[];
      thinking: string[];
      retrievedMemories: { id: string; text: string }[];
      sessionId: string;
      priorExtractions?: PriorExtractions;
    }
  | { type: "shutdown" }
  | { type: "status_request" };

/** Messages from daemon worker -> main thread. */
export type DaemonResponse =
  | {
      type: "extraction_complete";
      extractedTurnCount: number;
      causalCount: number;
      monologueCount: number;
      resolvedCount: number;
      conceptCount: number;
      correctionCount: number;
      preferenceCount: number;
      artifactCount: number;
      decisionCount: number;
      skillCount: number;
      extractedNames?: PriorExtractions;
    }
  | { type: "status"; extractedTurns: number; pendingBatches: number; errors: number }
  | { type: "shutdown_complete" }
  | { type: "error"; message: string };
