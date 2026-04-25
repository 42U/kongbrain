import { homedir } from "node:os";
import { join } from "node:path";

export interface SurrealConfig {
  url: string;
  httpUrl: string;
  user: string;
  pass: string;
  ns: string;
  db: string;
}

export type EmbeddingProvider = "local" | "openai-compat";

export interface OpenAICompatEmbeddingConfig {
  /** Model name passed in the embeddings request body (e.g. "text-embedding-3-small"). */
  model: string;
  /** Endpoint base URL. Default: "https://api.openai.com/v1". */
  baseURL: string;
  /** Name of the env var holding the API key. Default: "OPENAI_API_KEY". */
  apiKeyEnv: string;
}

export interface EmbeddingConfig {
  /** Which provider to use. Default "local" (BGE-M3 via node-llama-cpp). */
  provider: EmbeddingProvider;
  /** Vector dimensionality the active provider should produce. */
  dimensions: number;
  /** Path to the local GGUF model — only consulted when provider === "local". */
  modelPath: string;
  /** OpenAI-compatible provider settings — only consulted when provider === "openai-compat". */
  openaiCompat: OpenAICompatEmbeddingConfig;
}

export interface ThresholdConfig {
  /** Tokens accumulated before daemon flushes extraction (default: 4000) */
  daemonTokenThreshold: number;
  /** Cumulative tokens before mid-session cleanup fires (default: 25000) */
  midSessionCleanupThreshold: number;
  /** Per-extraction timeout in ms (default: 60000) */
  extractionTimeoutMs: number;
  /** Max pending thinking blocks kept in memory (default: 20) */
  maxPendingThinking: number;
  /** Retrieval outcome samples needed before ACAN training (default: 5000) */
  acanTrainingThreshold: number;
}

export interface KongBrainConfig {
  surreal: SurrealConfig;
  embedding: EmbeddingConfig;
  thresholds: ThresholdConfig;
}

function parseEmbeddingConfig(raw: Record<string, unknown>): EmbeddingConfig {
  const openaiCompatRaw = (raw.openaiCompat ?? {}) as Record<string, unknown>;

  // Provider precedence: env var > plugin config > default "local"
  const rawProvider =
    process.env.KONGBRAIN_EMBED_PROVIDER ??
    (typeof raw.provider === "string" ? raw.provider : null);
  const provider: EmbeddingProvider =
    rawProvider === "openai-compat" ? "openai-compat" : "local";

  return {
    provider,
    dimensions: typeof raw.dimensions === "number" ? raw.dimensions : 1024,
    modelPath:
      process.env.EMBED_MODEL_PATH ??
      (typeof raw.modelPath === "string"
        ? raw.modelPath
        : join(homedir(), ".node-llama-cpp", "models", "bge-m3-q4_k_m.gguf")),
    openaiCompat: {
      model:
        typeof openaiCompatRaw.model === "string"
          ? openaiCompatRaw.model
          : "text-embedding-3-small",
      // baseURL: env wins (matches the official openai SDK convention)
      baseURL:
        process.env.OPENAI_BASE_URL ??
        (typeof openaiCompatRaw.baseURL === "string"
          ? openaiCompatRaw.baseURL
          : "https://api.openai.com/v1"),
      apiKeyEnv:
        typeof openaiCompatRaw.apiKeyEnv === "string"
          ? openaiCompatRaw.apiKeyEnv
          : "OPENAI_API_KEY",
    },
  };
}

/**
 * Parse plugin config from openclaw.plugin.json configSchema values,
 * with env var overrides and sensible defaults.
 */
export function parsePluginConfig(raw?: Record<string, unknown>): KongBrainConfig {
  const surreal = (raw?.surreal ?? {}) as Record<string, unknown>;
  const embedding = (raw?.embedding ?? {}) as Record<string, unknown>;
  const thresholds = (raw?.thresholds ?? {}) as Record<string, unknown>;

  // Priority: plugin config > env vars > defaults
  const url =
    (typeof surreal.url === "string" ? surreal.url : null) ??
    process.env.SURREAL_URL ??
    "ws://localhost:8042/rpc";

  return {
    surreal: {
      url,
      get httpUrl() {
        const override = (typeof surreal.httpUrl === "string" ? surreal.httpUrl : null) ??
          process.env.SURREAL_HTTP_URL;
        if (override) return override;
        return this.url
          .replace("ws://", "http://")
          .replace("wss://", "https://")
          .replace("/rpc", "/sql");
      },
      user: (typeof surreal.user === "string" ? surreal.user : null) ?? process.env.SURREAL_USER ?? "root",
      pass: (typeof surreal.pass === "string" ? surreal.pass : null) ?? process.env.SURREAL_PASS ?? "root",
      ns: (typeof surreal.ns === "string" ? surreal.ns : null) ?? process.env.SURREAL_NS ?? "kong",
      db: (typeof surreal.db === "string" ? surreal.db : null) ?? process.env.SURREAL_DB ?? "memory",
    },
    embedding: parseEmbeddingConfig(embedding),
    thresholds: {
      daemonTokenThreshold:
        typeof thresholds.daemonTokenThreshold === "number" ? thresholds.daemonTokenThreshold : 4000,
      midSessionCleanupThreshold:
        typeof thresholds.midSessionCleanupThreshold === "number" ? thresholds.midSessionCleanupThreshold : 25_000,
      extractionTimeoutMs:
        typeof thresholds.extractionTimeoutMs === "number" ? thresholds.extractionTimeoutMs : 60_000,
      maxPendingThinking:
        typeof thresholds.maxPendingThinking === "number" ? thresholds.maxPendingThinking : 20,
      acanTrainingThreshold:
        typeof thresholds.acanTrainingThreshold === "number" ? thresholds.acanTrainingThreshold : 5000,
    },
  };
}
